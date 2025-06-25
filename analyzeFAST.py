import concurrent.futures
import re
from typing import Dict, Optional, Tuple

import numpy as np
from freud.box import Box
from freud.density import RDF
from numba import jit, njit, prange  # noqa: F401

# ----------------------------------------------------------------------------
#                               I/O PART
# ----------------------------------------------------------------------------


def readGPUMDdump(
    filename: str, every: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legge un dump **GPUMD** in formato XYZ e restituisce *array NumPy*:

    * ``positions   : (n_frames, n_atoms, 3)`` – coordinate in PBC
    * ``velocities  : (n_frames, n_atoms, 3)`` – velocità
    * ``cell_vecs   : (n_frames, 3, 3)``     – vettori della cella
    * ``atom_types  : (n_atoms,)``            – 1 = O, 2 = H (costante su tutti i frame)

    Parametri
    ----------
    filename : str
        Percorso al file ``.xyz`` prodotto da GPUMD.
    every : int, default 1
        Sub‑campionamento dei frame (1 = tutti, 2 = uno su due, …).
    """

    pos_frames = []  # list[np.ndarray]
    vel_frames = []
    cell_vecs = np.empty((3, 3), dtype=np.float64)  # inizializza vuoto
    atom_types = None  # np.ndarray una volta sola
    lattice_re = re.compile(r'Lattice="([0-9.eE+\- ]+)"')

    with open(filename, "r", encoding="utf-8") as fh:
        frame = 0
        while True:
            first = fh.readline()
            if not first:
                break  # EOF
            n_atoms = int(first)
            header = fh.readline()
            m = lattice_re.search(header)
            if m is None:
                raise ValueError("Header privo di campo Lattice=…")
            if frame == 0:
                cell_vecs = np.fromstring(m.group(1), sep=" ").reshape(3, 3)

            # skip frame se fuori da subsampling
            if (frame % every) != 0:
                for _ in range(n_atoms):
                    fh.readline()
                frame += 1
                continue

            # --- Legge tutte le righe degli atomi
            lines = [fh.readline() for _ in range(n_atoms)]

            # 1) calcola atom_types solo al primissimo frame
            if atom_types is None:
                first_chars = b"".join(line[0].lower().encode() for line in lines)
                t = np.frombuffer(first_chars, dtype="S1")
                atom_types = np.where(t == b"o", 1, 2).astype(np.int8)

            # 2) rimuove il simbolo atomico e concatena il resto
            data_str = "".join(line.split(" ", 1)[1] for line in lines)
            nums = np.fromstring(data_str, sep=" ").reshape(n_atoms, -1)

            pos_frames.append(nums[:, 0:3].astype(np.float64, copy=False))
            vel_frames.append(nums[:, 4:7].astype(np.float64, copy=False))
            frame += 1

    # Empila in array 3‑D
    positions = np.stack(pos_frames)
    velocities = np.stack(vel_frames)

    return positions, velocities, cell_vecs, atom_types


def writeGPUMDdump(
    filename: str,
    positions: np.ndarray,
    atom_types: np.ndarray,
    cell_vectors: Optional[np.ndarray] = None,
    comment_prefix: str = "",
) -> None:
    """
    Salva una traiettoria atomica in un file in formato XYZ.

    Parametri
    ----------
    filename : str
        Percorso del file .xyz da scrivere.
    positions : np.ndarray
        Array delle coordinate con forma (n_frames, n_atoms, 3).
    atom_types : np.ndarray
        Array dei tipi di atomi (1=O, 2=H) con forma (n_atoms,).
    cell_vectors : np.ndarray, optional
        Matrice (3, 3) dei vettori di cella. Se fornita, viene scritta
        nell'header di ogni frame in un formato leggibile da VMD/Ovito.
    comment_prefix : str, optional
        Una stringa da aggiungere all'inizio della riga di commento
        di ogni frame (es. per specificare se le coordinate sono "wrapped" o "unwrapped").
    """
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("L'array 'positions' deve avere forma (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions.shape

    # Mappa i tipi numerici (1, 2) ai simboli atomici ("O", "H")
    type_to_symbol = {1: "O", 2: "H"}
    symbols = [type_to_symbol.get(t, "X") for t in atom_types]

    print(f"Scrittura di {n_frames} frame su file '{filename}'...")

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n_frames):
            # --- Riga 1: Numero di atomi ---
            f.write(f"{n_atoms}\n")

            # --- Riga 2: Commento ---
            # Costruiamo una riga di commento informativa
            comment = f"frame={i}"
            if comment_prefix:
                comment = f"{comment_prefix} {comment}"

            # Aggiungi i vettori di cella nel formato "Lattice" riconosciuto da VMD/Ovito
            if cell_vectors is not None:
                # Appiattisci la matrice 3x3 in una stringa di 9 numeri
                lattice_str = " ".join(map(str, cell_vectors.flatten()))
                comment += f' Lattice="{lattice_str}"'

            f.write(f"{comment}\n")

            # --- Righe degli atomi ---
            for j in range(n_atoms):
                symbol = symbols[j]
                x, y, z = positions[i, j]
                # Scrivi simbolo e coordinate con formattazione allineata
                f.write(f"{symbol:<2} {x:>15.8f} {y:>15.8f} {z:>15.8f}\n")


# Costanti utili (possono essere definite globalmente o passate/definite localmente)
TWO_PI = 2.0 * np.pi
_ATOMIC_MASSES = np.array(
    [0.0, 15.999, 1.008], dtype=np.float64
)  # Fattore per convertire frequenza (in ps^-1 o THz) in wavenumbers (cm^-1)
# 1 / (c_luce_cm_per_s * 1e-12 s/ps) = 1 / (2.99792458e10 * 1e-12)
HZ_TO_CMINV_FACTOR = 33.3564095198152


def _prepare_vel_mass(
    velocities: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalizza velocities a 3‑D e produce la matrice delle masse.

    *Se* atom_types non è fornito, o il suo numero non coincide con n_atoms,
    viene creato un array di '1' (Ossigeno) di lunghezza n_atoms.
    """
    n_frames, n_atoms, _ = velocities.shape

    if atom_types is None or len(atom_types) != n_atoms:
        atom_types = np.ones(n_atoms, dtype=np.int8)  # default: tutti Ossigeno

    mass_per_atom = _ATOMIC_MASSES[atom_types]  # (n_atoms,)
    masses = np.broadcast_to(mass_per_atom, (n_frames, n_atoms))  # (n_frames,n_atoms)
    return velocities, masses


def _calculate_autocorr(x: np.ndarray, n: int) -> np.ndarray:
    return np.correlate(x, x, mode="full")[-n:]


# ---------------------------------------------------------------------------
# VACF — variante "NP"
# ---------------------------------------------------------------------------


def calculateVACFnp(
    velocities: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    max_correlation_len: Optional[int] = None,
    mass_weighted: bool = False,
) -> np.ndarray:
    """VACF in stile *NP*.

    Accetta sia per‑atomo che singolo vettore per frame.
    """
    velocities, masses = _prepare_vel_mass(velocities, atom_types)

    n_frames, n_atoms, _ = velocities.shape
    corr_len = max_correlation_len or (n_frames - 1)
    if corr_len > n_frames:
        raise ValueError("max_correlation_len > n_frames")

    vel_tr = np.transpose(velocities, (1, 2, 0))  # (n_atoms,3,n_frames)

    C_t = np.zeros(corr_len, dtype=np.float64)
    N_split = 0

    lbl = "mass-weighted " if mass_weighted else ""
    print(f"Calculating {lbl}VACF (NP) for n_atoms={n_atoms}, steps={corr_len}.")

    for t in range(corr_len, n_frames + 1, corr_len):
        blk = slice(t - corr_len, t)
        for j in range(n_atoms):
            for k in range(3):
                v_series = vel_tr[j, k, blk]
                c_jk = _calculate_autocorr(v_series, corr_len)
                mass = masses[blk, j] if mass_weighted else 1.0
                C_t += c_jk * mass
        N_split += 1

    C_t /= N_split
    C_t /= C_t[0]
    return C_t


def calculateVACFnp_groups(
    velocities: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    max_correlation_len: Optional[int] = None,
    mass_weighted: bool = False,
    group: str = "all",
    index_file: str = "model-indexes.dat",
) -> np.ndarray:
    """VACF in stile *NP* con selezione del gruppo di atomi.

    Accetta sia per‑atomo che singolo vettore per frame.

    group:
      - "all": tutti gli atomi (default)
      - "exc" : solo atomi eccitati (da index_file, 1-based)
      - "norm": tutti gli altri atomi
    """
    # 1) carico indici eccitati se necessario (convertendo da 1-based a 0-based)
    n_frames, n_atoms, _ = velocities.shape
    if group in ("exc", "norm"):
        exc_idxs = np.loadtxt(index_file, dtype=int) - 1
        all_idxs = np.arange(n_atoms, dtype=int)
        if group == "exc":
            sel = exc_idxs
        else:  # norm
            sel = np.setdiff1d(all_idxs, exc_idxs)
        velocities = velocities[:, sel, :]
        if atom_types is not None:
            atom_types = atom_types[sel]
        n_atoms = velocities.shape[1]

    velocities, masses = _prepare_vel_mass(velocities, atom_types)

    n_frames, n_atoms, _ = velocities.shape
    corr_len = max_correlation_len or (n_frames - 1)
    if corr_len > n_frames:
        raise ValueError("max_correlation_len > n_frames")

    vel_tr = np.transpose(velocities, (1, 2, 0))  # (n_atoms,3,n_frames)

    C_t = np.zeros(corr_len, dtype=np.float64)
    N_split = 0

    lbl = "mass-weighted " if mass_weighted else ""
    print(
        f"Calculating {lbl}VACF (NP) for group='{group}', n_atoms={n_atoms}, steps={corr_len}."
    )

    for t in range(corr_len, n_frames + 1, corr_len):
        blk = slice(t - corr_len, t)
        for j in range(n_atoms):
            for k in range(3):
                v_series = vel_tr[j, k, blk]
                c_jk = _calculate_autocorr(v_series, corr_len)
                mass = masses[blk, j] if mass_weighted else 1.0
                C_t += c_jk * mass
        N_split += 1

    C_t /= N_split
    C_t /= C_t[0]
    return C_t


# ---------------------------------------------------------------------------
# VACF — variante "LMP" (LAMMPS‑like)
# ---------------------------------------------------------------------------


def calculateVACFlmp(
    velocities: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    max_correlation_len: Optional[int] = None,
    mass_weighted: bool = False,
    norm: bool = True,
) -> np.ndarray:
    """VACF in stile *LAMMPS*.

    Funziona anche con un singolo vettore per frame (n_atoms = 1).
    """
    velocities, masses = _prepare_vel_mass(velocities, atom_types)
    n_frames, n_atoms, _ = velocities.shape

    corr_len = max_correlation_len or (n_frames - 1)

    v0 = velocities[0]
    vacf = np.zeros(corr_len, dtype=np.float64)

    lbl = "mass-weighted " if mass_weighted else ""
    print(f"Calculating {lbl}VACF (LMP) for n_atoms={n_atoms}, steps={corr_len}.")

    if mass_weighted:
        m0 = masses[0]
        m_sum = np.sum(m0)
        for lag in range(corr_len):
            vt = velocities[lag]
            dot = np.sum(v0 * vt, axis=1)
            vacf[lag] = np.sum(m0 * dot) / m_sum
    else:
        for lag in range(corr_len):
            vt = velocities[lag]
            dot = np.sum(v0 * vt, axis=1)
            vacf[lag] = np.mean(dot)

    if norm:
        vacf /= vacf[0]

    return vacf


def calculateVACFlmp_groups(
    velocities: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    max_correlation_len: Optional[int] = None,
    mass_weighted: bool = False,
    norm: bool = True,
    group: str = "all",
    index_file: str = "model-indexes.dat",
) -> np.ndarray:
    """
    VACF in stile LAMMPS, con selezione del gruppo di atomi.

    group:
      - "all": tutti gli atomi (default)
      - "exc" : solo atomi eccitati (da index_file, 1-based)
      - "norm": tutti gli altri atomi
    """
    # 1) carico indici eccitati se necessario (convertendo da 1-based a 0-based)
    n_frames, n_atoms, _ = velocities.shape
    if group in ("exc", "norm"):
        exc_idxs = np.loadtxt(index_file, dtype=int) - 1
        all_idxs = np.arange(n_atoms, dtype=int)
        if group == "exc":
            sel = exc_idxs
        else:  # norm
            sel = np.setdiff1d(all_idxs, exc_idxs)
        velocities = velocities[:, sel, :]
        if atom_types is not None:
            atom_types = atom_types[sel]
        n_atoms = velocities.shape[1]

    # 2) preparo velocità e masse
    velocities, masses = _prepare_vel_mass(velocities, atom_types)

    # 3) parametri per il VACF
    corr_len = max_correlation_len or (n_frames - 1)
    v0 = velocities[0]
    vacf = np.zeros(corr_len, dtype=np.float64)
    lbl = "mass-weighted " if mass_weighted else ""
    print(
        f"Calculating {lbl}VACF for group='{group}', n_atoms={n_atoms}, steps={corr_len}."
    )

    # 4) calcolo VACF
    if mass_weighted:
        m0 = masses[0]
        m_sum = np.sum(m0)
        for lag in range(corr_len):
            vt = velocities[lag]
            dot = np.sum(v0 * vt, axis=1)
            vacf[lag] = np.sum(m0 * dot) / m_sum
    else:
        for lag in range(corr_len):
            vt = velocities[lag]
            dot = np.sum(v0 * vt, axis=1)
            vacf[lag] = np.mean(dot)

    # 5) normalizzazione
    if norm and vacf[0] != 0:
        vacf /= vacf[0]

    return vacf


@njit(cache=True)  # Aggiunto cache=True per riutilizzare la compilazione
def _filon_cosine_transform_subroutine(
    dt_val: float,
    delta_omega_val: float,  # Questo è il DOM del Fortran, interpretato come Delta_Omega
    nmax_intervals: int,  # NMAX nel Fortran (numero di intervalli, deve essere pari)
    c_corr_func: np.ndarray,  # Funzione di correlazione C(t), array da 0 a nmax_intervals
    chat_spectrum: np.ndarray,  # Array di output per lo spettro CHAT(omega)
) -> None:
    """
    Replica Python della subroutine FILONC per calcolare la trasformata coseno
    di Fourier usando il metodo di Filon.

    Args:
        dt_val: Intervallo di tempo tra i punti in c_corr_func.
        delta_omega_val: Intervallo di frequenza angolare per chat_spectrum.
                         (omega_nu = nu * delta_omega_val).
        nmax_intervals: Numero di intervalli sull'asse del tempo. Deve essere pari.
                        c_corr_func deve avere nmax_intervals + 1 punti.
        c_corr_func: Array NumPy 1D della funzione di correlazione C(t).
        chat_spectrum: Array NumPy 1D (output, modificato in-place) per lo spettro.
    """
    if nmax_intervals % 2 != 0:
        raise ValueError(
            "NMAX (nmax_intervals) deve essere pari per il metodo di Filon."
        )
    if len(c_corr_func) != nmax_intervals + 1:
        raise ValueError(
            f"Lunghezza di c_corr_func ({len(c_corr_func)}) non corrisponde a "
            f"nmax_intervals + 1 ({nmax_intervals + 1})."
        )
    if len(chat_spectrum) != nmax_intervals + 1:
        raise ValueError(
            "chat_spectrum deve avere la stessa dimensione di c_corr_func."
        )

    t_max = float(nmax_intervals) * dt_val

    for nu_idx in range(nmax_intervals + 1):  # Loop su NU da 0 a NMAX
        omega = float(nu_idx) * delta_omega_val  # Frequenza angolare corrente
        theta = omega * dt_val  # Argomento adimensionale omega*dt

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if (
            abs(theta) < 1e-9
        ):  # Caso speciale per theta ~ 0 (per evitare divisione per zero)
            alpha = 0.0
            beta = 2.0 / 3.0
            gamma = 4.0 / 3.0
        else:
            th_sq = theta * theta
            th_cub = th_sq * theta
            # Parametri di Filon
            alpha = (1.0 / th_cub) * (
                th_sq + theta * sin_theta * cos_theta - 2.0 * sin_theta**2
            )
            beta = (2.0 / th_cub) * (
                theta * (1.0 + cos_theta**2) - 2.0 * sin_theta * cos_theta
            )
            gamma = (4.0 / th_cub) * (sin_theta - theta * cos_theta)

        # Somma sui termini con indice pari (CE)
        ce_sum = 0.0
        for tau_idx in range(0, nmax_intervals + 1, 2):  # tau = 0, 2, ..., NMAX
            ce_sum += c_corr_func[tau_idx] * np.cos(theta * float(tau_idx))

        # Sottrai metà del primo e ultimo termine (correzione per la regola di Filon/Simpson)
        ce_sum -= 0.5 * (
            c_corr_func[0] * np.cos(theta * 0.0)  # np.cos(0)=1
            + c_corr_func[nmax_intervals] * np.cos(theta * float(nmax_intervals))
        )
        # Nota: theta * nmax_intervals = omega * dt * nmax_intervals = omega * t_max

        # Somma sui termini con indice dispari (CO)
        co_sum = 0.0
        for tau_idx in range(1, nmax_intervals, 2):  # tau = 1, 3, ..., NMAX-1
            co_sum += c_corr_func[tau_idx] * np.cos(theta * float(tau_idx))

        # Calcola il valore dello spettro CHAT(NU)
        term_alpha_component = (
            alpha * c_corr_func[nmax_intervals] * np.sin(omega * t_max)
        )
        chat_spectrum[nu_idx] = (
            2.0 * (term_alpha_component + beta * ce_sum + gamma * co_sum) * dt_val
        )


def calculateVDOS(
    time_points: np.ndarray,
    corr_values: np.ndarray,
    norm: bool = True,
    gaussian_filter_width: Optional[float] = None,
    output_in_wavenumbers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Processa una funzione di correlazione temporale per ottenere il suo spettro di frequenza,
    replicando la logica del programma Fortran 'ft', incluso il metodo di Filon.

    Args:
        time_points: Array NumPy 1D dei punti temporali. Si assume dt costante.
        corr_values: Array NumPy 1D dei valori della funzione di correlazione.
        norm: (Opzionale) Se True, normalizza i valori della funzione di correlazione ad integrale = 1
        gaussian_filter_width: (Opzionale) Parametro 'width' per il filtro Gaussiano.
                               Se None, nessun filtro viene applicato.
        output_in_wavenumbers: (Opzionale) Se True (default), l'asse delle frequenze
                               dell'output è in cm^-1. Altrimenti, è in unità di
                               frequenza angolare (rad / unità di tempo di dt).

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            - frequencies_axis: Array delle frequenze (in cm^-1 o rad/unità di tempo).
            - spectrum: Array dello spettro calcolato (CHAT).
            - filter_info: (Opzionale) Se il filtro è applicato, una tupla contenente
                             (tempo_filtrato, valori_corr_filtrati, finestra_usata).
                             Altrimenti None.
    """
    if len(time_points) != len(corr_values):
        raise ValueError("time_points e corr_values devono avere la stessa lunghezza.")
    if (
        len(time_points) < 2
    ):  # Per Filon (NMAX pari) serve NMAX>=2, quindi almeno 3 punti.
        # Se NMAX=0 (1 punto), tmax=0, dom=inf.
        # Se NMAX<2, non ha senso.
        print(
            "Attenzione: Servono almeno 3 punti dati (2 intervalli) per il metodo di Filon "
            "come implementato. Se meno, si restituiscono array vuoti."
        )
        return np.array([]), np.array([]), None

    dt = time_points[1] - time_points[0]
    # Verifica grossolana della costanza di dt (opzionale ma buona pratica)
    # if not np.allclose(np.diff(time_points), dt):
    #     print("Attenzione: dt rilevato non costante. La precisione potrebbe essere affetta. Si usa il primo dt.")

    num_points_original = len(corr_values)

    # NMAX nel Fortran è il numero di INTERVALLI, e deve essere pari.
    # Se num_points_original è il numero di punti, n_intervals_initial = num_points_original - 1.
    nmax_intervals = num_points_original - 1

    # Lavora su una copia dei valori di correlazione
    current_corr_for_ft = np.copy(corr_values)

    if nmax_intervals < 0:  # Caso di 0 o 1 punto originale
        print("Non abbastanza punti per formare intervalli.")
        return np.array([]), np.array([]), None

    if nmax_intervals % 2 != 0:
        nmax_intervals -= 1  # Rendi nmax_intervals pari, scartando l'ultimo intervallo (e l'ultimo punto)
        num_points_to_use = nmax_intervals + 1
        current_corr_for_ft = current_corr_for_ft[:num_points_to_use]
        time_points_to_use = time_points[:num_points_to_use]
        print(
            f"Numero di intervalli reso pari: {nmax_intervals}. "
            f"Dati troncati a {num_points_to_use} punti."
        )
    else:
        num_points_to_use = nmax_intervals + 1
        time_points_to_use = (
            time_points  # Nessun troncamento necessario oltre alla parità di NMAX
        )

    if (
        nmax_intervals < 2 and nmax_intervals != 0
    ):  # Filon richiede NMAX pari >= 2, se non è NMAX=0 (1 punto, 0 intervalli)
        # Il caso NMAX=0 (1 punto) viene gestito da Filon (theta=0)
        if nmax_intervals == 0 and num_points_to_use == 1:
            pass  # NMAX=0 (1 punto) è un caso limite che Filon gestisce
        else:
            print(
                f"Attenzione: NMAX={nmax_intervals} (dopo aggiustamento parità) non è sufficiente per la procedura "
                "standard di Filon (richiede NMAX >= 2 o NMAX=0)."
            )
            return np.array([]), np.array([]), None

    t_max = float(nmax_intervals) * dt
    if (
        t_max == 0 and nmax_intervals > 0
    ):  # Dovrebbe accadere solo se dt=0, che è un problema
        raise ValueError(
            "t_max è zero con nmax_intervals > 0, implica dt=0, il che non è valido."
        )

    # DOM nel Fortran è interpretato come Delta_Omega per la subroutine FILONC.
    # delta_omega_initial = 1.0 / t_max # Questa era la riga del Fortran per DOM
    # Se t_max è 0 (caso NMAX=0, un solo punto), 1.0/t_max è infinito.
    # Per NMAX=0, t_max=0. DOM non ha molto senso, omega sarà sempre 0*DOM=0.
    # La subroutine Filon gestisce theta=0.
    if t_max > 0:
        delta_omega_for_filon = 1.0 / t_max
    else:  # Caso NMAX=0, un solo punto
        delta_omega_for_filon = 0.0  # O qualsiasi valore, omega sarà 0.

    print("\nParametri per la trasformata di Filon:")
    print(f"  Numero di punti usati (NMAX+1): {num_points_to_use}")
    print(f"  Numero di intervalli (NMAX): {nmax_intervals}")
    print(f"  dt   = {dt:.4e}")
    print(f"  tmax = {t_max:.4e}")
    print(
        f"  Delta Omega (DOM per Filon) = {delta_omega_for_filon:.4e} (rad/unità di tempo)"
    )

    filter_info_to_return = None
    if (
        gaussian_filter_width is not None and nmax_intervals > 0
    ):  # Filtro non ha senso per NMAX=0
        print(
            f"Applicazione del filtro Gaussiano con width_param = {gaussian_filter_width}..."
        )
        window_gauss = np.zeros(num_points_to_use)

        for i_point_idx in range(
            num_points_to_use
        ):  # i_point_idx da 0 a nmax_intervals
            # Fortran: i (1-based) da 1 a nmax (che è n_points_to_use nel mio codice Python)
            # DBLE(i-1) corrisponde a float(i_point_idx)
            # Il termine nel Fortran: (0.5d0 * width * DBLE(i-1)*dt / tmax)
            # Diventa: (0.5 * gaussian_filter_width * float(i_point_idx) * dt / t_max)
            #          = (0.5 * gaussian_filter_width * float(i_point_idx) / nmax_intervals)
            # Chiamiamo questo 'x_arg'
            x_arg_for_filter = (
                0.5 * gaussian_filter_width * float(i_point_idx) / float(nmax_intervals)
            )
            window_gauss[i_point_idx] = np.exp(-0.5 * x_arg_for_filter**2)
            current_corr_for_ft[i_point_idx] *= window_gauss[i_point_idx]

        print("  Finestra Gaussiana applicata.")
        filter_info_to_return = (
            time_points_to_use,
            np.copy(current_corr_for_ft),
            window_gauss,
        )
    elif gaussian_filter_width is not None and nmax_intervals == 0:
        print("  Filtro Gaussiano non applicato: NMAX=0 (un solo punto dati).")

    spectrum_chat_array = np.zeros(num_points_to_use)  # Lunghezza NMAX_intervals + 1

    _filon_cosine_transform_subroutine(
        dt_val=dt,
        delta_omega_val=delta_omega_for_filon,
        nmax_intervals=nmax_intervals,
        c_corr_func=current_corr_for_ft,
        chat_spectrum=spectrum_chat_array,
    )

    # Conversione finale dell'asse delle frequenze
    # Il 'dom' iniziale del Fortran era delta_omega_for_filon = 1.0 / t_max
    # L'asse delle frequenze per l'output è i * dom_converted
    # dom_converted = dom_initial / TWO_PI * HZ_TO_CMINV_FACTOR
    if output_in_wavenumbers:
        if t_max > 0:  # Evita divisione per zero se t_max è 0 (NMAX=0)
            # Frequenza lineare step: (Delta Omega) / 2pi
            linear_freq_step = delta_omega_for_filon / TWO_PI
            # Wavenumber step
            wavenumber_step = linear_freq_step * HZ_TO_CMINV_FACTOR
        else:  # NMAX=0, t_max=0. Solo la componente DC (nu=0) è significativa.
            # Per evitare NaN/Inf, impostiamo lo step a 0 ma l'asse avrà solo il punto a 0.
            wavenumber_step = 0.0

        frequencies_axis_final = np.arange(num_points_to_use) * wavenumber_step
        print(
            f"  Asse delle frequenze convertito in cm^-1 (step: {wavenumber_step:.4e} cm^-1)."
        )
    else:
        # Restituisci frequenze angolari omega = nu * delta_omega_for_filon
        frequencies_axis_final = np.arange(num_points_to_use) * delta_omega_for_filon
        print(
            f"  Asse delle frequenze in rad/unità_tempo (step: {delta_omega_for_filon:.4e})."
        )

        if norm:
            from scipy.integrate import simpson

            spectrum_chat_array /= simpson(
                y=spectrum_chat_array, x=frequencies_axis_final
            )

    return frequencies_axis_final, spectrum_chat_array, filter_info_to_return


# --- OTTIMIZZATO CON NUMBA ---
# Questa funzione è corretta e viene chiamata dalla funzione Numba sottostante.
@njit(cache=True)  # Aggiunto cache=True per riutilizzare la compilazione
def _unwrap_one_particle(pos: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    n_frames = pos.shape[0]
    unwrapped = np.empty_like(pos)
    unwrapped[0] = pos[0]
    half = cell_lengths / 2.0
    for f in range(1, n_frames):
        delta = pos[f] - pos[f - 1]
        for k in range(3):
            if delta[k] > half[k]:
                delta[k] -= cell_lengths[k]
            elif delta[k] < -half[k]:
                delta[k] += cell_lengths[k]
        unwrapped[f] = unwrapped[f - 1] + delta
    return unwrapped


# --- OTTIMIZZATO CON NUMBA (versione interna) ---
# Questa è la nuova funzione interna che contiene solo codice compilabile da Numba.
# Accetta `cell_lengths` (1D) invece di `cell_vectors` (2D).
@njit(cache=True)  # Aggiunto cache=True per riutilizzare la compilazione
def _calculate_unwrapped_coords_fast_numba(
    positions: np.ndarray,
    atom_types: np.ndarray,
    cell_lengths: np.ndarray,  # <-- NOTA: accetta l'array 1D
    atoms_per_molecule: int,
) -> np.ndarray:
    n_frames, n_atoms, _ = positions.shape
    n_mols = n_atoms // atoms_per_molecule

    unwrapped = np.empty_like(positions)

    # Step 1 – unwrappa gli Ossigeni
    for mol_idx in range(n_mols):
        o_idx = mol_idx * atoms_per_molecule
        unwrapped[:, o_idx, :] = _unwrap_one_particle(
            positions[:, o_idx, :], cell_lengths
        )

    # Step 2 – unwrappa gli H relativi all'O
    half_cell = cell_lengths / 2.0
    for f in range(n_frames):
        for mol_idx in range(n_mols):
            o_idx = mol_idx * atoms_per_molecule
            o_unw = unwrapped[f, o_idx]
            o_pbc = positions[f, o_idx]
            for offset in range(1, atoms_per_molecule):
                a_idx = o_idx + offset
                vec = positions[f, a_idx] - o_pbc
                for k in range(3):
                    if vec[k] > half_cell[k]:
                        vec[k] -= cell_lengths[k]
                    elif vec[k] < -half_cell[k]:
                        vec[k] += cell_lengths[k]
                unwrapped[f, a_idx] = o_unw + vec
    return unwrapped


# --- Wrapper per l'utente (NON compilata) ---
# Questa è la funzione che chiami tu. Prepara i dati e invoca la versione compilata.
def unwrapp_coords(
    positions: np.ndarray,
    atom_types: np.ndarray,
    cell_vectors: np.ndarray,
    atoms_per_molecule: int = 3,
) -> np.ndarray:
    """
    Calcola le coordinate unwrapped (wrapper per la versione compilata con Numba).

    Questa funzione esegue l'operazione `np.diagonal` non supportata da Numba
    e poi chiama una funzione interna JIT-compilata per i calcoli intensivi.
    """
    if cell_vectors.shape != (3, 3):
        raise ValueError("cell_vectors deve avere shape (3, 3)")

    # 1. Esegui l'operazione non supportata qui, in puro Python
    cell_lengths = np.diagonal(cell_vectors)

    # 2. Chiama la funzione veloce e compilata con i dati pronti
    return _calculate_unwrapped_coords_fast_numba(
        positions, atom_types, cell_lengths, atoms_per_molecule
    )


@njit(cache=True)
def _rdf_core_loop_per_frame(
    positions: np.ndarray,
    atom_types: np.ndarray,
    box_dims: np.ndarray,
    r_max: float,
    dr: float,
    type1: int,
    type2: int,
    hist_all_frames: np.ndarray,
) -> None:
    """
    Core loop per RDF, calcolando un istogramma per ogni frame.
    Questa versione è CORRETTA per tutti i tipi di coppie.
    """
    n_frames, n_atoms, _ = positions.shape
    n_bins = hist_all_frames.shape[1]
    r_max_sq = r_max * r_max

    # Loop su tutti i frame
    for i in range(n_frames):
        # Loop su ogni coppia unica di atomi (j, k) con k > j
        for j in range(n_atoms):
            type_j = atom_types[j]
            for k in range(j + 1, n_atoms):
                type_k = atom_types[k]

                # Controlla se la coppia (j, k) corrisponde alla coppia desiderata (type1, type2)
                # in entrambi gli ordini.
                is_matching_pair = (type_j == type1 and type_k == type2) or (
                    type_j == type2 and type_k == type1
                )

                if is_matching_pair:
                    # Calcola la distanza usando la Minimum Image Convention
                    d_vec = positions[i, j] - positions[i, k]
                    d_vec -= box_dims * np.rint(d_vec / box_dims)
                    dist_sq = d_vec[0] ** 2 + d_vec[1] ** 2 + d_vec[2] ** 2

                    if dist_sq < r_max_sq:
                        dist = np.sqrt(dist_sq)
                        bin_idx = int(dist / dr)
                        if bin_idx < n_bins:
                            # Ogni coppia unica trovata viene contata una volta.
                            hist_all_frames[i, bin_idx] += 1


def calculateRDF(
    positions: np.ndarray,
    atom_types: np.ndarray,
    cell_vectors: np.ndarray,
    pair: Tuple[str, str],
    r_max: float,
    dr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la Radial Distribution Function (RDF), g(r), per ogni singolo frame.
    Questa funzione wrapper è CORRETTA e non necessita di modifiche.
    """
    print(f"Avvio calcolo RDF per la coppia {pair} (frame per frame)...")

    # --- 1. Preparazione ---
    box_dims = np.diagonal(cell_vectors)
    volume = np.prod(box_dims)
    type_map = {"O": 1, "H": 2}  # Convenzione O=1, H=2
    type1 = type_map[pair[0].upper()]
    type2 = type_map[pair[1].upper()]

    n_frames, n_atoms, _ = positions.shape
    n1_atoms = np.sum(atom_types == type1)
    n2_atoms = np.sum(atom_types == type2)

    if n1_atoms == 0 or n2_atoms == 0:
        return np.array([]), np.array([])

    # --- 2. Esecuzione del loop Numba ---
    n_bins = int(r_max / dr)
    hist_all_frames = np.zeros((n_frames, n_bins), dtype=np.int64)

    _rdf_core_loop_per_frame(
        positions, atom_types, box_dims, r_max, dr, type1, type2, hist_all_frames
    )

    # --- 3. Normalizzazione ---
    r = (np.arange(n_bins) + 0.5) * dr
    shell_volumes = 4.0 * np.pi * r**2 * dr

    if type1 == type2:
        # Per coppie dello stesso tipo (es. O-O), il fattore di normalizzazione è N(N-1)/2
        if n1_atoms < 2:
            return r, np.zeros_like(shell_volumes)
        n_pairs = n1_atoms * (n1_atoms - 1) / 2.0
    else:
        # Per coppie di tipo diverso (es. O-H), il fattore è N_O * N_H
        n_pairs = n1_atoms * n2_atoms

    pair_density = n_pairs / volume

    # Conteggio ideale in un gas per un singolo frame
    ideal_gas_counts = pair_density * shell_volumes

    # Normalizza ogni riga (frame) dell'istogramma 2D
    g_r_all_frames = np.zeros_like(hist_all_frames, dtype=np.float64)
    # Evita la divisione per zero per i bin con volume nullo (es. r=0)
    non_zero = ideal_gas_counts > 1e-9

    # NumPy broadcasting: divide ogni riga di hist_all_frames per l'array 1D ideal_gas_counts
    g_r_all_frames[:, non_zero] = (
        hist_all_frames[:, non_zero] / ideal_gas_counts[non_zero]
    )

    print("Calcolo RDF per frame completato.")
    return r, g_r_all_frames


def calculateRDFfreud(
    positions: np.ndarray,
    atom_types: np.ndarray,
    cell_vectors: np.ndarray,
    pair: Tuple[str, str],
    r_max: float = 5.0,
    dr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la Radial Distribution Function (RDF), g(r), per ogni frame di una
    traiettoria utilizzando la libreria freud.

    Parametri
    ----------
    positions : np.ndarray
        Array delle coordinate con forma (n_frames, n_atoms, 3).
        Le coordinate devono essere "wrapped" (all'interno del box).
    atom_types : np.ndarray
        Array dei tipi atomici (1=O, 2=H) con forma (n_atoms,).
    cell_vectors : np.ndarray
        Matrice (3, 3) dei vettori di cella, costante per tutta la traiettoria.
    pair : Tuple[str, str]
        La coppia di atomi per cui calcolare la RDF (es. ('O', 'O'), ('O', 'H')).
        L'ordine non conta.
    bins : int, default 500
        Il numero di bin per l'istogramma della RDF.
    r_max : float, default 5.0
        La massima distanza (r) da considerare per il calcolo.

    Ritorna
    -------
    Tuple[np.ndarray, np.ndarray]
        Una tupla contenente:
        - r: i centri dei bin di distanza (asse x), forma (bins,).
        - rdf_per_frame: i valori della g(r) per ogni frame,
                         forma (n_frames, bins).
    """
    print(f"Avvio calcolo RDF con freud per la coppia {pair}...")

    # --- 1. Preparazione (eseguita una sola volta) ---
    n_frames, n_atoms, _ = positions.shape
    bins = int(r_max / dr)

    # Crea l'oggetto Box di freud dalla matrice di cella
    box = Box.from_matrix(cell_vectors)

    # Mappa i nomi degli atomi agli indici numerici (1=O, 2=H)
    type_map = {"O": 1, "H": 2}
    try:
        type1_str, type2_str = pair
        type1 = type_map[type1_str.upper()]
        type2 = type_map[type2_str.upper()]
    except KeyError:
        raise ValueError(
            f"Tipo di atomo non valido nella coppia {pair}. Usare 'O' o 'H'."
        )

    # Trova gli indici degli atomi che ci interessano
    indices1 = np.where(atom_types == type1)[0]
    indices2 = np.where(atom_types == type2)[0]

    if len(indices1) == 0 or len(indices2) == 0:
        print(
            f"Attenzione: nessun atomo trovato per la coppia {pair}. Restituisco array vuoti."
        )
        return np.array([]), np.array([])

    # --- 1.5. Preallocazione array RDF ---
    rdf_per_frame = np.empty((n_frames, bins), dtype=np.float32)

    # Crea l'oggetto per il calcolo della RDF
    rdf_computer = RDF(bins=bins, r_max=r_max)

    # --- 2. Loop sui frame ---
    # `freud` è così veloce che spesso un loop Python è sufficiente.
    for i in range(n_frames):
        # Estrai le posizioni per il frame corrente
        current_pos = positions[i]

        # Seleziona le posizioni degli atomi di interesse per questo frame
        points1 = current_pos[indices1]

        if type1 == type2:
            # Calcolo per coppie dello stesso tipo (es. O-O)
            # `system` contiene i punti di riferimento
            rdf_computer.compute(system=(box, points1), reset=True)
        else:
            # Calcolo per coppie di tipo diverso (es. O-H)
            # `system` ha i punti di riferimento (O), `query_points` i punti da cercare (H)
            points2 = current_pos[indices2]
            rdf_computer.compute(
                system=(box, points1), query_points=points2, reset=True
            )

        # Aggiungi la g(r) calcolata per questo frame alla lista
        rdf_per_frame[i] = rdf_computer.rdf

    # L'asse r è lo stesso per tutti i calcoli
    r_bins = rdf_computer.bin_centers

    print("Calcolo RDF con freud completato.")
    return r_bins, rdf_per_frame


# ==================================================================
# ------------------------- COSTANTI -------------------------------
# ==================================================================

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7
TRIG_EPSILON: float = 1e-12
DEFAULT_EPSILON: float = 1e-40
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5

# ==================================================================
# ---------------- FUNZIONI JIT OTTIMIZZATE ------------------------
# ==================================================================


@njit(fastmath=False, cache=True)
def _normalize_vector_numba(v_np: np.ndarray, epsilon_np: float) -> np.ndarray:
    """Helper function to normalize a vector, used for librations."""
    norm_val = np.linalg.norm(v_np)
    if norm_val < epsilon_np:
        return np.zeros_like(v_np)
    return v_np / norm_val


@njit(fastmath=False, cache=True)
def _process_single_frame_numba_jit(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    molecule_indices: np.ndarray,
    is_H_excited_mask: np.ndarray,
    oxygen_indices_for_hb: np.ndarray,
    kb_const: float,
    hb_cutoff_const: float,
    epsilon_const: float,
    trig_epsilon_const: float,
) -> Tuple:
    """
    Worker Numba JITtato per processare i dati di un singolo frame.
    Questa versione contiene la logica di calcolo COMPLETA E CORRETTA.
    """
    # Inizializza somme e contatori
    sum_T_stretch_exc, count_T_stretch_exc = 0.0, 0
    sum_T_stretch_norm, count_T_stretch_norm = 0.0, 0
    sum_T_bend_exc, count_T_bend_exc = 0.0, 0
    sum_T_bend_norm, count_T_bend_norm = 0.0, 0
    sum_T_bend_eq5_norm, count_T_bend_eq5_norm = 0.0, 0
    sum_T_bend_eq5_exc, count_T_bend_eq5_exc = 0.0, 0
    sum_T_bend_I_norm, count_T_bend_I_norm = 0.0, 0
    sum_T_bend_I_exc, count_T_bend_I_exc = 0.0, 0
    sum_T_hb, count_T_hb = 0.0, 0
    sum_T_twist_norm, count_T_twist_norm = 0.0, 0
    sum_T_twist_exc, count_T_twist_exc = 0.0, 0
    sum_T_wag_norm, count_T_wag_norm = 0.0, 0
    sum_T_wag_exc, count_T_wag_exc = 0.0, 0
    sum_T_rock_norm, count_T_rock_norm = 0.0, 0
    sum_T_rock_exc, count_T_rock_exc = 0.0, 0

    for mol_idx in range(molecule_indices.shape[0]):
        mol_atom_idxs = molecule_indices[mol_idx]
        O_gidx, H1_gidx, H2_gidx = mol_atom_idxs[0], mol_atom_idxs[1], mol_atom_idxs[2]

        m_O, m_H1, m_H2 = masses[O_gidx], masses[H1_gidx], masses[H2_gidx]
        M_total = m_O + m_H1 + m_H2

        # Stretch
        mu_OH1 = (m_O * m_H1) / (m_O + m_H1)
        d_OH1_vec = positions[O_gidx] - positions[H1_gidx]
        d_OH1_mag = np.linalg.norm(d_OH1_vec)
        if d_OH1_mag >= epsilon_const:
            u_OH1 = d_OH1_vec / d_OH1_mag
            v_rel1 = velocities[O_gidx] - velocities[H1_gidx]
            v_s1 = np.dot(v_rel1, u_OH1)
            T1 = (mu_OH1 * v_s1**2) / kb_const
            if is_H_excited_mask[H1_gidx]:
                sum_T_stretch_exc += T1
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += T1
                count_T_stretch_norm += 1

        mu_OH2 = (m_O * m_H2) / (m_O + m_H2)
        d_OH2_vec = positions[O_gidx] - positions[H2_gidx]
        d_OH2_mag = np.linalg.norm(d_OH2_vec)
        if d_OH2_mag >= epsilon_const:
            u_OH2 = d_OH2_vec / d_OH2_mag
            v_rel2 = velocities[O_gidx] - velocities[H2_gidx]
            v_s2 = np.dot(v_rel2, u_OH2)
            T2 = (mu_OH2 * v_s2**2) / kb_const
            if is_H_excited_mask[H2_gidx]:
                sum_T_stretch_exc += T2
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += T2
                count_T_stretch_norm += 1

        # Bending and Librations
        d_O_H1_v = positions[H1_gidx] - positions[O_gidx]
        d_O_H2_v = positions[H2_gidx] - positions[O_gidx]
        d_O_H1_m = np.linalg.norm(d_O_H1_v)
        d_O_H2_m = np.linalg.norm(d_O_H2_v)

        if d_O_H1_m >= epsilon_const and d_O_H2_m >= epsilon_const:
            u_O_H1 = d_O_H1_v / d_O_H1_m
            u_O_H2 = d_O_H2_v / d_O_H2_m

            v_O = velocities[O_gidx]
            v_H1 = velocities[H1_gidx]
            v_H2 = velocities[H2_gidx]

            # Bend (vels)
            v_H1_perp = v_H1 - np.dot(v_H1, u_O_H1) * u_O_H1
            v_H2_perp = v_H2 - np.dot(v_H2, u_O_H2) * u_O_H2
            d_H1H2_v = positions[H1_gidx] - positions[H2_gidx]
            d_H1H2_m = np.linalg.norm(d_H1H2_v)
            if d_H1H2_m >= epsilon_const:
                u_H1H2 = d_H1H2_v / d_H1H2_m
                delta_v_perp = v_H1_perp - v_H2_perp
                v_bend_scalar = np.dot(delta_v_perp, u_H1H2)
                mu_HH = (m_H1 * m_H2) / (m_H1 + m_H2)
                T_bend = (mu_HH * v_bend_scalar**2) / kb_const
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_exc += T_bend
                    count_T_bend_exc += 1
                else:
                    sum_T_bend_norm += T_bend
                    count_T_bend_norm += 1

            # Bend (angle)
            cos_theta = np.dot(u_O_H1, u_O_H2)
            cos_theta = min(1.0, max(-1.0, cos_theta))
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)

            if abs(sin_theta) > trig_epsilon_const and M_total > epsilon_const:
                d_dot_OH1 = np.dot(v_H1 - v_O, u_O_H1)
                d_dot_OH2 = np.dot(v_H2 - v_O, u_O_H2)
                u_dot_OH1 = ((v_H1 - v_O) - u_O_H1 * d_dot_OH1) / d_O_H1_m
                u_dot_OH2 = ((v_H2 - v_O) - u_O_H2 * d_dot_OH2) / d_O_H2_m
                theta_dot = (
                    -(np.dot(u_dot_OH1, u_O_H2) + np.dot(u_O_H1, u_dot_OH2)) / sin_theta
                )

                # Bend from Eq5
                theta_half = theta / 2.0
                c_h, s_h = np.cos(theta_half), np.sin(theta_half)
                term_mH_over_M = m_H1 / M_total
                term_mO_over_M = m_O / M_total
                v_O_y_dot = term_mH_over_M * (
                    (d_dot_OH1 * c_h - d_O_H1_m * s_h * (theta_dot / 2.0))
                    + (d_dot_OH2 * c_h - d_O_H2_m * s_h * (theta_dot / 2.0))
                )
                v_H1_x_dot = d_dot_OH1 * s_h + d_O_H1_m * c_h * (theta_dot / 2.0)
                v_H1_y_dot = -term_mO_over_M * (
                    d_dot_OH1 * c_h - d_O_H1_m * s_h * (theta_dot / 2.0)
                )
                v_H2_x_dot = d_dot_OH2 * s_h + d_O_H2_m * c_h * (theta_dot / 2.0)
                v_H2_y_dot = -term_mO_over_M * (
                    d_dot_OH2 * c_h - d_O_H2_m * s_h * (theta_dot / 2.0)
                )
                K_bend_eq5 = 0.5 * (
                    m_O * (v_O_y_dot**2)
                    + m_H1 * (v_H1_x_dot**2 + v_H1_y_dot**2)
                    + m_H2 * (v_H2_x_dot**2 + v_H2_y_dot**2)
                )
                T_bend_eq5 = K_bend_eq5 / (3.0 / 2.0 * kb_const)
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_eq5_exc += T_bend_eq5
                    count_T_bend_eq5_exc += 1
                else:
                    sum_T_bend_eq5_norm += T_bend_eq5
                    count_T_bend_eq5_norm += 1

                # Bend from Inertia
                d_OH_equil = 0.9572
                I_mom_HH = m_H1 * d_OH_equil**2 + m_H2 * d_OH_equil**2
                K_bend_I = 0.5 * I_mom_HH * (theta_dot**2)
                T_bend_I = 0.5 * K_bend_I / kb_const
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_I_exc += T_bend_I
                    count_T_bend_I_exc += 1
                else:
                    sum_T_bend_I_norm += T_bend_I
                    count_T_bend_I_norm += 1

                # Librations
                # ========= INIZIO BLOCCO CORRETTO =========
                masses_mol = np.empty(3, dtype=np.float64)
                masses_mol[0] = m_O
                masses_mol[1] = m_H1
                masses_mol[2] = m_H2

                positions_mol = np.empty((3, 3), dtype=np.float64)
                positions_mol[0, :] = positions[O_gidx]
                positions_mol[1, :] = positions[H1_gidx]
                positions_mol[2, :] = positions[H2_gidx]

                velocities_mol = np.empty((3, 3), dtype=np.float64)
                velocities_mol[0, :] = velocities[O_gidx]
                velocities_mol[1, :] = velocities[H1_gidx]
                velocities_mol[2, :] = velocities[H2_gidx]
                # ========= FINE BLOCCO CORRETTO =========

                R_cm = (
                    positions_mol[0] * m_O
                    + positions_mol[1] * m_H1
                    + positions_mol[2] * m_H2
                ) / M_total
                V_cm = (
                    velocities_mol[0] * m_O
                    + velocities_mol[1] * m_H1
                    + velocities_mol[2] * m_H2
                ) / M_total

                r_prime_lib_val = positions_mol - R_cm
                v_rel_cm_lib_val = velocities_mol - V_cm

                axis1_u = _normalize_vector_numba(u_O_H1 + u_O_H2, epsilon_const)
                axis3_w = _normalize_vector_numba(
                    np.cross(u_O_H1, u_O_H2), epsilon_const
                )
                axis2_v = _normalize_vector_numba(
                    np.cross(axis3_w, axis1_u), epsilon_const
                )

                axis3_w = _normalize_vector_numba(
                    np.cross(axis1_u, axis2_v), epsilon_const
                )

                I_11, I_22, I_33 = 0.0, 0.0, 0.0
                for i_atom in range(3):
                    r_p_i = r_prime_lib_val[i_atom]
                    m_i = masses_mol[i_atom]
                    r_p_dot_u, r_p_dot_v, r_p_dot_w = (
                        np.dot(r_p_i, axis1_u),
                        np.dot(r_p_i, axis2_v),
                        np.dot(r_p_i, axis3_w),
                    )
                    I_11 += m_i * (r_p_dot_v**2 + r_p_dot_w**2)
                    I_22 += m_i * (r_p_dot_u**2 + r_p_dot_w**2)
                    I_33 += m_i * (r_p_dot_u**2 + r_p_dot_v**2)

                L_lab = np.zeros(3, dtype=np.float64)
                for i_atom in range(3):
                    L_lab += masses_mol[i_atom] * np.cross(
                        r_prime_lib_val[i_atom], v_rel_cm_lib_val[i_atom]
                    )

                L1, L2, L3 = (
                    np.dot(L_lab, axis1_u),
                    np.dot(L_lab, axis2_v),
                    np.dot(L_lab, axis3_w),
                )
                mol_is_excited = (
                    is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]
                )

                if I_11 > epsilon_const:
                    T_twist = L1**2 / (I_11 * kb_const)
                    if mol_is_excited:
                        sum_T_twist_exc += T_twist
                        count_T_twist_exc += 1
                    else:
                        sum_T_twist_norm += T_twist
                        count_T_twist_norm += 1
                if I_22 > epsilon_const:
                    T_rock = L2**2 / (I_22 * kb_const)
                    if mol_is_excited:
                        sum_T_rock_exc += T_rock
                        count_T_rock_exc += 1
                    else:
                        sum_T_rock_norm += T_rock
                        count_T_rock_norm += 1
                if I_33 > epsilon_const:
                    T_wag = L3**2 / (I_33 * kb_const)
                    if mol_is_excited:
                        sum_T_wag_exc += T_wag
                        count_T_wag_exc += 1
                    else:
                        sum_T_wag_norm += T_wag
                        count_T_wag_norm += 1

    # Hydrogen Bond
    num_oxygens = len(oxygen_indices_for_hb)
    for i_o in range(num_oxygens):
        for j_o in range(i_o + 1, num_oxygens):
            O1_idx, O2_idx = oxygen_indices_for_hb[i_o], oxygen_indices_for_hb[j_o]
            d_OO_vec = positions[O1_idx] - positions[O2_idx]
            d_OO_mag = np.linalg.norm(d_OO_vec)
            if epsilon_const < d_OO_mag < hb_cutoff_const:
                u_OO = d_OO_vec / d_OO_mag
                v_rel_OO = velocities[O1_idx] - velocities[O2_idx]
                v_hb_scalar = np.dot(v_rel_OO, u_OO)
                mu_OO = masses[O1_idx] / 2.0
                sum_T_hb += (mu_OO * v_hb_scalar**2) / kb_const
                count_T_hb += 1

    return (
        sum_T_stretch_exc / count_T_stretch_exc if count_T_stretch_exc > 0 else np.nan,
        sum_T_stretch_norm / count_T_stretch_norm
        if count_T_stretch_norm > 0
        else np.nan,
        sum_T_bend_exc / count_T_bend_exc if count_T_bend_exc > 0 else np.nan,
        sum_T_bend_norm / count_T_bend_norm if count_T_bend_norm > 0 else np.nan,
        sum_T_bend_eq5_exc / count_T_bend_eq5_exc
        if count_T_bend_eq5_exc > 0
        else np.nan,
        sum_T_bend_eq5_norm / count_T_bend_eq5_norm
        if count_T_bend_eq5_norm > 0
        else np.nan,
        sum_T_bend_I_exc / count_T_bend_I_exc if count_T_bend_I_exc > 0 else np.nan,
        sum_T_bend_I_norm / count_T_bend_I_norm if count_T_bend_I_norm > 0 else np.nan,
        sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan,
        sum_T_twist_exc / count_T_twist_exc if count_T_twist_exc > 0 else np.nan,
        sum_T_twist_norm / count_T_twist_norm if count_T_twist_norm > 0 else np.nan,
        sum_T_wag_exc / count_T_wag_exc if count_T_wag_exc > 0 else np.nan,
        sum_T_wag_norm / count_T_wag_norm if count_T_wag_norm > 0 else np.nan,
        sum_T_rock_exc / count_T_rock_exc if count_T_rock_exc > 0 else np.nan,
        sum_T_rock_norm / count_T_rock_norm if count_T_rock_norm > 0 else np.nan,
    )


# ==================================================================
# ------------ FUNZIONE WORKER PER PARALLELISMO --------------------
# ==================================================================


def _parallel_worker_from_arrays(args_tuple: Tuple) -> Tuple[int, np.ndarray]:
    """
    Worker function for parallel processing that accepts NumPy arrays directly.
    """
    try:
        (
            frame_idx,
            pos_frame,
            vel_frame,
            masses,
            molecule_indices,
            is_H_excited_mask,
            oxygen_indices_for_hb,
            kb_const,
            hb_cutoff_const,
            epsilon_const,
            trig_epsilon_const,
        ) = args_tuple

        temps_tuple = _process_single_frame_numba_jit(
            pos_frame,
            vel_frame,
            masses,
            molecule_indices,
            is_H_excited_mask,
            oxygen_indices_for_hb,
            kb_const,
            hb_cutoff_const,
            epsilon_const,
            trig_epsilon_const,
        )
        return frame_idx, np.array(temps_tuple, dtype=np.float64)
    except Exception as e:
        print(f"Errore nel worker per il frame {args_tuple[0]}: {e}")
        return args_tuple[0], np.full(15, np.nan, dtype=np.float64)


# ==================================================================
# -------------------- FUNZIONE WRAPPER UTENTE ---------------------
# ==================================================================


def analyzeTEMPS(
    positions: np.ndarray,
    velocities: np.ndarray,
    atom_types: np.ndarray,
    excited_indices_filepath: str,
    kb_constant: float = DEFAULT_KB_CONSTANT,
    hb_cutoff: float = DEFAULT_HB_OO_DIST_CUTOFF,
    epsilon_val: float = DEFAULT_EPSILON,
    trig_epsilon: float = TRIG_EPSILON,
    num_workers: int = None,
) -> Dict[str, np.ndarray]:
    """
    Analizza le temperature molecolari partendo da array NumPy, usando un
    ProcessPoolExecutor per parallelizzare il calcolo sui frame.

    ASSUNZIONE: Gli atomi sono ordinati per molecola (O,H,H, O,H,H, ...)
    e i tipi sono 1 per Ossigeno e 2 per Idrogeno.
    """
    n_frames, n_atoms, _ = positions.shape

    # --- Preparazione Dati (eseguita una sola volta nel processo principale) ---
    if n_atoms % 3 != 0:
        raise ValueError("Il numero di atomi non è divisibile per 3.")
    n_mols = n_atoms // 3
    molecule_indices = np.arange(n_atoms, dtype=np.int32).reshape(n_mols, 3)

    mass_map = np.array([0.0, 15.999, 1.008], dtype=np.float64)
    try:
        masses = mass_map[atom_types]
    except IndexError:
        raise ValueError(
            "Trovato un tipo di atomo non valido. Ammessi solo 1 (O) e 2 (H)."
        )

    try:
        loaded_indices = np.loadtxt(excited_indices_filepath, dtype=np.int64)
        excited_h_indices = loaded_indices - 1
        if excited_h_indices.ndim == 0:
            excited_h_indices = np.array([excited_h_indices.item()])
    except Exception as e:
        print(f"Attenzione: impossibile caricare gli indici eccitati. Errore: {e}")
        excited_h_indices = np.array([], dtype=np.int64)

    is_H_excited_mask = np.zeros(n_atoms, dtype=np.bool_)
    if excited_h_indices.size > 0:
        is_H_excited_mask[excited_h_indices] = True

    oxygen_indices_for_hb = np.where(atom_types == 1)[0].astype(np.int32)

    # --- Creazione dei Task per i Worker ---
    tasks_args = []
    for i in range(n_frames):
        tasks_args.append(
            (
                i,
                positions[i],
                velocities[i],
                masses,
                molecule_indices,
                is_H_excited_mask,
                oxygen_indices_for_hb,
                kb_constant,
                hb_cutoff,
                epsilon_val,
                trig_epsilon,
            )
        )

    # --- Esecuzione Parallela ---
    results_array = np.full((n_frames, 15), np.nan, dtype=np.float64)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            from tqdm import tqdm

            future_results = list(
                tqdm(
                    executor.map(_parallel_worker_from_arrays, tasks_args),
                    total=n_frames,
                    desc="Processing Frames",
                )
            )
        except ImportError:
            future_results = list(
                executor.map(_parallel_worker_from_arrays, tasks_args)
            )

    for frame_idx, result_temps_array in future_results:
        results_array[frame_idx, :] = result_temps_array

    # --- Unpack dei risultati ---
    returned_data = {
        "stretch_excited_H": results_array[:, 0],
        "stretch_normal_H": results_array[:, 1],
        "bend_HOH_exc": results_array[:, 2],
        "bend_HOH_norm": results_array[:, 3],
        "bend_HOH_eq5_exc": results_array[:, 4],
        "bend_HOH_eq5_norm": results_array[:, 5],
        "bend_HOH_I_exc": results_array[:, 6],
        "bend_HOH_I_norm": results_array[:, 7],
        "hb": results_array[:, 8],
        "libr_twist_exc": results_array[:, 9],
        "libr_twist_norm": results_array[:, 10],
        "libr_wag_exc": results_array[:, 11],
        "libr_wag_norm": results_array[:, 12],
        "libr_rock_exc": results_array[:, 13],
        "libr_rock_norm": results_array[:, 14],
    }

    return returned_data
