from typing import Optional, Tuple

import numba  # Da importare all'inizio del file
import numpy as np

from MDtools.dataStructures import Frame

# Costanti utili (possono essere definite globalmente o passate/definite localmente)
TWO_PI = 2.0 * np.pi
# Fattore per convertire frequenza (in ps^-1 o THz) in wavenumbers (cm^-1)
# 1 / (c_luce_cm_per_s * 1e-12 s/ps) = 1 / (2.99792458e10 * 1e-12)
HZ_TO_CMINV_FACTOR = 33.3564095198152


# Questa funzione verrebbe chiamata da calculate_standard_vacf_optimized
# dopo aver preparato all_velocities_np e valid_atom_mask
@numba.jit(nopython=True, parallel=True, fastmath=True)
def _calculate_vacf_core_numba(
    Nframes: int,
    actual_max_lag_steps: int,
    total_atoms_in_system: int,
    all_velocities_np: np.ndarray,  # Array 3D (frame, atom_g_idx, xyz)
    valid_atom_mask: np.ndarray,  # Array 2D (frame, atom_g_idx)
) -> Tuple[np.ndarray, np.ndarray]:  # Ritorna vacf_sum, vacf_counts
    vacf_sum_all_nb = np.zeros(actual_max_lag_steps + 1, dtype=np.float64)
    vacf_counts_nb = np.zeros(actual_max_lag_steps + 1, dtype=np.int64)

    # Numba può parallelizzare questo loop esterno grazie a `parallel=True` e `numba.prange`
    for tau_s in numba.prange(actual_max_lag_steps + 1):
        # Le variabili di riduzione (somma/conteggio) devono essere locali al thread parallelo
        # o Numba deve essere in grado di gestirle correttamente (lo fa per somme semplici).
        # Per somme su array, Numba gestisce la riduzione se l'operazione è atomica o
        # se il loop è parallelizzabile in modo sicuro.
        # In questo caso, ogni tau_s è indipendente.

        current_sum_for_tau = 0.0  # Accumulatore locale per questo thread/tau_s
        current_count_for_tau = 0

        for t0_frame_idx in range(Nframes - tau_s):
            t0_plus_tau_s_frame_idx = t0_frame_idx + tau_s
            for atom_g_idx in range(total_atoms_in_system):
                if (
                    valid_atom_mask[t0_frame_idx, atom_g_idx]
                    and valid_atom_mask[t0_plus_tau_s_frame_idx, atom_g_idx]
                ):
                    v0_x = all_velocities_np[t0_frame_idx, atom_g_idx, 0]
                    v0_y = all_velocities_np[t0_frame_idx, atom_g_idx, 1]
                    v0_z = all_velocities_np[t0_frame_idx, atom_g_idx, 2]

                    v_tau_x = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 0]
                    v_tau_y = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 1]
                    v_tau_z = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 2]

                    dot_product = v0_x * v_tau_x + v0_y * v_tau_y + v0_z * v_tau_z
                    dot_product_norm = dot_product / (
                        v0_x * v0_x + v0_y * v0_y + v0_z * v0_z
                    )

                    current_sum_for_tau += dot_product_norm
                    current_count_for_tau += 1

        vacf_sum_all_nb[tau_s] = current_sum_for_tau
        vacf_counts_nb[tau_s] = current_count_for_tau

    return vacf_sum_all_nb, vacf_counts_nb


# --- Funzione Principale (Wrapper che chiama la logica Numba) ---
def calculateVACF(
    trajs: list[Frame],
    timestep: float = 1.0,
    corr_steps: int | None = None,
    Natoms_per_molecule: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la Funzione di Autocorrelazione delle Velocità (VACF) standard,
    ottimizzata con pre-estrazione delle velocità e Numba per i calcoli core.

    Args:
        trajs: Lista di oggetti Frame. Ogni Atom deve avere '.velocity' (np.ndarray).
        timestep: Intervallo di tempo tra i frame (in fs).
        corr_steps: Massimo numero di passi di lag per la correlazione, se non formito default al numero di frames -1
        Natoms_per_molecule_provided: (Opzionale ma raccomandato se costante)
                                      Numero di atomi per molecola. Usato per dimensionare
                                      l'array pre-estratto. Se None, si tenta di dedurlo
                                      assumendo struttura regolare.

    Returns:
        Un array NumPy contenente i valori della VACF mediata.
    """
    Nframes = len(trajs)
    if Nframes == 0:
        raise ValueError("Attenzione: La lista 'trajs' è vuota.")

    # Controlli preliminari sulla struttura dei dati
    if not trajs[0].molecules:
        raise ValueError("Attenzione: Il primo frame non contiene molecole.")

    Nmols = len(trajs[0].molecules)
    if Nmols == 0:
        raise ValueError("Attenzione: Nessuna molecola nel primo frame.")

    total_atoms_in_system = Nmols * Natoms_per_molecule
    if total_atoms_in_system == 0:
        raise ValueError("Attenzione: Numero totale di atomi calcolato come zero.")

    # --- Pre-estrazione delle velocità in array NumPy ---
    print(
        f"Pre-estrazione delle velocità per {Nframes} frames e {total_atoms_in_system} atomi target..."
    )
    all_velocities_np = np.zeros((Nframes, total_atoms_in_system, 3), dtype=np.float64)
    valid_atom_mask = np.zeros(
        (Nframes, total_atoms_in_system), dtype=np.bool_
    )  # np.bool_ per Numba

    global_atom_idx = 0
    for i_mol in range(Nmols):
        for j_atom_in_mol in range(Natoms_per_molecule):
            if global_atom_idx >= total_atoms_in_system:  # Sicurezza extra
                break

            atom_consistently_valid = True
            for frame_idx in range(Nframes):
                try:
                    # Questa indicizzazione assume una struttura regolare e costante
                    atom = trajs[frame_idx].molecules[i_mol][j_atom_in_mol]
                    if atom.velocity is not None and len(atom.velocity) == 3:
                        all_velocities_np[frame_idx, global_atom_idx, :] = atom.velocity
                        valid_atom_mask[frame_idx, global_atom_idx] = True
                    else:
                        valid_atom_mask[frame_idx, global_atom_idx] = False
                        if (
                            atom_consistently_valid
                        ):  # Segnala solo una volta per atomo logico
                            # print(f"Debug: Atomo logico (m{i_mol},a{j_atom_in_mol}) vel None/invalida al frame {frame_idx}")
                            pass
                        atom_consistently_valid = False
                except IndexError:
                    valid_atom_mask[frame_idx, global_atom_idx] = False
                    if atom_consistently_valid:
                        # print(f"Debug: Atomo logico (m{i_mol},a{j_atom_in_mol}) non trovato al frame {frame_idx}")
                        pass
                    atom_consistently_valid = False
            global_atom_idx += 1
        if global_atom_idx >= total_atoms_in_system:
            break
    print("Pre-estrazione completata.")

    # Determina i limiti effettivi per il lag
    actual_max_lag_steps = corr_steps or (Nframes - 1)
    if actual_max_lag_steps < 0:
        actual_max_lag_steps = 0

    print(f"Calcolo del core VACF con Numba (max_lag_steps={actual_max_lag_steps})...")
    # Chiamata alla funzione compilata con Numba
    vacf_sum_all, vacf_counts = _calculate_vacf_core_numba(
        Nframes,
        actual_max_lag_steps,
        total_atoms_in_system,
        all_velocities_np,
        valid_atom_mask,
    )
    print("Calcolo Numba completato.")

    # Calcola la VACF mediata finale
    vacf_final_averaged = np.full(actual_max_lag_steps + 1, np.nan, dtype=np.float64)
    valid_counts_mask_final = vacf_counts > 0  # Maschera per evitare divisione per zero
    vacf_final_averaged[valid_counts_mask_final] = (
        vacf_sum_all[valid_counts_mask_final] / vacf_counts[valid_counts_mask_final]
    )

    time_axis = np.arange(len(vacf_final_averaged)) * timestep

    return vacf_final_averaged, time_axis


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _calculate_mvacf_core_numba(
    Nframes: int,
    actual_max_lag_steps: int,
    total_atoms_in_system: int,
    all_velocities_np: np.ndarray,  # Array 3D (frame, atom_g_idx, xyz)
    valid_atom_mask: np.ndarray,  # Array 2D (frame, atom_g_idx)
    all_masses_np: np.ndarray,  # Array 1D (atom_g_idx) con le masse atomiche
) -> Tuple[np.ndarray, np.ndarray]:  # Ritorna mvacf_sum, mvacf_counts
    mvacf_sum_all_nb = np.zeros(actual_max_lag_steps + 1, dtype=np.float64)
    mvacf_counts_nb = np.zeros(
        actual_max_lag_steps + 1, dtype=np.int64
    )  # Conteggio delle coppie valide

    for tau_s in numba.prange(actual_max_lag_steps + 1):
        current_sum_for_tau = 0.0
        current_count_for_tau = 0  # Conteggio per la normalizzazione

        for t0_frame_idx in range(Nframes - tau_s):
            t0_plus_tau_s_frame_idx = t0_frame_idx + tau_s
            for atom_g_idx in range(total_atoms_in_system):
                if (
                    valid_atom_mask[t0_frame_idx, atom_g_idx]
                    and valid_atom_mask[t0_plus_tau_s_frame_idx, atom_g_idx]
                ):
                    v0_x = all_velocities_np[t0_frame_idx, atom_g_idx, 0]
                    v0_y = all_velocities_np[t0_frame_idx, atom_g_idx, 1]
                    v0_z = all_velocities_np[t0_frame_idx, atom_g_idx, 2]

                    v_tau_x = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 0]
                    v_tau_y = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 1]
                    v_tau_z = all_velocities_np[t0_plus_tau_s_frame_idx, atom_g_idx, 2]

                    dot_product = v0_x * v_tau_x + v0_y * v_tau_y + v0_z * v_tau_z
                    dot_product_norm = dot_product / (
                        v0_x * v0_x + v0_y * v0_y + v0_z * v0_z
                    )

                    mass_atom = all_masses_np[atom_g_idx]
                    current_sum_for_tau += (
                        mass_atom * dot_product_norm
                    )  # Prodotto scalare pesato per massa
                    current_count_for_tau += 1  # Normalizzazione per numero di termini

        mvacf_sum_all_nb[tau_s] = current_sum_for_tau
        mvacf_counts_nb[tau_s] = current_count_for_tau

    return mvacf_sum_all_nb, mvacf_counts_nb


def calculateMVACF(
    trajs: list[Frame],
    corr_steps: int,
    Natoms_per_molecule: int = 3,
) -> np.ndarray:
    """
    Calcola la Funzione di Autocorrelazione delle Velocità Pesata per Massa (MVACF),
    ottimizzata con pre-estrazione delle velocità e masse, e Numba per i calcoli core.

    Args:
        trajs: Lista di oggetti Frame. Ogni Atom deve avere '.velocity' (np.ndarray)
               e '.mass' (float). Si assume che la massa sia costante per ogni atomo
               attraverso i frame e viene estratta dal primo frame.
        corr_steps: Massimo numero di passi di lag per la correlazione.
        Natoms_per_molecule: Numero di atomi per molecola.

    Returns:
        Un array NumPy contenente i valori della MVACF mediata.
    """
    Nframes = len(trajs)
    if Nframes == 0:
        print("Attenzione (MVACF): La lista 'trajs' è vuota.")
        return np.array([])

    if not hasattr(trajs[0], "molecules") or not trajs[0].molecules:
        print("Attenzione (MVACF): Il primo frame non contiene molecole.")
        return np.array([])
    Nmols = len(trajs[0].molecules)
    if Nmols == 0:
        print("Attenzione (MVACF): Nessuna molecola nel primo frame.")
        return np.array([])

    total_atoms_in_system = Nmols * Natoms_per_molecule
    if total_atoms_in_system == 0:
        print("Attenzione (MVACF): Numero totale di atomi calcolato come zero.")
        return np.array([])

    # --- Pre-estrazione delle velocità e delle masse ---
    print(
        f"Pre-estrazione delle velocità e masse per MVACF ({Nframes} frames, {total_atoms_in_system} atomi target)..."
    )
    all_velocities_np = np.zeros((Nframes, total_atoms_in_system, 3), dtype=np.float64)
    valid_atom_mask = np.zeros((Nframes, total_atoms_in_system), dtype=np.bool_)
    all_masses_np = np.ones(
        total_atoms_in_system, dtype=np.float64
    )  # Default a 1.0 se la massa non è trovata

    global_atom_idx = 0
    for i_mol in range(Nmols):
        for j_atom_in_mol in range(Natoms_per_molecule):
            if global_atom_idx >= total_atoms_in_system:
                break

            # Estrarre la massa per questo global_atom_idx (dall'atomo nel primo frame)
            # Si assume che la massa dell'atomo (identificato da i_mol, j_atom_in_mol) sia costante.
            try:
                # Accedi all'atomo nel primo frame per ottenere la sua massa
                atom_for_mass = trajs[0].molecules[i_mol][j_atom_in_mol]
                if hasattr(atom_for_mass, "mass") and atom_for_mass.mass is not None:
                    mass_val = float(atom_for_mass.mass)
                    if mass_val > 0:
                        all_masses_np[global_atom_idx] = mass_val
                    else:
                        print(
                            f"Attenzione (MVACF): Massa non positiva ({mass_val}) per atomo (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}). Uso massa = 1.0."
                        )
                        all_masses_np[global_atom_idx] = 1.0
                else:
                    print(
                        f"Attenzione (MVACF): Attributo 'mass' mancante o None per atomo (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}) nel frame 0. Uso massa = 1.0."
                    )
                    all_masses_np[global_atom_idx] = 1.0
            except IndexError:
                print(
                    f"Attenzione (MVACF): Atomo (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}) non trovato nel frame 0 per estrazione massa. Uso massa = 1.0."
                )
                all_masses_np[global_atom_idx] = 1.0
            except Exception as e:  # Cattura altri errori imprevisti
                print(
                    f"Attenzione (MVACF): Errore imprevisto durante l'estrazione della massa per atomo (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}): {e}. Uso massa = 1.0."
                )
                all_masses_np[global_atom_idx] = 1.0

            # Estrarre le velocità per tutti i frame per l'atomo corrente
            atom_consistently_valid_for_velocity = True
            for frame_idx in range(Nframes):
                try:
                    atom = trajs[frame_idx].molecules[i_mol][j_atom_in_mol]
                    if (
                        hasattr(atom, "velocity")
                        and atom.velocity is not None
                        and len(atom.velocity) == 3
                    ):
                        all_velocities_np[frame_idx, global_atom_idx, :] = atom.velocity
                        valid_atom_mask[frame_idx, global_atom_idx] = True
                    else:
                        valid_atom_mask[frame_idx, global_atom_idx] = False
                        if atom_consistently_valid_for_velocity:
                            # print(f"Debug (MVACF): Atomo logico (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}) vel None/invalida al frame {frame_idx}")
                            pass
                        atom_consistently_valid_for_velocity = False
                except IndexError:
                    valid_atom_mask[frame_idx, global_atom_idx] = False
                    if atom_consistently_valid_for_velocity:
                        # print(f"Debug (MVACF): Atomo logico (m{i_mol},a{j_atom_in_mol}, g_idx {global_atom_idx}) non trovato al frame {frame_idx}")
                        pass
                    atom_consistently_valid_for_velocity = False
            global_atom_idx += 1
        if global_atom_idx >= total_atoms_in_system:
            break
    print("Pre-estrazione velocità e masse (MVACF) completata.")

    actual_max_lag_steps = min(corr_steps, Nframes - 1)
    if actual_max_lag_steps < 0:
        actual_max_lag_steps = 0

    print(f"Calcolo del core MVACF con Numba (max_lag_steps={actual_max_lag_steps})...")
    mvacf_sum_all, mvacf_counts = _calculate_mvacf_core_numba(
        Nframes,
        actual_max_lag_steps,
        total_atoms_in_system,
        all_velocities_np,
        valid_atom_mask,
        all_masses_np,  # Passa le masse alla funzione Numba
    )
    print("Calcolo Numba (MVACF) completato.")

    mvacf_final_averaged = np.full(actual_max_lag_steps + 1, np.nan, dtype=np.float64)
    valid_counts_mask_final = mvacf_counts > 0
    mvacf_final_averaged[valid_counts_mask_final] = (
        mvacf_sum_all[valid_counts_mask_final] / mvacf_counts[valid_counts_mask_final]
    )

    return mvacf_final_averaged


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
    gaussian_filter_width: Optional[float] = None,
    output_in_wavenumbers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Processa una funzione di correlazione temporale per ottenere il suo spettro di frequenza,
    replicando la logica del programma Fortran 'ft', incluso il metodo di Filon.

    Args:
        time_points: Array NumPy 1D dei punti temporali. Si assume dt costante.
        corr_values: Array NumPy 1D dei valori della funzione di correlazione.
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

    return frequencies_axis_final, spectrum_chat_array, filter_info_to_return
