import functools
import multiprocessing
import time
from typing import Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from numba.typed import List as NumbaTypedList

from MDtools.dataStructures import Atom, Frame

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5
DEFAULT_EPSILON: float = 1e-40


# ------------------------------------------------------------------
#                       NUMBA IMPLEMENTATION
# ------------------------------------------------------------------


# All'interno del tuo file, prima della funzione principale
def _prepare_frame_data_for_numba(
    current_frame: Frame,
    exc_indexes_set: Set[int],  # Il set originale va bene qui
    unwrapped_coords: bool,
    # Passa qui altri parametri costanti se servono solo per l'estrazione
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
    ]
]:
    """
    Estrae i dati da un oggetto Frame in array NumPy per l'uso con Numba.
    Restituisce una tupla di array:
    (positions, velocities, masses, atom_numeric_types, original_indices,
     molecule_atom_indices_list, is_H_excited_flat_mask)
    o None se il frame non può essere processato.

    'atom_numeric_types' usa una codifica (es., 0 per O, 1 per H, -1 per altro).
    'molecule_atom_indices_list' è una lista di array, ognuno [idx_O, idx_H1, idx_H2]
    che puntano agli indici negli array piatti (positions, velocities, ecc.).
    'is_H_excited_flat_mask' è una maschera booleana per tutti gli atomi estratti.
    """

    atoms_in_frame_flat: List[Atom] = []
    for mol_list in current_frame.molecules:
        atoms_in_frame_flat.extend(mol_list)

    if not atoms_in_frame_flat:
        return None

    num_all_atoms = len(atoms_in_frame_flat)
    positions_np = np.zeros((num_all_atoms, 3), dtype=np.float64)
    velocities_np = np.zeros(
        (num_all_atoms, 3), dtype=np.float64
    )  # Numba gestirà se alcune velocità sono zero
    masses_np = np.zeros(num_all_atoms, dtype=np.float64)
    atom_numeric_types_np = np.full(
        num_all_atoms, -1, dtype=np.int32
    )  # -1 per tipo sconosciuto/non rilevante
    original_indices_np = np.zeros(num_all_atoms, dtype=np.int32)
    is_H_excited_flat_mask_np = np.zeros(num_all_atoms, dtype=np.bool_)

    for i, atom_obj in enumerate(atoms_in_frame_flat):
        # Controlli di base (puoi renderli più robusti)
        if not all(
            hasattr(atom_obj, attr)
            for attr in ["index", "mass", "position", "velocity", "atom_string"]
        ):
            # print(f"Warning: Atomo con indice lista {i} manca di attributi, saltato nell'estrazione.")
            # Lascia i valori a zero/False per questo atomo, o gestisci diversamente
            continue

        pos_to_use = atom_obj.position
        if (
            unwrapped_coords
            and hasattr(atom_obj, "unwrapped_position")
            and atom_obj.unwrapped_position is not None
        ):
            pos_to_use = atom_obj.unwrapped_position

        positions_np[i, :] = pos_to_use
        if atom_obj.velocity is not None:  # Importante per Numba
            velocities_np[i, :] = atom_obj.velocity
        # else: velocities_np[i,:] rimane [0,0,0], la maschera di validità è implicita qui.
        #      La funzione Numba dovrà essere attenta se una velocità è zero perché non era definita.

        masses_np[i] = atom_obj.mass
        original_indices_np[i] = atom_obj.index

        if atom_obj.atom_string == "O":
            atom_numeric_types_np[i] = 0  # Es: 0 per Ossigeno
        elif atom_obj.atom_string == "H":
            atom_numeric_types_np[i] = 1  # Es: 1 per Idrogeno
            if atom_obj.index in exc_indexes_set:
                is_H_excited_flat_mask_np[i] = True

    # Costruzione di molecule_atom_indices_list:
    # Questa parte è cruciale e dipende da come identifichi O, H1, H2.
    # Il worker originale fa un loop e trova O, H1, H2 per ogni `molecule_atoms`.
    # Dobbiamo replicare questa logica per popolare `molecule_atom_indices_list`
    # con gli indici corretti degli array `positions_np`, `velocities_np`, etc.

    molecule_atom_indices_list_for_numba: List[np.ndarray] = []
    current_flat_idx = 0
    for mol_list_original in current_frame.molecules:
        if not isinstance(mol_list_original, list) or len(mol_list_original) != 3:
            # Avanza current_flat_idx per il numero di atomi in questa "molecola" non valida
            current_flat_idx += (
                len(mol_list_original) if isinstance(mol_list_original, list) else 0
            )
            continue

        o_idx_flat, h1_idx_flat, h2_idx_flat = -1, -1, -1
        temp_h_indices_flat = []

        # Mappa gli atomi di questa molecola ai loro indici nell'array piatto
        # Questa è una semplificazione; la corrispondenza deve essere robusta
        mol_atom_indices_in_flat_array = list(
            range(current_flat_idx, current_flat_idx + len(mol_list_original))
        )

        for k, local_atom_idx_in_flat_array in enumerate(
            mol_atom_indices_in_flat_array
        ):
            # atom_obj = atoms_in_frame_flat[local_atom_idx_in_flat_array] # Già processato sopra
            if atom_numeric_types_np[local_atom_idx_in_flat_array] == 0:  # Ossigeno
                o_idx_flat = local_atom_idx_in_flat_array
            elif atom_numeric_types_np[local_atom_idx_in_flat_array] == 1:  # Idrogeno
                temp_h_indices_flat.append(local_atom_idx_in_flat_array)

        if o_idx_flat != -1 and len(temp_h_indices_flat) == 2:
            h1_idx_flat, h2_idx_flat = temp_h_indices_flat[0], temp_h_indices_flat[1]
            molecule_atom_indices_list_for_numba.append(
                np.array([o_idx_flat, h1_idx_flat, h2_idx_flat], dtype=np.int32)
            )
        current_flat_idx += len(mol_list_original)

    return (
        positions_np,
        velocities_np,
        masses_np,
        atom_numeric_types_np,
        original_indices_np,
        molecule_atom_indices_list_for_numba,
        is_H_excited_flat_mask_np,
    )


@numba.jit(
    nopython=True, fastmath=True
)  # parallel=False, perché opera su un singolo frame
def _normalize_vector_numba(v_np: np.ndarray, epsilon_np: float) -> np.ndarray:
    norm_val = np.linalg.norm(v_np)  # Numba supporta np.linalg.norm
    if norm_val < epsilon_np:
        return np.zeros_like(v_np)  # Numba supporta np.zeros_like
    return v_np / norm_val


@numba.jit(nopython=True, fastmath=True)
def _process_single_frame_numba_jit(
    positions: np.ndarray,  # (N_all_atoms, 3)
    velocities: np.ndarray,  # (N_all_atoms, 3)
    masses: np.ndarray,  # (N_all_atoms,)
    # atom_types: np.ndarray,   # (N_all_atoms,) 0 per O, 1 per H (già usato per creare mol_indices)
    # original_indices: np.ndarray, # (N_all_atoms,)
    molecule_indices_list_of_arrays: numba.typed.List,  # Lista di array [O_idx, H1_idx, H2_idx]
    # Questo è un tipo Numba specifico.
    is_H_excited_mask: np.ndarray,  # (N_all_atoms,) maschera booleana
    oxygen_indices_for_hb_np: np.ndarray,  # Array di indici globali degli atomi O
    # Costanti
    kb_const_np: float,
    hb_cutoff_const_np: float,
    epsilon_const_np: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Worker Numba JITtato per processare i dati estratti di un singolo frame.
    """
    # Inizializza accumulatori (somme e conteggi) per ogni temperatura
    # Esempio per T_stretch_exc:
    sum_T_stretch_exc = 0.0
    count_T_stretch_exc = 0
    sum_T_stretch_norm = 0.0
    count_T_stretch_norm = 0
    sum_T_bend = 0.0
    count_T_bend = 0
    sum_T_hb = 0.0
    count_T_hb = 0
    sum_T_twist = 0.0
    count_T_twist = 0
    sum_T_wag = 0.0
    count_T_wag = 0
    sum_T_rock = 0.0
    count_T_rock = 0

    # Loop sulle molecole (definite da molecule_indices_list_of_arrays)
    for mol_idx in range(len(molecule_indices_list_of_arrays)):
        mol_atom_idxs = molecule_indices_list_of_arrays[
            mol_idx
        ]  # [O_gidx, H1_gidx, H2_gidx]

        O_gidx, H1_gidx, H2_gidx = mol_atom_idxs[0], mol_atom_idxs[1], mol_atom_idxs[2]

        # --- Calcolo Stretching (esempio di traduzione) ---
        m_O_val = masses[O_gidx]
        # Per H1
        m_H1_val = masses[H1_gidx]
        mu_OH1 = (m_O_val * m_H1_val) / (m_O_val + m_H1_val)

        d_OH1_vec_np = positions[O_gidx] - positions[H1_gidx]  # Array NumPy
        d_OH1_mag_np = np.linalg.norm(d_OH1_vec_np)

        if d_OH1_mag_np >= epsilon_const_np:  # Evita divisione per zero
            u_OH1_hat_np = d_OH1_vec_np / d_OH1_mag_np
            v_rel_OH1_np = velocities[O_gidx] - velocities[H1_gidx]
            v_stretch_scalar1 = np.dot(v_rel_OH1_np, u_OH1_hat_np)  # np.dot funziona
            temp_val1 = (mu_OH1 * v_stretch_scalar1**2) / kb_const_np

            if is_H_excited_mask[H1_gidx]:  # Usa la maschera precalcolata
                sum_T_stretch_exc += temp_val1
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += temp_val1
                count_T_stretch_norm += 1

        # Similmente per H2
        m_H2_val = masses[H2_gidx]
        mu_OH2 = (m_O_val * m_H2_val) / (m_O_val + m_H2_val)
        d_OH2_vec_np = positions[O_gidx] - positions[H2_gidx]
        d_OH2_mag_np = np.linalg.norm(d_OH2_vec_np)
        if d_OH2_mag_np >= epsilon_const_np:
            u_OH2_hat_np = d_OH2_vec_np / d_OH2_mag_np
            v_rel_OH2_np = velocities[O_gidx] - velocities[H2_gidx]
            v_stretch_scalar2 = np.dot(v_rel_OH2_np, u_OH2_hat_np)
            temp_val2 = (mu_OH2 * v_stretch_scalar2**2) / kb_const_np
            if is_H_excited_mask[H2_gidx]:
                sum_T_stretch_exc += temp_val2
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += temp_val2
                count_T_stretch_norm += 1

        # --- Calcolo Bending (simile, traducendo la logica) ---
        d_O_H1_v_np = positions[H1_gidx] - positions[O_gidx]
        d_O_H2_v_np = positions[H2_gidx] - positions[O_gidx]
        d_O_H1_m_np = np.linalg.norm(d_O_H1_v_np)
        d_O_H2_m_np = np.linalg.norm(d_O_H2_v_np)

        if d_O_H1_m_np >= epsilon_const_np and d_O_H2_m_np >= epsilon_const_np:
            u_O_H1_h_np = d_O_H1_v_np / d_O_H1_m_np
            u_O_H2_h_np = d_O_H2_v_np / d_O_H2_m_np

            v_H1_np = velocities[H1_gidx]
            v_H2_np = velocities[H2_gidx]

            # v_H1_perp = v_H1_np - np.dot(v_H1_np, u_O_H1_h_np) * u_O_H1_h_np <- errore di shape, np.dot è scalare
            # Bisogna fare (scalare) * vettore
            v_H1_perp_np = v_H1_np - (np.dot(v_H1_np, u_O_H1_h_np) * u_O_H1_h_np)
            v_H2_perp_np = v_H2_np - (np.dot(v_H2_np, u_O_H2_h_np) * u_O_H2_h_np)

            d_H1H2_v_np = positions[H1_gidx] - positions[H2_gidx]
            d_H1H2_m_np = np.linalg.norm(d_H1H2_v_np)

            if d_H1H2_m_np >= epsilon_const_np:
                u_H1H2_h_np = d_H1H2_v_np / d_H1H2_m_np
                v_bend_scalar_np = np.dot(v_H1_perp_np - v_H2_perp_np, u_H1H2_h_np)
                # m_H è la massa di un idrogeno (es. H1)
                mu_HH_np = masses[H1_gidx] / 2.0
                T_bend_val = (mu_HH_np * v_bend_scalar_np**2) / kb_const_np
                sum_T_bend += T_bend_val
                count_T_bend += 1

        # --- Calcolo Librazioni (molto più complesso da tradurre direttamente) ---
        # Questa parte richiede un'attenta traduzione di:
        # - Calcolo del centro di massa (CM) e velocità del CM per la molecola [O_gidx, H1_gidx, H2_gidx]
        # - Posizioni e velocità relative al CM
        # - Definizione degli assi corpo-fissi (u,v,w) usando i vettori OH1 e OH2
        #   (richiede _normalize_vector_numba e np.cross)
        # - Calcolo dei momenti d'inerzia (I_11, I_22, I_33) rispetto a questi assi
        # - Calcolo del momento angolare (L_lab) e delle sue componenti (L1, L2, L3) sugli assi corpo-fissi
        # - Calcolo delle temperature librazionali.
        # È fattibile ma richiede attenzione ai dettagli e all'uso di operazioni Numba-compatibili.
        # Per brevità, ometto l'implementazione dettagliata qui, ma la logica sarebbe:

        # --- Esempio scheletrico per Librazioni ---
        mol_indices_for_lib = mol_atom_idxs  # già [O_gidx, H1_gidx, H2_gidx]

        # Estrarre masse, posizioni, velocità per questa molecola
        masses_mol = np.array(
            [
                masses[mol_indices_for_lib[0]],
                masses[mol_indices_for_lib[1]],
                masses[mol_indices_for_lib[2]],
            ]
        )
        positions_mol = np.array(
            [
                positions[mol_indices_for_lib[0]],
                positions[mol_indices_for_lib[1]],
                positions[mol_indices_for_lib[2]],
            ]
        )
        velocities_mol = np.array(
            [
                velocities[mol_indices_for_lib[0]],
                velocities[mol_indices_for_lib[1]],
                velocities[mol_indices_for_lib[2]],
            ]
        )

        M_total_lib_val = np.sum(masses_mol)
        if M_total_lib_val >= epsilon_const_np:
            R_cm_lib_val = np.zeros(3)
            V_cm_lib_val = np.zeros(3)
            for k_dim in range(3):  # Per x,y,z
                R_cm_lib_val[k_dim] = (
                    np.sum(positions_mol[:, k_dim] * masses_mol) / M_total_lib_val
                )
                V_cm_lib_val[k_dim] = (
                    np.sum(velocities_mol[:, k_dim] * masses_mol) / M_total_lib_val
                )

            r_prime_lib_val = positions_mol - R_cm_lib_val  # Shape (3,3)
            v_rel_cm_lib_val = velocities_mol - V_cm_lib_val  # Shape (3,3)

            # Assi corpo-fissi (u, v, w)
            vec_OH1_lib_np = positions[H1_gidx] - positions[O_gidx]  # dal frame globale
            vec_OH2_lib_np = positions[H2_gidx] - positions[O_gidx]  # dal frame globale

            # _normalize_vector_numba è definita sopra
            norm_OH1 = _normalize_vector_numba(vec_OH1_lib_np, epsilon_const_np)
            norm_OH2 = _normalize_vector_numba(vec_OH2_lib_np, epsilon_const_np)

            axis1_u_np = _normalize_vector_numba(norm_OH1 + norm_OH2, epsilon_const_np)

            # np.cross è supportato da Numba
            cross_OH1_OH2 = np.cross(
                vec_OH1_lib_np, vec_OH2_lib_np
            )  # non usare i normalizzati qui per np.cross
            axis3_w_np = _normalize_vector_numba(cross_OH1_OH2, epsilon_const_np)

            cross_w_u = np.cross(axis3_w_np, axis1_u_np)
            axis2_v_np = _normalize_vector_numba(cross_w_u, epsilon_const_np)

            # Re-ortogonalizza w per assicurare la destrorsità, se necessario (come nel codice originale)
            cross_u_v_reorth = np.cross(axis1_u_np, axis2_v_np)
            axis3_w_np = _normalize_vector_numba(cross_u_v_reorth, epsilon_const_np)

            # Momenti d'inerzia I_uu, I_vv, I_ww (I_11, I_22, I_33)
            I_11, I_22, I_33 = 0.0, 0.0, 0.0
            for i_atom_in_mol in range(3):  # O, H1, H2
                r_p_i = r_prime_lib_val[i_atom_in_mol]  # Vettore pos relativo al CM
                m_i = masses_mol[i_atom_in_mol]

                # I_uu = sum m_i * ( (r_p_i . v_axis)^2 + (r_p_i . w_axis)^2 )
                r_p_dot_v = np.dot(r_p_i, axis2_v_np)
                r_p_dot_w = np.dot(r_p_i, axis3_w_np)
                I_11 += m_i * (r_p_dot_v**2 + r_p_dot_w**2)

                r_p_dot_u = np.dot(r_p_i, axis1_u_np)
                # r_p_dot_w (già calcolato)
                I_22 += m_i * (r_p_dot_u**2 + r_p_dot_w**2)

                # r_p_dot_u, r_p_dot_v (già calcolati)
                I_33 += m_i * (r_p_dot_u**2 + r_p_dot_v**2)

            # Momento angolare L_lab = sum m_i * (r_prime_i x v_rel_cm_i)
            L_lab_val = np.zeros(3)
            for i_atom_in_mol in range(3):
                L_lab_val += masses_mol[i_atom_in_mol] * np.cross(
                    r_prime_lib_val[i_atom_in_mol], v_rel_cm_lib_val[i_atom_in_mol]
                )

            L1_val = np.dot(L_lab_val, axis1_u_np)  # Twist
            L2_val = np.dot(L_lab_val, axis2_v_np)  # Rock
            L3_val = np.dot(L_lab_val, axis3_w_np)  # Wag

            if I_11 > epsilon_const_np:
                sum_T_twist += L1_val**2 / (
                    I_11 * kb_const_np
                )  # Aggiunto + epsilon_const_np al denominatore per sicurezza
                count_T_twist += 1
            if I_22 > epsilon_const_np:
                sum_T_rock += L2_val**2 / (I_22 * kb_const_np)
                count_T_rock += 1
            if I_33 > epsilon_const_np:
                sum_T_wag += L3_val**2 / (I_33 * kb_const_np)
                count_T_wag += 1

    # --- Calcolo Legami H (tra molecole) ---
    # Iterare su coppie di Ossigeni (usando oxygen_indices_for_hb_np)
    num_oxygens = len(oxygen_indices_for_hb_np)
    if num_oxygens >= 2:
        for i_idx_o_list in range(num_oxygens):
            O1_gidx_hb = oxygen_indices_for_hb_np[i_idx_o_list]

            for j_idx_o_list in range(i_idx_o_list + 1, num_oxygens):
                O2_gidx_hb = oxygen_indices_for_hb_np[j_idx_o_list]

                pos_O1 = positions[O1_gidx_hb]
                pos_O2 = positions[O2_gidx_hb]
                d_O1O2_vec_hb = pos_O1 - pos_O2
                d_O1O2_mag_hb = np.linalg.norm(d_O1O2_vec_hb)

                if epsilon_const_np < d_O1O2_mag_hb < hb_cutoff_const_np:
                    u_O1O2_hat_hb = d_O1O2_vec_hb / d_O1O2_mag_hb
                    vel_O1 = velocities[O1_gidx_hb]
                    vel_O2 = velocities[O2_gidx_hb]
                    v_rel_O1O2_hb = vel_O1 - vel_O2
                    v_HB_scalar_hb = np.dot(v_rel_O1O2_hb, u_O1O2_hat_hb)

                    # Assumiamo che la massa di O1 e O2 sia la stessa
                    m_O_for_hb = masses[O1_gidx_hb]
                    mu_OO_hb = m_O_for_hb / 2.0

                    sum_T_hb += (mu_OO_hb * v_HB_scalar_hb**2) / kb_const_np
                    count_T_hb += 1

    # Calcola medie (evita divisione per zero)
    avg_T_stretch_exc = (
        sum_T_stretch_exc / count_T_stretch_exc if count_T_stretch_exc > 0 else np.nan
    )
    avg_T_stretch_norm = (
        sum_T_stretch_norm / count_T_stretch_norm
        if count_T_stretch_norm > 0
        else np.nan
    )
    avg_T_bend = sum_T_bend / count_T_bend if count_T_bend > 0 else np.nan
    avg_T_hb = sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan
    avg_T_twist = sum_T_twist / count_T_twist if count_T_twist > 0 else np.nan
    avg_T_wag = sum_T_wag / count_T_wag if count_T_wag > 0 else np.nan
    avg_T_rock = sum_T_rock / count_T_rock if count_T_rock > 0 else np.nan

    return (
        avg_T_stretch_exc,
        avg_T_stretch_norm,
        avg_T_bend,
        avg_T_hb,
        avg_T_twist,
        avg_T_wag,
        avg_T_rock,
    )


def analyze_temps_numba(
    trajs_data: List[Frame],
    exc_indexes_file_path: str,  # Usato per popolare exc_indexes_set
    unwrapped_coords: bool = False,
    # num_processes: Optional[int] = None, # Non più usato direttamente per Pool
    kb_constant: float = DEFAULT_KB_CONSTANT,
    hb_cutoff: float = DEFAULT_HB_OO_DIST_CUTOFF,
    epsilon_val: float = DEFAULT_EPSILON,
    # Aggiungi un flag per abilitare la parallelizzazione Numba sui frame
    parallel_numba_frames: bool = True,
) -> Dict[str, List[float]]:
    """
    Versione Numba-accelerated di analyze_molecular_temperatures.
    """
    if not trajs_data:
        raise ValueError("Nessun frame fornito.")
    print(f"Analisi Numba: {len(trajs_data)} frames.")
    overall_start_time = time.time()

    # Carica exc_indexes_set (come prima)
    exc_indexes_set: Set[int] = set()
    try:
        # ... (logica di caricamento file come nel tuo script originale) ...
        # Per esempio:
        loaded_indices = np.loadtxt(exc_indexes_file_path, dtype=int)
        if loaded_indices.ndim == 0:  # Caso di un singolo numero nel file
            exc_indexes_set = {int(loaded_indices.item())}
        else:
            exc_indexes_set = set(loaded_indices.flatten().astype(int))
        print(f"Loaded {len(exc_indexes_set)} excited atom indices.")
    except Exception as e:
        print(
            f"Warning: Could not load excited indices from '{exc_indexes_file_path}': {e}"
        )

    # 1. Pre-processa tutti i frame per Numba
    # Questa lista conterrà tuple di array NumPy per ogni frame, o None
    print("Pre-processing frames for Numba...")
    preprocessed_data_for_frames = []
    for frame_obj in trajs_data:
        data_tuple = _prepare_frame_data_for_numba(
            frame_obj, exc_indexes_set, unwrapped_coords
        )
        preprocessed_data_for_frames.append(data_tuple)
    print("Pre-processing completato.")

    num_frames = len(trajs_data)
    # Prealloca array per i risultati
    # (7 temperature, num_frames)
    results_array = np.full((num_frames, 7), np.nan, dtype=np.float64)

    # Converti exc_indexes_set in array NumPy per Numba JIT function
    exc_indices_np_for_jit = np.array(list(exc_indexes_set), dtype=np.int32)

    # Scegli se eseguire il loop sui frame in parallelo con Numba o sequenzialmente
    # La parallelizzazione a livello di frame con Numba è più complessa se la funzione
    # JIT per singolo frame (_process_single_frame_numba_jit) non è thread-safe
    # o se la gestione dei dati pre-processati diventa complicata.
    # Un approccio più semplice è rendere _process_single_frame_numba_jit veloce,
    # e il loop sui frame rimane in Python (ma chiama codice compilato).

    # Per parallelizzare il loop sui frame, la funzione JIT principale dovrebbe
    # prendere tutti i dati pre-processati (come liste di array) e iterare con prange.
    # Data la complessità di passare List[List[np.ndarray]] a Numba in modo efficiente,
    # per ora chiameremo la funzione JIT per singolo frame sequenzialmente.
    # La parallelizzazione `parallel=True` in `_process_single_frame_numba_jit`
    # non avrà effetto se chiamata per un solo frame alla volta.
    # È più per parallelizzare loop *interni* a quella funzione, se presenti e adatti.

    print("Processing frames con Numba JIT function (sequenzialmente)...")
    jit_processing_start_time = time.time()
    for i in range(num_frames):
        frame_data_np = preprocessed_data_for_frames[i]
        if frame_data_np is None:  # Frame non processabile
            # results_array[i,:] rimane NaN
            continue

        (
            positions_np,
            velocities_np,
            masses_np,
            atom_numeric_types_np,
            original_indices_np,
            molecule_atom_indices_list_for_numba,
            is_H_excited_flat_mask_np,
        ) = frame_data_np

        # Per passare List[np.ndarray] a Numba, usa numba.typed.List
        molecule_indices_typed_list = NumbaTypedList()
        if molecule_atom_indices_list_for_numba:  # Verifica se la lista non è vuota
            for arr in molecule_atom_indices_list_for_numba:
                molecule_indices_typed_list.append(arr)

        # Estrai oxygen_indices_for_hb dagli dati pre-processati
        # (Assumendo che _prepare_frame_data_for_numba lo restituisca)
        # Per questo scheletro, devo aggiungerlo all'output di _prepare_frame_data_for_numba
        # e passarlo qui. Esempio:
        # _, _, _, _, _, _, _, oxygen_indices_hb_np = frame_data_np
        # Ho aggiornato _prepare_frame_data_for_numba per includerlo

        # Estrarre oxygen_indices_for_hb. Nell'attuale _prepare_frame_data_for_numba,
        # non è esplicitamente ritornato nella tupla. Dovrebbe essere.
        # Per ora, lo ricalcolo qui basandomi su atom_numeric_types_np.
        # (Questo è inefficiente, dovrebbe essere parte di frame_data_np)

        temp_oxygen_indices = []
        for atom_idx_loop in range(len(atom_numeric_types_np)):
            if atom_numeric_types_np[atom_idx_loop] == 0:  # 0 per Ossigeno
                temp_oxygen_indices.append(atom_idx_loop)
        oxygen_indices_for_hb_np_arg = np.array(temp_oxygen_indices, dtype=np.int32)

        # Chiamata alla funzione JITtata
        # Nota: la gestione di exc_indices_np e original_indices_np deve essere
        # coerente con come la funzione JIT si aspetta di fare il matching.
        # `is_H_excited_mask` è già una maschera booleana sugli indici piatti.

        try:
            temps_tuple = _process_single_frame_numba_jit(
                positions_np,
                velocities_np,
                masses_np,
                # atom_numeric_types_np, # Non passato direttamente se mol_indices lo gestisce
                molecule_indices_typed_list,
                is_H_excited_flat_mask_np,
                oxygen_indices_for_hb_np_arg,  # Passa gli indici degli ossigeni
                kb_constant,
                hb_cutoff,
                epsilon_val,
            )
            results_array[i, :] = np.array(temps_tuple)
        except Exception as e_jit:
            print(f"Errore durante l'esecuzione JIT per frame {i}: {e_jit}")
            # results_array[i,:] rimane NaN

    jit_processing_end_time = time.time()
    print(
        f"Elaborazione JIT dei frame finita in {jit_processing_end_time - jit_processing_start_time:.2f} secondi."
    )

    # Unpack results (come nel tuo codice originale)
    frame_avg_T_stretch_exc_list = list(results_array[:, 0])
    frame_avg_T_stretch_norm_list = list(results_array[:, 1])
    frame_avg_T_bend_list = list(results_array[:, 2])
    frame_avg_T_hb_list = list(results_array[:, 3])
    frame_avg_T_twist_list = list(results_array[:, 4])
    frame_avg_T_wag_list = list(results_array[:, 5])
    frame_avg_T_rock_list = list(results_array[:, 6])

    # Calcola medie complessive (come nel tuo codice originale)
    # ... (uso di robust_nanmean) ...
    # Esempio:
    def robust_nanmean(lst: List[float]) -> float:
        valid_values = [x for x in lst if not np.isnan(x)]
        return float(np.mean(valid_values)) if valid_values else np.nan

    print_overall_avg_T_stretch_exc = robust_nanmean(frame_avg_T_stretch_exc_list)
    # ... calcola le altre medie ...
    print(
        f"Overall Avg T_stretch_exc (Numba): {print_overall_avg_T_stretch_exc:.2f} K (esempio)"
    )

    overall_end_time = time.time()
    print(
        f"Tempo totale esecuzione funzione Numba: {overall_end_time - overall_start_time:.2f} secondi."
    )

    returned_data = {
        "stretch_excited_H": frame_avg_T_stretch_exc_list,
        "stretch_normal_H": frame_avg_T_stretch_norm_list,
        "bend_HOH": frame_avg_T_bend_list,
        "hb": frame_avg_T_hb_list,
        "libr_twist": frame_avg_T_twist_list,
        "libr_wag": frame_avg_T_wag_list,
        "libr_rock": frame_avg_T_rock_list,
    }
    return returned_data


# ------------------------------------------------------------------
#                 MULTIPROCESSING IMPLEMENTATION
# ------------------------------------------------------------------


# --- Inner Helper Function (Not intended for direct external use) ---
def _normalize_vector(v: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalizes a vector, handling near-zero norms."""
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return np.zeros_like(v)
    return v / norm


# --- Inner Worker Function (for parallel processing, not for direct external use) ---
def _process_single_frame_worker(
    frame_data_tuple: Tuple[int, Frame],
    exc_indexes_set_arg: Set[int],
    unwrapped_coords: bool,
    kb_const: float,
    hb_cutoff_const: float,
    epsilon_const: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Processes a single frame to calculate various molecular temperatures.
    Returns a tuple of 7 temperatures:
    (T_stretch_exc, T_stretch_norm, T_bend, T_hb, T_twist, T_wag, T_rock)
    Each can be np.nan if not calculable for the frame.
    """
    frame_idx, current_frame = frame_data_tuple

    # Helper function to get the right position based on flag
    def get_position(atom: Atom) -> np.ndarray:
        if (
            unwrapped_coords
            and hasattr(atom, "unwrapped_position")
            and atom.unwrapped_position is not None
        ):
            return atom.unwrapped_position
        return atom.position

    molecules_in_frame: List[List[Atom]] = current_frame.molecules
    T_stretch_exc_frame: List[float] = []
    T_stretch_norm_frame: List[float] = []
    T_bend_frame: List[float] = []
    T_hb_frame: List[float] = []  # For H-bonds between molecules
    T_twist_mol: List[float] = []  # Per-molecule librational temps
    T_wag_mol: List[float] = []
    T_rock_mol: List[float] = []

    for mol_idx, molecule_atoms in enumerate(molecules_in_frame):
        if not isinstance(molecule_atoms, list) or len(molecule_atoms) != 3:
            continue

        O_atom: Optional[Atom] = None
        H1_atom: Optional[Atom] = None
        H2_atom: Optional[Atom] = None
        temp_h_atoms: List[Atom] = []

        try:
            for atom in molecule_atoms:
                # Basic attribute check
                if not all(
                    hasattr(atom, attr)
                    for attr in ["atom_string", "index", "mass", "position", "velocity"]
                ):
                    continue
                if atom.atom_string == "O":
                    O_atom = atom
                elif atom.atom_string == "H":
                    temp_h_atoms.append(atom)

            if O_atom and len(temp_h_atoms) == 2:
                # A simple way to get H1 and H2; could be refined if order matters significantly
                H1_atom, H2_atom = temp_h_atoms[0], temp_h_atoms[1]
            else:
                continue
        except AttributeError:  # Should be caught by hasattr checks mostly
            continue

        if not (
            O_atom and H1_atom and H2_atom
        ):  # Ensure all three were found and assigned
            continue

        # --- Vibrational Temperatures (Stretching, Bending) for the current molecule ---
        try:
            m_O, m_H = O_atom.mass, H1_atom.mass
            mu_OH = (m_O * m_H) / (m_O + m_H)

            for h_atom_current in [H1_atom, H2_atom]:  # OH Stretching
                d_OH_vec = get_position(O_atom) - get_position(h_atom_current)
                d_OH_mag = np.linalg.norm(d_OH_vec)
                if d_OH_mag < epsilon_const:
                    continue
                u_OH_hat = d_OH_vec / d_OH_mag
                v_rel_OH = O_atom.velocity - h_atom_current.velocity
                v_stretch_scalar = np.dot(v_rel_OH, u_OH_hat)
                temp_val = (mu_OH * v_stretch_scalar**2) / kb_const
                if h_atom_current.index in exc_indexes_set_arg:
                    T_stretch_exc_frame.append(temp_val)
                else:
                    T_stretch_norm_frame.append(temp_val)

            # HOH Bending
            d_O_H1_vec = get_position(H1_atom) - get_position(O_atom)
            d_O_H2_vec = get_position(H2_atom) - get_position(O_atom)
            d_O_H1_mag = np.linalg.norm(d_O_H1_vec)
            d_O_H2_mag = np.linalg.norm(d_O_H2_vec)

            if d_O_H1_mag < epsilon_const or d_O_H2_mag < epsilon_const:
                continue
            u_O_H1_hat = d_O_H1_vec / d_O_H1_mag
            u_O_H2_hat = d_O_H2_vec / d_O_H2_mag

            v_H1_perp = (
                H1_atom.velocity - np.dot(H1_atom.velocity, u_O_H1_hat) * u_O_H1_hat
            )
            v_H2_perp = (
                H2_atom.velocity - np.dot(H2_atom.velocity, u_O_H2_hat) * u_O_H2_hat
            )

            d_H1H2_vec = get_position(H1_atom) - get_position(H2_atom)
            d_H1H2_mag = np.linalg.norm(d_H1H2_vec)
            if d_H1H2_mag < epsilon_const:
                continue
            u_H1H2_hat = d_H1H2_vec / d_H1H2_mag

            v_bend_scalar = np.dot(v_H1_perp - v_H2_perp, u_H1H2_hat)
            mu_HH = m_H / 2.0  # Effective mass for H-H bending mode
            T_bend_frame.append((mu_HH * v_bend_scalar**2) / kb_const)
        except Exception:
            pass  # Skip this molecule for vib if error occurs

        # --- Librational Temperatures for the current molecule ---
        try:
            atoms_in_mol_lib: List[Atom] = [O_atom, H1_atom, H2_atom]
            masses_lib = np.array([atom.mass for atom in atoms_in_mol_lib])
            positions_lib = np.array([get_position(atom) for atom in atoms_in_mol_lib])
            velocities_lib = np.array([atom.velocity for atom in atoms_in_mol_lib])

            M_total_lib = np.sum(masses_lib)
            if M_total_lib < epsilon_const:
                continue

            R_cm_lib = (
                np.sum(positions_lib * masses_lib[:, np.newaxis], axis=0) / M_total_lib
            )
            V_cm_lib = (
                np.sum(velocities_lib * masses_lib[:, np.newaxis], axis=0) / M_total_lib
            )

            r_prime_lib = positions_lib - R_cm_lib  # positions relative to CM
            v_rel_cm_lib = velocities_lib - V_cm_lib  # velocities relative to CM

            vec_OH1_lib = get_position(H1_atom) - get_position(O_atom)
            vec_OH2_lib = get_position(H2_atom) - get_position(O_atom)

            # Define body-fixed axes (u: bisector, w: normal to plane, v: in-plane perp to u)
            axis1_u = _normalize_vector(
                _normalize_vector(vec_OH1_lib, epsilon_const)
                + _normalize_vector(vec_OH2_lib, epsilon_const),
                epsilon_const,
            )
            axis3_w = _normalize_vector(
                np.cross(vec_OH1_lib, vec_OH2_lib), epsilon_const
            )
            axis2_v = _normalize_vector(np.cross(axis3_w, axis1_u), epsilon_const)
            axis3_w = _normalize_vector(
                np.cross(axis1_u, axis2_v), epsilon_const
            )  # Re-orthogonalize

            # Moments of Inertia
            I_11, I_22, I_33 = 0.0, 0.0, 0.0
            for i in range(3):  # O, H1, H2
                r_p_on_axis2 = np.dot(r_prime_lib[i], axis2_v)
                r_p_on_axis3 = np.dot(r_prime_lib[i], axis3_w)
                I_11 += masses_lib[i] * (r_p_on_axis2**2 + r_p_on_axis3**2)

                r_p_on_axis1 = np.dot(r_prime_lib[i], axis1_u)
                I_22 += masses_lib[i] * (r_p_on_axis1**2 + r_p_on_axis3**2)

                I_33 += masses_lib[i] * (r_p_on_axis1**2 + r_p_on_axis2**2)

            # Angular momentum
            L_lab_lib = np.zeros(3)
            for i in range(3):
                L_lab_lib += masses_lib[i] * np.cross(r_prime_lib[i], v_rel_cm_lib[i])

            L1 = np.dot(L_lab_lib, axis1_u)  # Component along u (twist)
            L2 = np.dot(L_lab_lib, axis2_v)  # Component along v (rock)
            L3 = np.dot(L_lab_lib, axis3_w)  # Component along w (wag)

            if I_11 > epsilon_const:
                T_twist_mol.append(L1**2 / (I_11 * kb_const + epsilon_const))
            if I_22 > epsilon_const:
                T_rock_mol.append(L2**2 / (I_22 * kb_const + epsilon_const))
            if I_33 > epsilon_const:
                T_wag_mol.append(L3**2 / (I_33 * kb_const + epsilon_const))
        except Exception:
            pass  # Skip this molecule for lib if error occurs

    # --- Hydrogen Bonds (calculated once per frame, between molecules) ---
    num_molecules_in_current_frame = len(molecules_in_frame)
    if num_molecules_in_current_frame >= 2:
        for i in range(num_molecules_in_current_frame):
            mol1_atoms_list = molecules_in_frame[i]
            O1_hb: Optional[Atom] = None
            try:  # Find Oxygen in molecule i
                for atom_m1 in mol1_atoms_list:
                    if (
                        hasattr(atom_m1, "atom_string")
                        and atom_m1.atom_string == "O"
                        and all(
                            hasattr(atom_m1, attr)
                            for attr in ["mass", "position", "velocity"]
                        )
                    ):
                        O1_hb = atom_m1
                        break
            except AttributeError:
                continue
            if O1_hb is None:
                continue

            for j in range(i + 1, num_molecules_in_current_frame):
                mol2_atoms_list = molecules_in_frame[j]
                O2_hb: Optional[Atom] = None
                try:  # Find Oxygen in molecule j
                    for atom_m2 in mol2_atoms_list:
                        if (
                            hasattr(atom_m2, "atom_string")
                            and atom_m2.atom_string == "O"
                            and all(
                                hasattr(atom_m2, attr)
                                for attr in ["mass", "position", "velocity"]
                            )
                        ):
                            O2_hb = atom_m2
                            break
                except AttributeError:
                    continue
                if O2_hb is None:
                    continue

                try:  # Calculate H-bond temperature
                    d_O1O2_vec = get_position(O1_hb) - get_position(O2_hb)
                    d_O1O2_mag = np.linalg.norm(d_O1O2_vec)
                    if epsilon_const < d_O1O2_mag < hb_cutoff_const:
                        u_O1O2_hat = d_O1O2_vec / d_O1O2_mag
                        v_rel_O1O2 = O1_hb.velocity - O2_hb.velocity
                        v_HB_scalar = np.dot(v_rel_O1O2, u_O1O2_hat)
                        mu_OO = O1_hb.mass / 2.0  # Effective mass for O-O pair
                        T_hb_frame.append((mu_OO * v_HB_scalar**2) / kb_const)
                except Exception:
                    pass  # Skip H-bond pair if error

    # Calculate average for the frame
    avg_T_stretch_exc = (
        float(np.mean(T_stretch_exc_frame)) if T_stretch_exc_frame else np.nan
    )
    avg_T_stretch_norm = (
        float(np.mean(T_stretch_norm_frame)) if T_stretch_norm_frame else np.nan
    )
    avg_T_bend = float(np.mean(T_bend_frame)) if T_bend_frame else np.nan
    avg_T_hb = (
        float(np.mean(T_hb_frame)) if T_hb_frame else np.nan
    )  # Avg H-bond temps for the frame
    avg_T_twist = float(np.mean(T_twist_mol)) if T_twist_mol else np.nan
    avg_T_wag = float(np.mean(T_wag_mol)) if T_wag_mol else np.nan
    avg_T_rock = float(np.mean(T_rock_mol)) if T_rock_mol else np.nan

    return (
        avg_T_stretch_exc,
        avg_T_stretch_norm,
        avg_T_bend,
        avg_T_hb,
        avg_T_twist,
        avg_T_wag,
        avg_T_rock,
    )


# --- Main Public Function ---
def analyze_temps_mpi(
    trajs_data: List[Frame],
    exc_indexes_file_path: str,
    unwrapped_coords: bool = False,
    num_processes: Optional[int] = None,
    kb_constant: float = DEFAULT_KB_CONSTANT,
    hb_cutoff: float = DEFAULT_HB_OO_DIST_CUTOFF,
    epsilon_val: float = DEFAULT_EPSILON,
) -> Dict[str, List[float]]:
    """
    Analyzes trajectory data to calculate and optionally plot molecular temperatures.

    Args:
        trajs_data: List of Frame objects. Each Frame contains molecules (lists of AtomType).
                    AtomType objects must have 'index', 'atom_string', 'mass',
                    'position' (np.array in Å), and 'velocity' (np.array in Å/fs).
        exc_indexes_file_path: Path to the directory containing 'nnp-indexes.dat' for excited atoms.
        time_step_fs: Time step between frames in femtoseconds. If None, plots against frame number.
        num_processes: Number of processes to use for parallel computation.
                       If None or <=0, uses all available CPU cores.
        kb_constant: Boltzmann constant in (amu * Å^2 / (fs^2 * K)).
        hb_cutoff: Cutoff distance for O-O in hydrogen bonds (Angstroms).
        epsilon_val: Small value to prevent division by zero.
        plot_results: If True, generates and shows a plot of temperature evolution.

    Returns:
        A dictionary where keys are temperature names (e.g., 'stretch_exc', 'bend')
        and values are lists of the average temperature for each frame.
    """
    if len(trajs_data) <= 0:
        raise ValueError(f"No frames provided in {trajs_data}.")

    print(f"Analyzing trajectory with {len(trajs_data)} frames.")
    overall_start_time = time.time()

    # Prepare list of (index, frame_object) for the worker
    indexed_trajs_list: List[Tuple[int, Frame]] = list(enumerate(trajs_data))

    # Load excited atom indices
    exc_indexes_set_loaded: Set[int] = set()
    try:
        exc_indexes_array = np.loadtxt(exc_indexes_file_path, dtype=int)
        if exc_indexes_array.ndim > 0:
            exc_indexes_set_loaded = set(int(i) for i in exc_indexes_array.flatten())
        else:
            exc_indexes_set_loaded = {int(exc_indexes_array)}

        print(
            f"Loaded {len(exc_indexes_set_loaded)} excited atom indices from '{exc_indexes_file_path}'."
        )
    except FileNotFoundError:
        print(
            f"Warning: File not found - '{exc_indexes_file_path}'. No excited H-atom differentiation for stretch."
        )
    except Exception as e:
        print(
            f"Error loading '{exc_indexes_file_path}': {e}. No excited H-atom differentiation."
        )

    # Determine number of processes for multiprocessing
    actual_num_processes: int
    if num_processes is None or num_processes <= 0:
        actual_num_processes = multiprocessing.cpu_count()
    else:
        actual_num_processes = num_processes
    print(f"Starting parallel processing with {actual_num_processes} processes...")

    # Create a partial function with fixed arguments for the worker
    # The worker _process_single_frame_worker will use these constants
    partial_worker_fn = functools.partial(
        _process_single_frame_worker,
        exc_indexes_set_arg=exc_indexes_set_loaded,
        unwrapped_coords=unwrapped_coords,
        kb_const=kb_constant,
        hb_cutoff_const=hb_cutoff,
        epsilon_const=epsilon_val,
    )

    pool_start_time = time.time()
    results: List[Tuple[float, float, float, float, float, float, float]]
    with multiprocessing.Pool(processes=actual_num_processes) as pool:
        results = pool.map(partial_worker_fn, indexed_trajs_list)
    pool_end_time = time.time()
    print(
        f"Parallel processing of frames finished in {pool_end_time - pool_start_time:.2f} seconds."
    )

    # Unpack results - 7 temperatures
    frame_avg_T_stretch_exc_list = [res[0] for res in results]
    frame_avg_T_stretch_norm_list = [res[1] for res in results]
    frame_avg_T_bend_list = [res[2] for res in results]
    frame_avg_T_hb_list = [res[3] for res in results]
    frame_avg_T_twist_list = [res[4] for res in results]
    frame_avg_T_wag_list = [res[5] for res in results]
    frame_avg_T_rock_list = [res[6] for res in results]

    # Robust mean calculation function
    def robust_nanmean(lst: List[float]) -> float:
        # Filters out NaNs then calculates mean; returns NaN if all are NaN or list is empty
        valid_values = [x for x in lst if not np.isnan(x)]
        return float(np.mean(valid_values)) if valid_values else np.nan

    overall_avg_T_stretch_exc = robust_nanmean(frame_avg_T_stretch_exc_list)
    overall_avg_T_stretch_norm = robust_nanmean(frame_avg_T_stretch_norm_list)
    overall_avg_T_bend = robust_nanmean(frame_avg_T_bend_list)
    overall_avg_T_hb = robust_nanmean(frame_avg_T_hb_list)
    overall_avg_T_twist = robust_nanmean(frame_avg_T_twist_list)
    overall_avg_T_wag = robust_nanmean(frame_avg_T_wag_list)
    overall_avg_T_rock = robust_nanmean(frame_avg_T_rock_list)

    print("-" * 50)
    print("OVERALL AVERAGE TEMPERATURES (K)")
    print("-" * 50)
    print(f"OH Stretching (Excited H): {overall_avg_T_stretch_exc:.2f}")
    print(f"OH Stretching (Normal H):  {overall_avg_T_stretch_norm:.2f}")
    print(f"HOH Bending:               {overall_avg_T_bend:.2f}")
    print(f"Hydrogen Bond:             {overall_avg_T_hb:.2f}")
    print(f"Librational Twist:         {overall_avg_T_twist:.2f}")
    print(f"Librational Wag:           {overall_avg_T_wag:.2f}")
    print(f"Librational Rock:          {overall_avg_T_rock:.2f}")
    print("-" * 50)

    overall_end_time = time.time()
    print(
        f"Total analysis function execution time: {overall_end_time - overall_start_time:.2f} seconds."
    )

    # Prepare data to return
    returned_data = {
        "stretch_excited_H": frame_avg_T_stretch_exc_list,
        "stretch_normal_H": frame_avg_T_stretch_norm_list,
        "bend_HOH": frame_avg_T_bend_list,
        "hydrogen_bond_OO": frame_avg_T_hb_list,
        "libration_twist": frame_avg_T_twist_list,
        "libration_wag": frame_avg_T_wag_list,
        "libration_rock": frame_avg_T_rock_list,
    }
    return returned_data
