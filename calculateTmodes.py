import time
from typing import Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from numba.typed import List as NumbaTypedList

from MDtools.dataStructures import Atom, Frame

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5
DEFAULT_EPSILON: float = 1e-40
TRIG_EPSILON: float = 1e-12


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


# @numba.jit(nopython=True, fastmath=True)
# def _process_single_frame_numba_jit(
#     positions: np.ndarray,  # (N_all_atoms, 3)
#     velocities: np.ndarray,  # (N_all_atoms, 3)
#     masses: np.ndarray,  # (N_all_atoms,)
#     molecule_indices_list_of_arrays: numba.typed.List,  # Lista di array [O_idx, H1_idx, H2_idx]
#     is_H_excited_mask: np.ndarray,  # (N_all_atoms,) maschera booleana
#     oxygen_indices_for_hb_np: np.ndarray,  # Array di indici globali degli atomi O
#     kb_const_np: float,
#     hb_cutoff_const_np: float,
#     epsilon_const_np: float,
# ) -> Tuple[float, float, float, float, float, float, float]:
#     """
#     Worker Numba JITtato per processare i dati estratti di un singolo frame.
#     """
#     sum_T_stretch_exc = 0.0
#     count_T_stretch_exc = 0
#     sum_T_stretch_norm = 0.0
#     count_T_stretch_norm = 0
#     sum_T_bend = 0.0
#     count_T_bend = 0
#     sum_T_hb = 0.0
#     count_T_hb = 0
#     sum_T_twist = 0.0
#     count_T_twist = 0
#     sum_T_wag = 0.0
#     count_T_wag = 0
#     sum_T_rock = 0.0
#     count_T_rock = 0
#
#     for mol_idx in range(len(molecule_indices_list_of_arrays)):
#         mol_atom_idxs = molecule_indices_list_of_arrays[mol_idx]
#
#         O_gidx, H1_gidx, H2_gidx = mol_atom_idxs[0], mol_atom_idxs[1], mol_atom_idxs[2]
#
#         m_O_val = masses[O_gidx]
#         m_H1_val = masses[H1_gidx]
#         mu_OH1 = (m_O_val * m_H1_val) / (m_O_val + m_H1_val)
#         d_OH1_vec_np = positions[O_gidx] - positions[H1_gidx]
#         d_OH1_mag_np = np.linalg.norm(d_OH1_vec_np)
#
#         if d_OH1_mag_np >= epsilon_const_np:
#             u_OH1_hat_np = d_OH1_vec_np / d_OH1_mag_np
#             v_rel_OH1_np = velocities[O_gidx] - velocities[H1_gidx]
#             v_stretch_scalar1 = np.dot(v_rel_OH1_np, u_OH1_hat_np)
#             temp_val1 = (mu_OH1 * v_stretch_scalar1**2) / kb_const_np
#             if is_H_excited_mask[H1_gidx]:
#                 sum_T_stretch_exc += temp_val1
#                 count_T_stretch_exc += 1
#             else:
#                 sum_T_stretch_norm += temp_val1
#                 count_T_stretch_norm += 1
#
#         m_H2_val = masses[H2_gidx]
#         mu_OH2 = (m_O_val * m_H2_val) / (m_O_val + m_H2_val)
#         d_OH2_vec_np = positions[O_gidx] - positions[H2_gidx]
#         d_OH2_mag_np = np.linalg.norm(d_OH2_vec_np)
#         if d_OH2_mag_np >= epsilon_const_np:
#             u_OH2_hat_np = d_OH2_vec_np / d_OH2_mag_np
#             v_rel_OH2_np = velocities[O_gidx] - velocities[H2_gidx]
#             v_stretch_scalar2 = np.dot(v_rel_OH2_np, u_OH2_hat_np)
#             temp_val2 = (mu_OH2 * v_stretch_scalar2**2) / kb_const_np
#             if is_H_excited_mask[H2_gidx]:
#                 sum_T_stretch_exc += temp_val2
#                 count_T_stretch_exc += 1
#             else:
#                 sum_T_stretch_norm += temp_val2
#                 count_T_stretch_norm += 1
#
#         d_O_H1_v_np = positions[H1_gidx] - positions[O_gidx]
#         d_O_H2_v_np = positions[H2_gidx] - positions[O_gidx]
#         d_O_H1_m_np = np.linalg.norm(d_O_H1_v_np)
#         d_O_H2_m_np = np.linalg.norm(d_O_H2_v_np)
#
#         if d_O_H1_m_np >= epsilon_const_np and d_O_H2_m_np >= epsilon_const_np:
#             u_O_H1_h_np = d_O_H1_v_np / d_O_H1_m_np
#             u_O_H2_h_np = d_O_H2_v_np / d_O_H2_m_np
#             v_H1_np = velocities[H1_gidx]
#             v_H2_np = velocities[H2_gidx]
#             v_H1_perp_np = v_H1_np - (np.dot(v_H1_np, u_O_H1_h_np) * u_O_H1_h_np)
#             v_H2_perp_np = v_H2_np - (np.dot(v_H2_np, u_O_H2_h_np) * u_O_H2_h_np)
#             d_H1H2_v_np = positions[H1_gidx] - positions[H2_gidx]
#             d_H1H2_m_np = np.linalg.norm(d_H1H2_v_np)
#             if d_H1H2_m_np >= epsilon_const_np:
#                 u_H1H2_h_np = d_H1H2_v_np / d_H1H2_m_np
#                 v_bend_scalar_np = np.dot(v_H1_perp_np - v_H2_perp_np, u_H1H2_h_np)
#                 mu_HH_np = masses[H1_gidx] / 2.0
#                 T_bend_val = (mu_HH_np * v_bend_scalar_np**2) / kb_const_np
#                 sum_T_bend += T_bend_val
#                 count_T_bend += 1
#
#         mol_indices_for_lib = mol_atom_idxs
#
#         # MODIFIED PART FOR ROBUST ARRAY CREATION IN NUMBA
#         masses_mol = np.array(
#             [
#                 masses[mol_indices_for_lib[0]],
#                 masses[mol_indices_for_lib[1]],
#                 masses[mol_indices_for_lib[2]],
#             ],
#             dtype=np.float64,
#         )
#
#         positions_mol = np.empty((3, 3), dtype=np.float64)
#         positions_mol[0, :] = positions[mol_indices_for_lib[0]]
#         positions_mol[1, :] = positions[mol_indices_for_lib[1]]
#         positions_mol[2, :] = positions[mol_indices_for_lib[2]]
#
#         velocities_mol = np.empty((3, 3), dtype=np.float64)
#         velocities_mol[0, :] = velocities[mol_indices_for_lib[0]]
#         velocities_mol[1, :] = velocities[mol_indices_for_lib[1]]
#         velocities_mol[2, :] = velocities[mol_indices_for_lib[2]]
#         # END OF MODIFIED PART
#
#         M_total_lib_val = np.sum(masses_mol)
#         if M_total_lib_val >= epsilon_const_np:
#             R_cm_lib_val = np.zeros(3, dtype=np.float64)  # Explicit dtype
#             V_cm_lib_val = np.zeros(3, dtype=np.float64)  # Explicit dtype
#             for k_dim in range(3):
#                 R_cm_lib_val[k_dim] = (
#                     np.sum(positions_mol[:, k_dim] * masses_mol) / M_total_lib_val
#                 )
#                 V_cm_lib_val[k_dim] = (
#                     np.sum(velocities_mol[:, k_dim] * masses_mol) / M_total_lib_val
#                 )
#
#             r_prime_lib_val = positions_mol - R_cm_lib_val
#             v_rel_cm_lib_val = velocities_mol - V_cm_lib_val
#
#             vec_OH1_lib_np = positions[H1_gidx] - positions[O_gidx]
#             vec_OH2_lib_np = positions[H2_gidx] - positions[O_gidx]
#
#             norm_OH1 = _normalize_vector_numba(vec_OH1_lib_np, epsilon_const_np)
#             norm_OH2 = _normalize_vector_numba(vec_OH2_lib_np, epsilon_const_np)
#
#             axis1_u_np = _normalize_vector_numba(norm_OH1 + norm_OH2, epsilon_const_np)
#             cross_OH1_OH2 = np.cross(vec_OH1_lib_np, vec_OH2_lib_np)
#             axis3_w_np = _normalize_vector_numba(cross_OH1_OH2, epsilon_const_np)
#             cross_w_u = np.cross(axis3_w_np, axis1_u_np)
#             axis2_v_np = _normalize_vector_numba(cross_w_u, epsilon_const_np)
#             cross_u_v_reorth = np.cross(axis1_u_np, axis2_v_np)
#             axis3_w_np = _normalize_vector_numba(cross_u_v_reorth, epsilon_const_np)
#
#             I_11, I_22, I_33 = 0.0, 0.0, 0.0
#             for i_atom_in_mol in range(3):
#                 r_p_i = r_prime_lib_val[i_atom_in_mol]
#                 m_i = masses_mol[i_atom_in_mol]
#                 r_p_dot_v = np.dot(r_p_i, axis2_v_np)
#                 r_p_dot_w = np.dot(r_p_i, axis3_w_np)
#                 I_11 += m_i * (r_p_dot_v**2 + r_p_dot_w**2)
#                 r_p_dot_u = np.dot(r_p_i, axis1_u_np)
#                 I_22 += m_i * (r_p_dot_u**2 + r_p_dot_w**2)
#                 I_33 += m_i * (r_p_dot_u**2 + r_p_dot_v**2)
#
#             L_lab_val = np.zeros(3, dtype=np.float64)  # Explicit dtype
#             for i_atom_in_mol in range(3):
#                 L_lab_val += masses_mol[i_atom_in_mol] * np.cross(
#                     r_prime_lib_val[i_atom_in_mol], v_rel_cm_lib_val[i_atom_in_mol]
#                 )
#
#             L1_val = np.dot(L_lab_val, axis1_u_np)
#             L2_val = np.dot(L_lab_val, axis2_v_np)
#             L3_val = np.dot(L_lab_val, axis3_w_np)
#
#             if I_11 > epsilon_const_np:
#                 sum_T_twist += L1_val**2 / (I_11 * kb_const_np)
#                 count_T_twist += 1
#             if I_22 > epsilon_const_np:
#                 sum_T_rock += L2_val**2 / (I_22 * kb_const_np)
#                 count_T_rock += 1
#             if I_33 > epsilon_const_np:
#                 sum_T_wag += L3_val**2 / (I_33 * kb_const_np)
#                 count_T_wag += 1
#
#     num_oxygens = len(oxygen_indices_for_hb_np)
#     if num_oxygens >= 2:
#         for i_idx_o_list in range(num_oxygens):
#             O1_gidx_hb = oxygen_indices_for_hb_np[i_idx_o_list]
#             for j_idx_o_list in range(i_idx_o_list + 1, num_oxygens):
#                 O2_gidx_hb = oxygen_indices_for_hb_np[j_idx_o_list]
#                 pos_O1 = positions[O1_gidx_hb]
#                 pos_O2 = positions[O2_gidx_hb]
#                 d_O1O2_vec_hb = pos_O1 - pos_O2
#                 d_O1O2_mag_hb = np.linalg.norm(d_O1O2_vec_hb)
#
#                 if epsilon_const_np < d_O1O2_mag_hb < hb_cutoff_const_np:
#                     u_O1O2_hat_hb = d_O1O2_vec_hb / d_O1O2_mag_hb
#                     vel_O1 = velocities[O1_gidx_hb]
#                     vel_O2 = velocities[O2_gidx_hb]
#                     v_rel_O1O2_hb = vel_O1 - vel_O2
#                     v_HB_scalar_hb = np.dot(v_rel_O1O2_hb, u_O1O2_hat_hb)
#                     m_O_for_hb = masses[O1_gidx_hb]
#                     mu_OO_hb = m_O_for_hb / 2.0
#                     sum_T_hb += (mu_OO_hb * v_HB_scalar_hb**2) / kb_const_np
#                     count_T_hb += 1
#
#     avg_T_stretch_exc = (
#         sum_T_stretch_exc / count_T_stretch_exc if count_T_stretch_exc > 0 else np.nan
#     )
#     avg_T_stretch_norm = (
#         sum_T_stretch_norm / count_T_stretch_norm
#         if count_T_stretch_norm > 0
#         else np.nan
#     )
#     avg_T_bend = sum_T_bend / count_T_bend if count_T_bend > 0 else np.nan
#     avg_T_hb = sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan
#     avg_T_twist = sum_T_twist / count_T_twist if count_T_twist > 0 else np.nan
#     avg_T_wag = sum_T_wag / count_T_wag if count_T_wag > 0 else np.nan
#     avg_T_rock = sum_T_rock / count_T_rock if count_T_rock > 0 else np.nan
#
#     return (
#         avg_T_stretch_exc,
#         avg_T_stretch_norm,
#         avg_T_bend,
#         avg_T_hb,
#         avg_T_twist,
#         avg_T_wag,
#         avg_T_rock,
#     )


@numba.jit(nopython=True, fastmath=True)
def _process_single_frame_numba_jit(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    molecule_indices_list_of_arrays: numba.typed.List,
    is_H_excited_mask: np.ndarray,
    oxygen_indices_for_hb_np: np.ndarray,
    kb_const_np: float,
    hb_cutoff_const_np: float,
    epsilon_const_np: float,
    trig_epsilon_np: float,  # Added epsilon for trig functions
) -> Tuple[
    float, float, float, float, float, float, float, float
]:  # Added one more float for T_bend_eq5
    """
    Worker Numba JITtato per processare i dati estratti di un singolo frame.
    """
    sum_T_stretch_exc = 0.0
    count_T_stretch_exc = 0
    sum_T_stretch_norm = 0.0
    count_T_stretch_norm = 0
    sum_T_bend = 0.0
    count_T_bend = 0
    sum_T_bend_eq5 = 0.0  # For Equation 5 bend
    count_T_bend_eq5 = 0  # For Equation 5 bend
    sum_T_hb = 0.0
    count_T_hb = 0
    sum_T_twist = 0.0
    count_T_twist = 0
    sum_T_wag = 0.0
    count_T_wag = 0
    sum_T_rock = 0.0
    count_T_rock = 0

    for mol_idx in range(len(molecule_indices_list_of_arrays)):
        mol_atom_idxs = molecule_indices_list_of_arrays[mol_idx]

        O_gidx, H1_gidx, H2_gidx = mol_atom_idxs[0], mol_atom_idxs[1], mol_atom_idxs[2]

        m_O_val = masses[O_gidx]
        m_H1_val = masses[H1_gidx]
        m_H2_val = masses[H2_gidx]

        # For Eq5: Total mass of the molecule
        M_total_mol = m_O_val + m_H1_val + m_H2_val

        # --- Stretch Calculations (unchanged) ---
        mu_OH1 = (m_O_val * m_H1_val) / (m_O_val + m_H1_val)
        d_OH1_vec_np = positions[O_gidx] - positions[H1_gidx]  # Vector H1->O
        # For consistency with paper's Eq.3 (vector O->H for d_OH)
        # For d_OH calculation, paper uses d_OH = vector identifying OH bond.
        # v_stretch = (v_O - v_H) . d_OH / |d_OH|
        # If d_OH points from O to H: d_OH_paper = positions[H1_gidx] - positions[O_gidx]
        d_OH1_paper_vec = positions[H1_gidx] - positions[O_gidx]
        d_OH1_mag_np = np.linalg.norm(d_OH1_paper_vec)

        if d_OH1_mag_np >= epsilon_const_np:
            u_OH1_hat_np = d_OH1_paper_vec / d_OH1_mag_np
            # v_rel_OH1_np = velocities[O_gidx] - velocities[H1_gidx] # Matches (vO - vH)
            # Paper Eq 3: v_stretch = (v_O - v_H) . (d_OH / |d_OH|)
            # My code has v_rel_OH1 = v_O - v_H1. u_OH1_hat points H->O in my original stretch.
            # To match paper for stretch:
            v_rel_OH1_for_paper_stretch = (
                velocities[H1_gidx] - velocities[O_gidx]
            )  # v_H - v_O
            # The original script calculates T_stretch using d_OH as O->H (implicit) or H->O
            # d_OH1_vec_np (H1->O). Original u_OH1_hat was H1->O. v_rel was O-H1. (O-H1).(H1->O) = -(H1-O).(H1->O)
            # This means the original stretch velocity was -(v_H1-v_O) projected on u_OH1
            # The definition in the paper uses v_stretch = (v_O - v_H) . u_OH (where u_OH is O->H)
            # Let's keep the script's original stretch definition for now, assuming it's validated,
            # but be mindful if comparing directly to paper's T_stretch plot values.
            # For internal coordinate rates (d_dot_OH1), we need (V_H1 - V_O) dot u_OH1_paper

            # Original script stretch calculation:
            d_OH1_orig_vec_np = positions[O_gidx] - positions[H1_gidx]
            d_OH1_orig_mag_np = np.linalg.norm(d_OH1_orig_vec_np)
            if d_OH1_orig_mag_np >= epsilon_const_np:
                u_OH1_orig_hat_np = d_OH1_orig_vec_np / d_OH1_orig_mag_np
                v_rel_OH1_orig_np = velocities[O_gidx] - velocities[H1_gidx]
                v_stretch_scalar1 = np.dot(v_rel_OH1_orig_np, u_OH1_orig_hat_np)
                temp_val1 = (mu_OH1 * v_stretch_scalar1**2) / kb_const_np
                if is_H_excited_mask[H1_gidx]:
                    sum_T_stretch_exc += temp_val1
                    count_T_stretch_exc += 1
                else:
                    sum_T_stretch_norm += temp_val1
                    count_T_stretch_norm += 1

        # mu_OH2 = (m_O_val * m_H2_val) / (m_O_val + m_H2_val)
        d_OH2_paper_vec = positions[H2_gidx] - positions[O_gidx]
        d_OH2_mag_np = np.linalg.norm(d_OH2_paper_vec)

        # Original script stretch calculation for H2:
        d_OH2_orig_vec_np = positions[O_gidx] - positions[H2_gidx]
        d_OH2_orig_mag_np = np.linalg.norm(d_OH2_orig_vec_np)
        if d_OH2_orig_mag_np >= epsilon_const_np:
            mu_OH2 = (m_O_val * m_H2_val) / (m_O_val + m_H2_val)  # moved here
            u_OH2_orig_hat_np = d_OH2_orig_vec_np / d_OH2_orig_mag_np
            v_rel_OH2_orig_np = velocities[O_gidx] - velocities[H2_gidx]
            v_stretch_scalar2 = np.dot(v_rel_OH2_orig_np, u_OH2_orig_hat_np)
            temp_val2 = (mu_OH2 * v_stretch_scalar2**2) / kb_const_np
            if is_H_excited_mask[H2_gidx]:
                sum_T_stretch_exc += temp_val2
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += temp_val2
                count_T_stretch_norm += 1

        # --- Bend Calculation (Original - Eq. 7 from paper) ---
        # d_O_H1_v_np points O->H1, d_O_H2_v_np points O->H2
        d_O_H1_v_np = positions[H1_gidx] - positions[O_gidx]
        d_O_H2_v_np = positions[H2_gidx] - positions[O_gidx]
        d_O_H1_m_np = np.linalg.norm(d_O_H1_v_np)  # This is d_OH1_mag_np
        d_O_H2_m_np = np.linalg.norm(d_O_H2_v_np)  # This is d_OH2_mag_np

        if d_O_H1_m_np >= epsilon_const_np and d_O_H2_m_np >= epsilon_const_np:
            u_O_H1_h_np = d_O_H1_v_np / d_O_H1_m_np  # unit vec O->H1
            u_O_H2_h_np = d_O_H2_v_np / d_O_H2_m_np  # unit vec O->H2
            v_H1_np = velocities[H1_gidx]
            v_H2_np = velocities[H2_gidx]
            v_H1_perp_np = v_H1_np - (np.dot(v_H1_np, u_O_H1_h_np) * u_O_H1_h_np)
            v_H2_perp_np = v_H2_np - (np.dot(v_H2_np, u_O_H2_h_np) * u_O_H2_h_np)

            d_H1H2_v_np = positions[H1_gidx] - positions[H2_gidx]  # H2->H1
            d_H1H2_m_np = np.linalg.norm(d_H1H2_v_np)
            if d_H1H2_m_np >= epsilon_const_np:
                u_H1H2_h_np = d_H1H2_v_np / d_H1H2_m_np
                v_bend_scalar_np = np.dot(v_H1_perp_np - v_H2_perp_np, u_H1H2_h_np)
                mu_HH_np = (m_H1_val * m_H2_val) / (
                    m_H1_val + m_H2_val
                )  # More general reduced mass
                # Original script was: mu_HH_np = masses[H1_gidx] / 2.0, assumes m_H1=m_H2
                T_bend_val = (mu_HH_np * v_bend_scalar_np**2) / kb_const_np
                sum_T_bend += T_bend_val
                count_T_bend += 1

        # --- Bend Calculation (New - Eq. 5 from paper) ---
        # Needs: d_OH1_paper_vec, d_OH1_mag_np, d_OH2_paper_vec, d_OH2_mag_np (calculated above for stretch part)
        # u_OH1_paper_hat, u_OH2_paper_hat
        if (
            d_OH1_mag_np >= epsilon_const_np
            and d_OH2_mag_np >= epsilon_const_np
            and M_total_mol > epsilon_const_np
        ):
            u_OH1_paper_hat = d_OH1_paper_vec / d_OH1_mag_np  # O -> H1 unit vector
            u_OH2_paper_hat = d_OH2_paper_vec / d_OH2_mag_np  # O -> H2 unit vector

            # Angle theta at O
            cos_theta_val = np.dot(u_OH1_paper_hat, u_OH2_paper_hat)
            if cos_theta_val > 1.0:
                cos_theta_val = 1.0
            if cos_theta_val < -1.0:
                cos_theta_val = -1.0
            theta_val = np.arccos(cos_theta_val)
            sin_theta_val = np.sin(theta_val)

            # Velocities of atoms
            v_O_lab = velocities[O_gidx]
            v_H1_lab = velocities[H1_gidx]
            v_H2_lab = velocities[H2_gidx]

            # Rates of change of internal coordinates: d_dot_OH1, d_dot_OH2, theta_dot
            # d_dot = (v_H - v_O) . u_OH_paper
            d_dot_OH1 = np.dot(v_H1_lab - v_O_lab, u_OH1_paper_hat)
            d_dot_OH2 = np.dot(v_H2_lab - v_O_lab, u_OH2_paper_hat)

            theta_dot = 0.0  # Default if sin_theta_val is too small
            if abs(sin_theta_val) > trig_epsilon_np:
                # u_dot = ( (v_H-v_O) - u_OH_paper * d_dot_OH ) / d_OH
                u_dot_OH1_vec = (
                    (v_H1_lab - v_O_lab) - u_OH1_paper_hat * d_dot_OH1
                ) / d_OH1_mag_np
                u_dot_OH2_vec = (
                    (v_H2_lab - v_O_lab) - u_OH2_paper_hat * d_dot_OH2
                ) / d_OH2_mag_np
                theta_dot = (
                    -(
                        np.dot(u_dot_OH1_vec, u_OH2_paper_hat)
                        + np.dot(u_OH1_paper_hat, u_dot_OH2_vec)
                    )
                    / sin_theta_val
                )

            # 2D representation velocities from Eq.4 derivatives
            # c_h = cos(theta/2), s_h = sin(theta/2)
            # d1 = d_OH1_mag_np, d2 = d_OH2_mag_np
            theta_half = theta_val / 2.0
            c_h = np.cos(theta_half)
            s_h = np.sin(theta_half)

            # Using specific masses as discussed in thought process
            # y_O_dot = (1/M_total_mol) * (m_H1_val * (d_dot_OH1*c_h - d_OH1_mag_np*s_h*theta_dot/2) + \
            #                              m_H2_val * (d_dot_OH2*c_h - d_OH2_mag_np*s_h*theta_dot/2) )
            # Paper uses generic m_H for these terms in r_O.
            # Let's use the m_H from paper's formula for m_H/M terms in r_O, r_H1, r_H2 definition.
            # If m_H1 != m_H2, the original formula needs clarification. Assuming average m_H for m_H/M term.
            # For simplicity and adherence to paper's m_H/M, assume m_H is average H mass or m_H1.
            # Let's use m_H1_val where paper indicates m_H in m_H/M for consistency with structure of eq 4.
            # The mass M in paper for Eq. 4 is m_O + 2*m_H. Here M_total_mol is m_O+m_H1+m_H2.
            # If the formula implies m_H1 and m_H2 can be different for the d_OH1 and d_OH2 terms in y_O:
            # y_O_dot formulation from my thought process:
            # (m_H_for_M_calc / M_total_mol) used for m_H/M terms. Let m_H_for_M_calc = m_H1_val for now.

            # Re-evaluating Eq. 4: ro = {0, mH/M dOH1 cos(t/2) + mH/M dOH2 cos(t/2)}
            # This mH/M suggests M = mO + 2mH.
            # And mH is a single hydrogen mass.
            # The individual masses m_O_val, m_H1_val, m_H2_val are for K.E. (Eq. 5).
            # For terms like m_O/M, m_H/M in Eq.4, use M_total_mol as M, and m_H1_val as representative m_H.

            term_mH_over_M = m_H1_val / M_total_mol  # Approximation if mH1~mH2
            term_mO_over_M = m_O_val / M_total_mol

            # Velocities (Eq. 4 derivatives)
            # v_O_x_dot = 0.0
            v_O_y_dot = term_mH_over_M * (
                (d_dot_OH1 * c_h - d_OH1_mag_np * s_h * (theta_dot / 2.0))
                + (d_dot_OH2 * c_h - d_OH2_mag_np * s_h * (theta_dot / 2.0))
            )

            v_H1_x_dot = d_dot_OH1 * s_h + d_OH1_mag_np * c_h * (theta_dot / 2.0)
            v_H1_y_dot = -term_mO_over_M * (
                d_dot_OH1 * c_h - d_OH1_mag_np * s_h * (theta_dot / 2.0)
            )

            # Using literal form of r_H2 from paper for x component
            v_H2_x_dot = d_dot_OH2 * s_h + d_OH2_mag_np * c_h * (theta_dot / 2.0)
            v_H2_y_dot = -term_mO_over_M * (
                d_dot_OH2 * c_h - d_OH2_mag_np * s_h * (theta_dot / 2.0)
            )

            # Kinetic Energy (Eq. 5)
            v_O_sq = v_O_y_dot**2  # v_O_x_dot is 0
            v_H1_sq = v_H1_x_dot**2 + v_H1_y_dot**2
            v_H2_sq = v_H2_x_dot**2 + v_H2_y_dot**2

            K_bend_eq5 = 0.5 * (
                m_O_val * v_O_sq + m_H1_val * v_H1_sq + m_H2_val * v_H2_sq
            )
            T_bend_eq5_val = K_bend_eq5 / (3 / 2 * kb_const_np)

            sum_T_bend_eq5 += T_bend_eq5_val
            count_T_bend_eq5 += 1

        # --- Librations (unchanged) ---
        mol_indices_for_lib = mol_atom_idxs
        masses_mol = np.array(
            [
                masses[mol_indices_for_lib[0]],
                masses[mol_indices_for_lib[1]],
                masses[mol_indices_for_lib[2]],
            ],
            dtype=np.float64,
        )

        positions_mol = np.empty((3, 3), dtype=np.float64)
        positions_mol[0, :] = positions[mol_indices_for_lib[0]]
        positions_mol[1, :] = positions[mol_indices_for_lib[1]]
        positions_mol[2, :] = positions[mol_indices_for_lib[2]]

        velocities_mol = np.empty((3, 3), dtype=np.float64)
        velocities_mol[0, :] = velocities[mol_indices_for_lib[0]]
        velocities_mol[1, :] = velocities[mol_indices_for_lib[1]]
        velocities_mol[2, :] = velocities[mol_indices_for_lib[2]]

        M_total_lib_val = np.sum(masses_mol)
        if M_total_lib_val >= epsilon_const_np:
            R_cm_lib_val = np.zeros(3, dtype=np.float64)
            V_cm_lib_val = np.zeros(3, dtype=np.float64)
            for k_dim in range(3):
                R_cm_lib_val[k_dim] = (
                    np.sum(positions_mol[:, k_dim] * masses_mol) / M_total_lib_val
                )
                V_cm_lib_val[k_dim] = (
                    np.sum(velocities_mol[:, k_dim] * masses_mol) / M_total_lib_val
                )

            r_prime_lib_val = positions_mol - R_cm_lib_val
            v_rel_cm_lib_val = velocities_mol - V_cm_lib_val

            # d_O_H1_v_np and d_O_H2_v_np are already O->H1 and O->H2
            vec_OH1_lib_np = d_O_H1_v_np
            vec_OH2_lib_np = d_O_H2_v_np

            norm_OH1 = _normalize_vector_numba(vec_OH1_lib_np, epsilon_const_np)
            norm_OH2 = _normalize_vector_numba(vec_OH2_lib_np, epsilon_const_np)

            axis1_u_np = _normalize_vector_numba(norm_OH1 + norm_OH2, epsilon_const_np)
            cross_OH1_OH2 = np.cross(
                vec_OH1_lib_np, vec_OH2_lib_np
            )  # Uses O->H1, O->H2 vectors
            axis3_w_np = _normalize_vector_numba(cross_OH1_OH2, epsilon_const_np)
            cross_w_u = np.cross(axis3_w_np, axis1_u_np)
            axis2_v_np = _normalize_vector_numba(cross_w_u, epsilon_const_np)
            cross_u_v_reorth = np.cross(axis1_u_np, axis2_v_np)  # Re-orthogonalize w
            axis3_w_np = _normalize_vector_numba(cross_u_v_reorth, epsilon_const_np)

            I_11, I_22, I_33 = 0.0, 0.0, 0.0
            for i_atom_in_mol in range(3):
                r_p_i = r_prime_lib_val[i_atom_in_mol]
                m_i = masses_mol[i_atom_in_mol]
                r_p_dot_v = np.dot(r_p_i, axis2_v_np)
                r_p_dot_w = np.dot(r_p_i, axis3_w_np)
                I_11 += m_i * (r_p_dot_v**2 + r_p_dot_w**2)
                r_p_dot_u = np.dot(r_p_i, axis1_u_np)
                I_22 += m_i * (r_p_dot_u**2 + r_p_dot_w**2)
                I_33 += m_i * (r_p_dot_u**2 + r_p_dot_v**2)

            L_lab_val = np.zeros(3, dtype=np.float64)
            for i_atom_in_mol in range(3):
                L_lab_val += masses_mol[i_atom_in_mol] * np.cross(
                    r_prime_lib_val[i_atom_in_mol], v_rel_cm_lib_val[i_atom_in_mol]
                )

            L1_val = np.dot(L_lab_val, axis1_u_np)
            L2_val = np.dot(L_lab_val, axis2_v_np)
            L3_val = np.dot(L_lab_val, axis3_w_np)

            if I_11 > epsilon_const_np:  # Use epsilon_const_np for inertia
                sum_T_twist += (
                    L1_val** 2 / (I_11 * kb_const_np)
                )  # Paper uses 1/2 I w^2 / kb = L^2/(2 I kb). Check if factor of 2 is needed.
                # Original script: L^2 / (I * kb). This corresponds to 2 * K.E / kb.
                # If T = 2 K.E / kb, then it's correct. If T = K.E / (0.5 kb), also correct.
                # Standard K.E_rot = 0.5 * I * w^2.  T = I * w^2 / kb. So L^2 / (I * kb). This is correct.
                count_T_twist += 1
            if I_22 > epsilon_const_np:
                sum_T_rock += L2_val**2 / (I_22 * kb_const_np)
                count_T_rock += 1
            if I_33 > epsilon_const_np:
                sum_T_wag += L3_val**2 / (I_33 * kb_const_np)
                count_T_wag += 1

    # --- Hydrogen Bond (unchanged) ---
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
                    m_O_for_hb = masses[O1_gidx_hb]  # mass of one oxygen
                    mu_OO_hb = (
                        m_O_for_hb / 2.0
                    )  # Reduced mass for two identical oxygens
                    sum_T_hb += (mu_OO_hb * v_HB_scalar_hb**2) / kb_const_np
                    count_T_hb += 1

    avg_T_stretch_exc = (
        sum_T_stretch_exc / count_T_stretch_exc if count_T_stretch_exc > 0 else np.nan
    )
    avg_T_stretch_norm = (
        sum_T_stretch_norm / count_T_stretch_norm
        if count_T_stretch_norm > 0
        else np.nan
    )
    avg_T_bend = sum_T_bend / count_T_bend if count_T_bend > 0 else np.nan
    avg_T_bend_eq5 = (
        sum_T_bend_eq5 / count_T_bend_eq5 if count_T_bend_eq5 > 0 else np.nan
    )  # Avg for Eq5 bend
    avg_T_hb = sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan
    avg_T_twist = sum_T_twist / count_T_twist if count_T_twist > 0 else np.nan
    avg_T_wag = sum_T_wag / count_T_wag if count_T_wag > 0 else np.nan
    avg_T_rock = sum_T_rock / count_T_rock if count_T_rock > 0 else np.nan

    return (
        avg_T_stretch_exc,
        avg_T_stretch_norm,
        avg_T_bend,
        avg_T_bend_eq5,  # Added new temperature
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
    trig_epsilon_np: float = TRIG_EPSILON,
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
    results_array = np.full((num_frames, 8), np.nan, dtype=np.float64)

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
                trig_epsilon_np,
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
    frame_avg_T_bend_eq5_list = list(results_array[:, 3])  # Nuovo per Eq5
    frame_avg_T_hb_list = list(results_array[:, 4])
    frame_avg_T_twist_list = list(results_array[:, 5])
    frame_avg_T_wag_list = list(results_array[:, 6])
    frame_avg_T_rock_list = list(results_array[:, 7])

    # Calcola medie complessive (come nel tuo codice originale)
    # ... (uso di robust_nanmean) ...
    # Esempio:
    def robust_nanmean(lst: List[float]) -> float:
        valid_values = [x for x in lst if not np.isnan(x)]
        return float(np.mean(valid_values)) if valid_values else np.nan

    print_overall_avg_T_stretch_exc = robust_nanmean(frame_avg_T_stretch_exc_list)
    print_overall_avg_T_stretch_norm = robust_nanmean(frame_avg_T_stretch_norm_list)
    print_overall_avg_T_bend = robust_nanmean(frame_avg_T_bend_list)
    print_overall_avg_T_bend_eq5 = robust_nanmean(
        frame_avg_T_bend_eq5_list
    )  # Nuovo per Eq5
    print_overall_avg_T_hb = robust_nanmean(frame_avg_T_hb_list)
    print_overall_avg_T_twist = robust_nanmean(frame_avg_T_twist_list)
    print_overall_avg_T_wag = robust_nanmean(frame_avg_T_wag_list)
    print_overall_avg_T_rock = robust_nanmean(frame_avg_T_rock_list)

    print(
        f"Overall Avg T_stretch_exc (Numba): {print_overall_avg_T_stretch_exc:.2f} K (esempio)"
    )
    print(
        f"Overall Avg T_stretch_norm (Numba): {print_overall_avg_T_stretch_norm:.2f} K"
    )
    print(f"Overall Avg T_bend (Numba): {print_overall_avg_T_bend:.2f} K")
    print(
        f"Overall Avg T_bend_eq5 (Numba): {print_overall_avg_T_bend_eq5:.2f} K"
    )  # Nuovo per Eq5
    print(f"Overall Avg T_hb (Numba): {print_overall_avg_T_hb:.2f} K")
    print(f"Overall Avg T_twist (Numba): {print_overall_avg_T_twist:.2f} K")
    print(f"Overall Avg T_wag (Numba): {print_overall_avg_T_wag:.2f} K")
    print(f"Overall Avg T_rock (Numba): {print_overall_avg_T_rock:.2f} K")

    overall_end_time = time.time()
    print(
        f"Tempo totale esecuzione funzione Numba: {overall_end_time - overall_start_time:.2f} secondi."
    )

    returned_data = {
        "stretch_excited_H": frame_avg_T_stretch_exc_list,
        "stretch_normal_H": frame_avg_T_stretch_norm_list,
        "bend_HOH": frame_avg_T_bend_list,
        "bend_HOH_eq5": frame_avg_T_bend_eq5_list,  # Nuovo per Eq5
        "hb": frame_avg_T_hb_list,
        "libr_twist": frame_avg_T_twist_list,
        "libr_wag": frame_avg_T_wag_list,
        "libr_rock": frame_avg_T_rock_list,
    }
    return returned_data
