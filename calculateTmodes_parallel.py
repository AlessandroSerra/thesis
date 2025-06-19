import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from numba.typed import List as NumbaTypedList

# Assuming MDtools.dataStructures is in your PYTHONPATH or installed
# If not, you might need to copy the Atom and Frame dataclass definitions here
# or ensure this script can find dataStructures.py
from MDtools.dataStructures import Atom, Frame

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5
DEFAULT_EPSILON: float = 1e-40
TRIG_EPSILON: float = 1e-12


# ------------------------------------------------------------------
#                       DATA PREPARATION FOR NUMBA
# ------------------------------------------------------------------


def _prepare_frame_data_for_numba(
    current_frame: Frame,
    exc_indexes_set: Set[int],
    unwrapped_coords: bool,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    Estrae i dati da un oggetto Frame in array NumPy per l'uso con Numba.
    Restituisce una tupla di array:
    (positions, velocities, masses, atom_numeric_types, original_indices,
     molecule_atom_indices_list, is_H_excited_flat_mask, oxygen_indices_for_hb)
    o None se il frame non può essere processato.
    """

    atoms_in_frame_flat: List[Atom] = []
    for mol_list in current_frame.molecules:
        atoms_in_frame_flat.extend(mol_list)

    if not atoms_in_frame_flat:
        return None

    num_all_atoms = len(atoms_in_frame_flat)
    positions_np = np.zeros((num_all_atoms, 3), dtype=np.float64)
    velocities_np = np.zeros((num_all_atoms, 3), dtype=np.float64)
    masses_np = np.zeros(num_all_atoms, dtype=np.float64)
    atom_numeric_types_np = np.full(num_all_atoms, -1, dtype=np.int32)
    original_indices_np = np.zeros(num_all_atoms, dtype=np.int32)
    is_H_excited_flat_mask_np = np.zeros(num_all_atoms, dtype=np.bool_)

    for i, atom_obj in enumerate(atoms_in_frame_flat):
        if not all(
            hasattr(atom_obj, attr)
            for attr in ["index", "mass", "position", "velocity", "atom_string"]
        ):
            continue

        pos_to_use = atom_obj.position
        if (
            unwrapped_coords
            and hasattr(atom_obj, "unwrapped_position")
            and atom_obj.unwrapped_position is not None
        ):
            pos_to_use = atom_obj.unwrapped_position

        positions_np[i, :] = pos_to_use
        if atom_obj.velocity is not None:
            velocities_np[i, :] = atom_obj.velocity

        masses_np[i] = atom_obj.mass
        original_indices_np[i] = atom_obj.index

        if atom_obj.atom_string == "O":
            atom_numeric_types_np[i] = 0
        elif atom_obj.atom_string == "H":
            atom_numeric_types_np[i] = 1
            if atom_obj.index in exc_indexes_set:
                is_H_excited_flat_mask_np[i] = True

    molecule_atom_indices_list_for_numba: List[np.ndarray] = []
    current_flat_idx = 0
    for mol_list_original in current_frame.molecules:
        if not isinstance(mol_list_original, list) or len(mol_list_original) != 3:
            current_flat_idx += (
                len(mol_list_original) if isinstance(mol_list_original, list) else 0
            )
            continue

        o_idx_flat, h1_idx_flat, h2_idx_flat = -1, -1, -1
        temp_h_indices_flat = []

        mol_atom_indices_in_flat_array = list(
            range(current_flat_idx, current_flat_idx + len(mol_list_original))
        )

        for k, local_atom_idx_in_flat_array in enumerate(
            mol_atom_indices_in_flat_array
        ):
            if atom_numeric_types_np[local_atom_idx_in_flat_array] == 0:
                o_idx_flat = local_atom_idx_in_flat_array
            elif atom_numeric_types_np[local_atom_idx_in_flat_array] == 1:
                temp_h_indices_flat.append(local_atom_idx_in_flat_array)

        if o_idx_flat != -1 and len(temp_h_indices_flat) == 2:
            h1_idx_flat, h2_idx_flat = temp_h_indices_flat[0], temp_h_indices_flat[1]
            molecule_atom_indices_list_for_numba.append(
                np.array([o_idx_flat, h1_idx_flat, h2_idx_flat], dtype=np.int32)
            )
        current_flat_idx += len(mol_list_original)

    # Prepare oxygen_indices_for_hb_np_arg
    temp_oxygen_indices = []
    for i in range(num_all_atoms):
        if atom_numeric_types_np[i] == 0:  # 0 for Oxygen
            temp_oxygen_indices.append(i)
    oxygen_indices_for_hb_np_arg = np.array(temp_oxygen_indices, dtype=np.int32)

    return (
        positions_np,
        velocities_np,
        masses_np,
        atom_numeric_types_np,
        original_indices_np,
        molecule_atom_indices_list_for_numba,
        is_H_excited_flat_mask_np,
        oxygen_indices_for_hb_np_arg,
    )


# ------------------------------------------------------------------
#                       NUMBA JIT IMPLEMENTATION
# ------------------------------------------------------------------


@numba.jit(nopython=True, fastmath=False)
def _normalize_vector_numba(v_np: np.ndarray, epsilon_np: float) -> np.ndarray:
    norm_val = np.linalg.norm(v_np)
    if norm_val < epsilon_np:
        return np.zeros_like(v_np)
    return v_np / norm_val


@numba.jit(nopython=True, fastmath=False)
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
    trig_epsilon_np: float,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Worker Numba JITtato per processare i dati estratti di un singolo frame.
    Returns 13 temperature values.
    """
    sum_T_stretch_exc = 0.0
    count_T_stretch_exc = 0
    sum_T_stretch_norm = 0.0
    count_T_stretch_norm = 0
    sum_T_bend_norm = 0.0
    count_T_bend_norm = 0
    sum_T_bend_exc = 0.0
    count_T_bend_exc = 0
    sum_T_bend_eq5_norm = 0.0
    count_T_bend_eq5_norm = 0
    sum_T_bend_eq5_exc = 0.0
    count_T_bend_eq5_exc = 0
    sum_T_bend_I_norm = 0.0
    count_T_bend_I_norm = 0
    sum_T_bend_I_exc = 0.0
    count_T_bend_I_exc = 0
    sum_T_hb = 0.0
    count_T_hb = 0
    sum_T_twist_norm = 0.0
    count_T_twist_norm = 0
    sum_T_twist_exc = 0.0
    count_T_twist_exc = 0
    sum_T_wag_norm = 0.0
    count_T_wag_norm = 0
    sum_T_wag_exc = 0.0
    count_T_wag_exc = 0
    sum_T_rock_norm = 0.0
    count_T_rock_norm = 0
    sum_T_rock_exc = 0.0
    count_T_rock_exc = 0

    for mol_idx in range(len(molecule_indices_list_of_arrays)):
        mol_atom_idxs = molecule_indices_list_of_arrays[mol_idx]

        O_gidx, H1_gidx, H2_gidx = (
            mol_atom_idxs[0],
            mol_atom_idxs[1],
            mol_atom_idxs[2],
        )

        m_O_val = masses[O_gidx]
        m_H1_val = masses[H1_gidx]
        m_H2_val = masses[H2_gidx]
        M_total_mol = m_O_val + m_H1_val + m_H2_val

        # --- Stretch Calculations ---
        # H1 atom
        d_OH1_paper_vec = positions[H1_gidx] - positions[O_gidx]
        d_OH1_mag_np = np.linalg.norm(d_OH1_paper_vec)
        if d_OH1_mag_np >= epsilon_const_np:
            mu_OH1 = (m_O_val * m_H1_val) / (m_O_val + m_H1_val)
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
        # H2 atom
        d_OH2_paper_vec = positions[H2_gidx] - positions[O_gidx]
        d_OH2_mag_np = np.linalg.norm(d_OH2_paper_vec)
        if (
            d_OH2_mag_np >= epsilon_const_np
        ):  # Check if d_OH2_mag_np needs this check like d_OH1_mag_np. Added for safety.
            mu_OH2 = (m_O_val * m_H2_val) / (m_O_val + m_H2_val)
            d_OH2_orig_vec_np = positions[O_gidx] - positions[H2_gidx]
            d_OH2_orig_mag_np = np.linalg.norm(d_OH2_orig_vec_np)
            if d_OH2_orig_mag_np >= epsilon_const_np:
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

        # --- Bend Calculation (vels - Eq. 7 from paper if applicable) ---
        d_O_H1_v_np = positions[H1_gidx] - positions[O_gidx]  # vettore distanza O->H1
        d_O_H2_v_np = positions[H2_gidx] - positions[O_gidx]  # vettore distanza O->H2
        d_O_H1_m_np = np.linalg.norm(d_O_H1_v_np)  # modulo della distanza O->H1
        d_O_H2_m_np = np.linalg.norm(d_O_H2_v_np)  # modulo della distanza O->H2

        if d_O_H1_m_np >= epsilon_const_np and d_O_H2_m_np >= epsilon_const_np:
            u_O_H1_h_np = d_O_H1_v_np / d_O_H1_m_np  # versore O->H1
            u_O_H2_h_np = d_O_H2_v_np / d_O_H2_m_np  # versore O->H2
            v_H1_np = velocities[H1_gidx]  # velocity H1
            v_H2_np = velocities[H2_gidx]  # velocity H2
            v_H1_perp_np = v_H1_np - (
                np.dot(v_H1_np, u_O_H1_h_np) * u_O_H1_h_np
            )  # velocita' perpendicolare H1
            v_H2_perp_np = v_H2_np - (
                np.dot(v_H2_np, u_O_H2_h_np) * u_O_H2_h_np
            )  # velocita' perpendicolare H2

            d_H1H2_v_np = (
                positions[H1_gidx] - positions[H2_gidx]
            )  # vettore distanza H1->H2
            d_H1H2_m_np = np.linalg.norm(d_H1H2_v_np)  # modulo della distanza H1->H2
            if d_H1H2_m_np >= epsilon_const_np:
                u_H1H2_h_np = d_H1H2_v_np / d_H1H2_m_np  # versore H1->H2
                delta_H1H2_v_np = (
                    v_H1_perp_np - v_H2_perp_np
                )  # differenza delle velocita' perpendicolari
                v_bend_scalar_np = np.dot(
                    delta_H1H2_v_np, u_H1H2_h_np
                )  # differenza di velocita' proiettata su H1->H2

                # NOTE: ho provato a togliere massa ridotta e mettere solo massa H
                mu_HH_np = (m_H1_val * m_H2_val) / (m_H1_val + m_H2_val)
                T_bend_val = (mu_HH_np * v_bend_scalar_np**2) / kb_const_np

                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_exc += T_bend_val
                    count_T_bend_exc += 1
                else:
                    sum_T_bend_norm += T_bend_val
                    count_T_bend_norm += 1

        # --- Bend Calculation (angle - Eq. 5 from paper) ---
        # d_OH1_mag_np and d_OH2_mag_np are norms of O->H1 and O->H2 respectively.
        # These are d_O_H1_m_np and d_O_H2_m_np calculated above for "Bend (vels)"
        if (
            d_O_H1_m_np >= epsilon_const_np  # Using d_O_H1_m_np for d_OH1_mag_np
            and d_O_H2_m_np >= epsilon_const_np  # Using d_O_H2_m_np for d_OH2_mag_np
            and M_total_mol > epsilon_const_np
        ):
            # u_OH1_paper_hat is unit vector O->H1, u_OH2_paper_hat is O->H2
            u_OH1_paper_hat = d_O_H1_v_np / d_O_H1_m_np  # This is u_O_H1_h_np
            u_OH2_paper_hat = d_O_H2_v_np / d_O_H2_m_np  # This is u_O_H2_h_np

            cos_theta_val = np.dot(u_OH1_paper_hat, u_OH2_paper_hat)
            cos_theta_val = min(1.0, max(-1.0, cos_theta_val))  # Clamp value
            theta_val = np.arccos(cos_theta_val)
            sin_theta_val = np.sin(theta_val)

            v_O_lab = velocities[O_gidx]
            v_H1_lab = velocities[H1_gidx]
            v_H2_lab = velocities[H2_gidx]

            d_dot_OH1 = np.dot(v_H1_lab - v_O_lab, u_OH1_paper_hat)
            d_dot_OH2 = np.dot(v_H2_lab - v_O_lab, u_OH2_paper_hat)

            theta_dot = 0.0
            if abs(sin_theta_val) > trig_epsilon_np:
                # Using d_O_H1_m_np for d_OH1_mag_np and d_O_H2_m_np for d_OH2_mag_np
                u_dot_OH1_vec = (
                    (v_H1_lab - v_O_lab) - u_OH1_paper_hat * d_dot_OH1
                ) / d_O_H1_m_np
                u_dot_OH2_vec = (
                    (v_H2_lab - v_O_lab) - u_OH2_paper_hat * d_dot_OH2
                ) / d_O_H2_m_np
                theta_dot = (
                    -(
                        np.dot(u_dot_OH1_vec, u_OH2_paper_hat)
                        + np.dot(u_OH1_paper_hat, u_dot_OH2_vec)
                    )
                ) / sin_theta_val

            theta_half = theta_val / 2.0
            c_h = np.cos(theta_half)
            s_h = np.sin(theta_half)

            # Assuming m_H is represented by m_H1_val for mH/M terms as in original code.
            # M_total_mol is m_O + m_H1 + m_H2.
            term_mH_over_M = m_H1_val / M_total_mol
            term_mO_over_M = m_O_val / M_total_mol

            # Velocities (Eq. 4 derivatives from paper)
            # Using d_O_H1_m_np for d_OH1_mag_np, d_O_H2_m_np for d_OH2_mag_np
            v_O_y_dot = term_mH_over_M * (
                (d_dot_OH1 * c_h - d_O_H1_m_np * s_h * (theta_dot / 2.0))
                + (d_dot_OH2 * c_h - d_O_H2_m_np * s_h * (theta_dot / 2.0))
            )
            v_H1_x_dot = d_dot_OH1 * s_h + d_O_H1_m_np * c_h * (theta_dot / 2.0)
            v_H1_y_dot = -term_mO_over_M * (
                d_dot_OH1 * c_h - d_O_H1_m_np * s_h * (theta_dot / 2.0)
            )
            v_H2_x_dot = d_dot_OH2 * s_h + d_O_H2_m_np * c_h * (
                theta_dot / 2.0
            )  # Corrected variable for H2 x-component
            v_H2_y_dot = -term_mO_over_M * (
                d_dot_OH2 * c_h - d_O_H2_m_np * s_h * (theta_dot / 2.0)
            )

            K_bend_eq5 = 0.5 * (
                m_O_val * (v_O_y_dot**2)
                + m_H1_val * (v_H1_x_dot**2 + v_H1_y_dot**2)
                + m_H2_val * (v_H2_x_dot**2 + v_H2_y_dot**2)
            )
            T_bend_eq5_val = K_bend_eq5 / (3 / 2 * kb_const_np)

            if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                sum_T_bend_eq5_exc += T_bend_eq5_val
                count_T_bend_eq5_exc += 1
            else:
                sum_T_bend_eq5_norm += T_bend_eq5_val
                count_T_bend_eq5_norm += 1

            # calcolo con momento di inerzia
            d_OH_equil = 0.9572  # Angstrom, distanza di equilibrio O-H
            I_mom_HH = m_H1_val * d_OH_equil**2 + m_H2_val * d_OH_equil**2
            K_bend_I = 0.5 * I_mom_HH * (theta_dot**2)
            T_bend_I = 0.5 * K_bend_I / (kb_const_np)

            if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                sum_T_bend_I_exc += T_bend_I
                count_T_bend_I_exc += 1
            else:
                sum_T_bend_I_norm += T_bend_I
                count_T_bend_I_norm += 1

        # --- Librations ---
        masses_mol = np.array([m_O_val, m_H1_val, m_H2_val], dtype=np.float64)
        positions_mol = np.empty((3, 3), dtype=np.float64)
        positions_mol[0, :] = positions[O_gidx]
        positions_mol[1, :] = positions[H1_gidx]
        positions_mol[2, :] = positions[H2_gidx]
        velocities_mol = np.empty((3, 3), dtype=np.float64)
        velocities_mol[0, :] = velocities[O_gidx]
        velocities_mol[1, :] = velocities[H1_gidx]
        velocities_mol[2, :] = velocities[H2_gidx]

        M_total_lib_val = M_total_mol  # Already calculated
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

            vec_OH1_lib_np = d_O_H1_v_np  # O->H1 vector
            vec_OH2_lib_np = d_O_H2_v_np  # O->H2 vector

            norm_OH1 = _normalize_vector_numba(vec_OH1_lib_np, epsilon_const_np)
            norm_OH2 = _normalize_vector_numba(vec_OH2_lib_np, epsilon_const_np)

            axis1_u_np = _normalize_vector_numba(norm_OH1 + norm_OH2, epsilon_const_np)
            cross_OH1_OH2 = np.cross(vec_OH1_lib_np, vec_OH2_lib_np)
            axis3_w_np = _normalize_vector_numba(cross_OH1_OH2, epsilon_const_np)
            cross_w_u = np.cross(axis3_w_np, axis1_u_np)
            axis2_v_np = _normalize_vector_numba(cross_w_u, epsilon_const_np)

            # Re-orthogonalize w for robustness (as in original code)
            axis3_w_np = _normalize_vector_numba(
                np.cross(axis1_u_np, axis2_v_np), epsilon_const_np
            )

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

            mol_is_excited = is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]

            if I_11 > epsilon_const_np:
                T_twist_val = L1_val**2 / (I_11 * kb_const_np)
                if mol_is_excited:
                    sum_T_twist_exc += T_twist_val
                    count_T_twist_exc += 1
                else:
                    sum_T_twist_norm += T_twist_val
                    count_T_twist_norm += 1
            if I_22 > epsilon_const_np:
                T_rock_val = L2_val**2 / (I_22 * kb_const_np)
                if mol_is_excited:
                    sum_T_rock_exc += T_rock_val
                    count_T_rock_exc += 1
                else:
                    sum_T_rock_norm += T_rock_val
                    count_T_rock_norm += 1
            if I_33 > epsilon_const_np:
                T_wag_val = L3_val**2 / (I_33 * kb_const_np)
                if mol_is_excited:
                    sum_T_wag_exc += T_wag_val
                    count_T_wag_exc += 1
                else:
                    sum_T_wag_norm += T_wag_val
                    count_T_wag_norm += 1

    # --- Hydrogen Bond ---
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
                    m_O_for_hb = masses[O1_gidx_hb]
                    mu_OO_hb = m_O_for_hb / 2.0
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
    avg_T_bend_exc = (
        sum_T_bend_exc / count_T_bend_exc if count_T_bend_exc > 0 else np.nan
    )
    avg_T_bend_norm = (
        sum_T_bend_norm / count_T_bend_norm if count_T_bend_norm > 0 else np.nan
    )
    avg_T_bend_eq5_exc = (
        sum_T_bend_eq5_exc / count_T_bend_eq5_exc
        if count_T_bend_eq5_exc > 0
        else np.nan
    )
    avg_T_bend_eq5_norm = (
        sum_T_bend_eq5_norm / count_T_bend_eq5_norm
        if count_T_bend_eq5_norm > 0
        else np.nan
    )
    avg_T_hb = sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan
    avg_T_twist_exc = (
        sum_T_twist_exc / count_T_twist_exc if count_T_twist_exc > 0 else np.nan
    )
    avg_T_twist_norm = (
        sum_T_twist_norm / count_T_twist_norm if count_T_twist_norm > 0 else np.nan
    )
    avg_T_wag_exc = sum_T_wag_exc / count_T_wag_exc if count_T_wag_exc > 0 else np.nan
    avg_T_wag_norm = (
        sum_T_wag_norm / count_T_wag_norm if count_T_wag_norm > 0 else np.nan
    )
    avg_T_rock_exc = (
        sum_T_rock_exc / count_T_rock_exc if count_T_rock_exc > 0 else np.nan
    )
    avg_T_rock_norm = (
        sum_T_rock_norm / count_T_rock_norm if count_T_rock_norm > 0 else np.nan
    )
    avg_T_bend_I_exc = (
        sum_T_bend_I_exc / count_T_bend_I_exc if count_T_bend_I_exc > 0 else np.nan
    )
    avg_T_bend_I_norm = (
        sum_T_bend_I_norm / count_T_bend_I_norm if count_T_bend_I_norm > 0 else np.nan
    )

    return (
        avg_T_stretch_exc,
        avg_T_stretch_norm,
        avg_T_bend_exc,
        avg_T_bend_norm,
        avg_T_bend_eq5_exc,
        avg_T_bend_eq5_norm,
        avg_T_bend_I_exc,
        avg_T_bend_I_norm,
        avg_T_hb,
        avg_T_twist_exc,
        avg_T_twist_norm,
        avg_T_wag_exc,
        avg_T_wag_norm,
        avg_T_rock_exc,
        avg_T_rock_norm,
    )


# ------------------------------------------------------------------
#          PARALLEL PROCESSING WORKER FUNCTION
# ------------------------------------------------------------------


def _parallel_worker_function(
    args_tuple: Tuple[int, Optional[Tuple[Any, ...]], float, float, float, float],
) -> Tuple[int, np.ndarray]:
    """
    Worker function for parallel processing.
    Takes a tuple: (frame_idx, frame_data_np_tuple, kb, hb_cutoff, eps, trig_eps)
    Returns: (frame_idx, temperature_results_array)
    """
    (
        frame_idx,
        frame_data_np,  # This is the tuple from _prepare_frame_data_for_numba
        kb_constant,
        hb_cutoff,
        epsilon_val,
        trig_epsilon_val,  # Renamed to avoid conflict with module name
    ) = args_tuple

    if frame_data_np is None:
        # Return NaNs for all 13 temperature values if frame data is None
        return frame_idx, np.full(13, np.nan, dtype=np.float64)

    (
        positions_np,
        velocities_np,
        masses_np,
        _atom_numeric_types_np,  # Not directly used by JIT func, but by prepare
        _original_indices_np,  # Not directly used by JIT func
        molecule_atom_indices_list_py,  # Python list of np.arrays
        is_H_excited_flat_mask_np,
        oxygen_indices_for_hb_np_arg,  # Now passed directly
    ) = frame_data_np

    molecule_indices_typed_list = NumbaTypedList()
    if molecule_atom_indices_list_py:
        for arr in molecule_atom_indices_list_py:
            molecule_indices_typed_list.append(arr)

    try:
        temps_tuple = _process_single_frame_numba_jit(
            positions_np,
            velocities_np,
            masses_np,
            molecule_indices_typed_list,
            is_H_excited_flat_mask_np,
            oxygen_indices_for_hb_np_arg,
            kb_constant,
            hb_cutoff,
            epsilon_val,
            trig_epsilon_val,
        )
        return frame_idx, np.array(temps_tuple, dtype=np.float64)
    except Exception as e_jit:
        # Optionally log the error with frame_idx if needed, e.g., using a logging queue
        # For now, return NaNs to avoid crashing the whole pool
        print(
            f"Error in JIT worker for frame {frame_idx}: {e_jit}"
        )  # Avoid printing directly from workers in general
        return frame_idx, np.full(13, np.nan, dtype=np.float64)


# ------------------------------------------------------------------
#          PARALLELIZED ANALYSIS FUNCTION
# ------------------------------------------------------------------
import concurrent.futures


def analyze_temps_numba_parallel(
    trajs_data: List[Frame],
    exc_indexes_file_path: str,
    unwrapped_coords: bool = False,
    kb_constant: float = DEFAULT_KB_CONSTANT,
    hb_cutoff: float = DEFAULT_HB_OO_DIST_CUTOFF,
    epsilon_val: float = DEFAULT_EPSILON,
    trig_epsilon: float = TRIG_EPSILON,  # Renamed from trig_epsilon_np for clarity
    num_workers: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Versione Numba-accelerated e parallelizzata di analyze_molecular_temperatures.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print(
            "tqdm not found. Install with 'pip install tqdm' for progress bar support."
        )
        tqdm = lambda x, **kwargs: x  # Fallback to no progress bar

    if not trajs_data:
        raise ValueError("Nessun frame fornito.")
    print(f"Analisi Numba Parallela: {len(trajs_data)} frames.")
    overall_start_time = time.time()

    exc_indexes_set: Set[int] = set()
    try:
        loaded_indices = np.loadtxt(exc_indexes_file_path, dtype=int)
        if loaded_indices.ndim == 0:
            exc_indexes_set = {int(loaded_indices.item())}
        else:
            exc_indexes_set = set(loaded_indices.flatten().astype(int))
        print(f"Loaded {len(exc_indexes_set)} excited atom indices.")
    except Exception as e:
        print(
            f"Warning: Could not load excited indices from '{exc_indexes_file_path}': {e}"
        )

    print("Pre-processing frames for Numba...")
    preprocessed_data_for_frames: List[Optional[Tuple[Any, ...]]] = []
    for frame_obj in trajs_data:
        data_tuple = _prepare_frame_data_for_numba(
            frame_obj, exc_indexes_set, unwrapped_coords
        )
        preprocessed_data_for_frames.append(data_tuple)
    print("Pre-processing completato.")

    num_frames = len(trajs_data)
    results_array = np.full(
        (num_frames, 15), np.nan, dtype=np.float64
    )  # 13 temperature modes

    tasks_args = []
    for i, frame_data_np_tuple in enumerate(preprocessed_data_for_frames):
        tasks_args.append(
            (
                i,  # frame index
                frame_data_np_tuple,
                kb_constant,
                hb_cutoff,
                epsilon_val,
                trig_epsilon,  # Use the renamed variable
            )
        )

    actual_num_workers = num_workers if num_workers is not None else os.cpu_count()
    print(
        f"Processing {num_frames} frames in parallel with up to {actual_num_workers} workers..."
    )
    jit_processing_start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_results = list(
            tqdm(
                executor.map(_parallel_worker_function, tasks_args),
                total=len(tasks_args),
                desc="Processing frames in parallel",
                unit="frame",
            )
        )

    for frame_idx, result_temps_array in future_results:
        if (
            result_temps_array is not None
        ):  # Should always be an array (of NaNs if error)
            results_array[frame_idx, :] = result_temps_array
        # If result_temps_array is None (should not happen with current worker), NaNs are kept

    jit_processing_end_time = time.time()
    print(
        f"Elaborazione JIT parallela dei frame finita in {jit_processing_end_time - jit_processing_start_time:.2f} secondi."
    )

    # Unpack results
    frame_avg_T_stretch_exc_list = list(results_array[:, 0])
    frame_avg_T_stretch_norm_list = list(results_array[:, 1])
    frame_avg_T_bend_exc_list = list(results_array[:, 2])
    frame_avg_T_bend_norm_list = list(results_array[:, 3])
    frame_avg_T_bend_eq5_exc_list = list(results_array[:, 4])
    frame_avg_T_bend_eq5_norm_list = list(results_array[:, 5])
    frame_avg_T_bend_I_exc_list = list(results_array[:, 6])
    frame_avg_T_bend_I_norm_list = list(results_array[:, 7])
    frame_avg_T_hb_list = list(results_array[:, 8])
    frame_avg_T_twist_exc_list = list(results_array[:, 9])
    frame_avg_T_twist_norm_list = list(results_array[:, 10])
    frame_avg_T_wag_exc_list = list(results_array[:, 11])
    frame_avg_T_wag_norm_list = list(results_array[:, 12])
    frame_avg_T_rock_exc_list = list(results_array[:, 13])
    frame_avg_T_rock_norm_list = list(results_array[:, 14])

    returned_data = {
        "stretch_excited_H": frame_avg_T_stretch_exc_list,
        "stretch_normal_H": frame_avg_T_stretch_norm_list,
        "bend_HOH_exc": frame_avg_T_bend_exc_list,
        "bend_HOH_norm": frame_avg_T_bend_norm_list,
        "bend_HOH_eq5_exc": frame_avg_T_bend_eq5_exc_list,
        "bend_HOH_eq5_norm": frame_avg_T_bend_eq5_norm_list,
        "bend_HOH_I_exc": frame_avg_T_bend_I_exc_list,
        "bend_HOH_I_norm": frame_avg_T_bend_I_norm_list,
        "hb": frame_avg_T_hb_list,
        "libr_twist_exc": frame_avg_T_twist_exc_list,
        "libr_twist_norm": frame_avg_T_twist_norm_list,
        "libr_wag_exc": frame_avg_T_wag_exc_list,
        "libr_wag_norm": frame_avg_T_wag_norm_list,
        "libr_rock_exc": frame_avg_T_rock_exc_list,
        "libr_rock_norm": frame_avg_T_rock_norm_list,
    }
    print(
        f"Tempo totale di esecuzione parallela: {time.time() - overall_start_time:.2f} secondi."
    )
    return returned_data
