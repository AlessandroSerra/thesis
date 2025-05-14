import functools
import multiprocessing
import os
import time
from typing import Dict, List, Optional, Set, Tuple, TypeVar

import numpy as np

# Define missing type variables
AtomType = TypeVar("AtomType")
FrameType = TypeVar("FrameType")

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7  # in appropriate units
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5  # Angstroms
DEFAULT_EPSILON: float = 1e-40  # Small number to prevent division by zero


# --- Inner Helper Function (Not intended for direct external use) ---
def _normalize_vector(v: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalizes a vector, handling near-zero norms."""
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return np.zeros_like(v)
    return v / norm


# --- Inner Worker Function (for parallel processing, not for direct external use) ---
def _process_single_frame_worker(
    frame_data_tuple: Tuple[int, FrameType],
    exc_indexes_set_arg: Set[int],
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

    molecules_in_frame: List[List[AtomType]] = current_frame.molecules
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

        O_atom: Optional[AtomType] = None
        H1_atom: Optional[AtomType] = None
        H2_atom: Optional[AtomType] = None
        temp_h_atoms: List[AtomType] = []

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
                d_OH_vec = O_atom.position - h_atom_current.position
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
            d_O_H1_vec = H1_atom.position - O_atom.position
            d_O_H2_vec = H2_atom.position - O_atom.position
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

            d_H1H2_vec = H1_atom.position - H2_atom.position
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
            atoms_in_mol_lib: List[AtomType] = [O_atom, H1_atom, H2_atom]
            masses_lib = np.array([atom.mass for atom in atoms_in_mol_lib])
            positions_lib = np.array([atom.position for atom in atoms_in_mol_lib])
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

            vec_OH1_lib = H1_atom.position - O_atom.position
            vec_OH2_lib = H2_atom.position - O_atom.position

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
            O1_hb: Optional[AtomType] = None
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
                O2_hb: Optional[AtomType] = None
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
                    d_O1O2_vec = O1_hb.position - O2_hb.position
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
def analyze_molecular_temperatures(
    trajs_data: List[FrameType],
    run_path: str,
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
        run_path: Path to the directory containing 'nnp-indexes.dat' for excited atoms.
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
    indexed_trajs_list: List[Tuple[int, FrameType]] = list(enumerate(trajs_data))

    # Load excited atom indices
    exc_indexes_set_loaded: Set[int] = set()
    nnp_indexes_file_path = os.path.join(run_path, "nnp-indexes.dat")
    try:
        exc_indexes_array = np.loadtxt(nnp_indexes_file_path, dtype=int)
        exc_indexes_set_loaded = set(
            exc_indexes_array.tolist()
            if exc_indexes_array.ndim > 0
            else [int(exc_indexes_array)]
        )
        print(
            f"Loaded {len(exc_indexes_set_loaded)} excited atom indices from '{nnp_indexes_file_path}'."
        )
    except FileNotFoundError:
        print(
            f"Warning: File not found - '{nnp_indexes_file_path}'. No excited H-atom differentiation for stretch."
        )
    except Exception as e:
        print(
            f"Error loading '{nnp_indexes_file_path}': {e}. No excited H-atom differentiation."
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
        "overall_avg_stretch_excited_H": overall_avg_T_stretch_exc,
        "overall_avg_stretch_normal_H": overall_avg_T_stretch_norm,
        "overall_avg_bend_HOH": overall_avg_T_bend,
        "overall_avg_hydrogen_bond_OO": overall_avg_T_hb,
        "overall_avg_libration_twist": overall_avg_T_twist,
        "overall_avg_libration_wag": overall_avg_T_wag,
        "overall_avg_libration_rock": overall_avg_T_rock,
    }
    return returned_data
