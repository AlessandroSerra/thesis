#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.linalg import eigh_tridiagonal

from MDtools.dataStructures import Atom, Simulation

# --------------------------------------------------------------
#                   --- Constants and Parameters ---
# --------------------------------------------------------------

# NOTE: using 1amu = 1.6605390666e-27 kg
HBAR_amu_A2_fs = 0.00635078  # amu * A^2 / fs (hbar in amu*A^2/fs)
AMU_A2_fs2_to_eV = 103.642
MU_OH = (15.999 * 1.008) / (15.999 + 1.008)  # amu (reduced mass)
MASSES = {"O": 15.999, "H": 1.008}  # amu
K_B_eV = 8.617333262e-5  # eV/K (Boltzmann constant)
EV_TO_CM1 = 8065.54429  # eV to cm^-1 conversion factor
C_U = 0.01036427  # cm^-1 / fs (speed of light in cm/fs)

# --- Lippincott-Schroeder Potential Parameters from Staib & Hynes ---
r0 = 0.97  # Angstrom
n = 9.18  # Angstrom^-1
g = 1.45
n_star = g * n
r0_star = r0

# Define the O---O distance (from PDF 2) [cite: 182]
R_OO = 2.85  # Angstrom

# Dissociation Energy D (Required for correct energy scale)
D_OH = 4.82
D_star = D_OH / g  # eV

# Global variables to be set during execution
verbose_output = None
plot = False


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readGPUMDdump(
    filename: str, atom_per_molecule: int, keep_vels: bool
) -> Tuple[List[List[Atom]], Simulation]:
    # read GPUMD xyz dump file
    with open(f"{filename}", "r") as f:
        lines = f.readlines()

        n_atoms = int(lines[0])
        comment_line = lines[1]

        lattice_string_match = re.search(r'Lattice="([^"]*)"', comment_line)
        properties_string_match = re.search(r'Properties=([^"]*)', comment_line)

        if lattice_string_match is None or properties_string_match is None:
            raise ValueError(
                "Invalid file format: missing lattice or properties string."
            )

        lattice_string = lattice_string_match.group(1)
        properties_string = properties_string_match.group(1)
        lattice_values = np.array([float(x) for x in lattice_string.split()])
        cell_vectors = lattice_values.reshape((3, 3))
        current_molecule = []

        simulation = Simulation(
            n_atoms=n_atoms,
            lattice_string=lattice_string,
            cell_vectors=cell_vectors,
            properties_string=properties_string,
        )

        molecules = []

        for i, line in enumerate(lines[2:]):
            line_split = line.split()
            atom_string = line_split[0]
            atom_type = 1 if atom_string == "O" else 2
            atom_position = np.array([float(x) for x in line_split[1:4]])
            atom_mass = float(line_split[4])
            atom_velocity = (
                np.array([float(x) for x in line_split[5:8]])
                if keep_vels
                else np.zeros(3)
            )

            atom = Atom(
                index=(i + 1),
                atom_type=atom_type,
                atom_string=atom_string,
                mass=atom_mass,
                position=atom_position,
                velocity=atom_velocity,
            )
            current_molecule.append(atom)
            if len(current_molecule) == atom_per_molecule:
                molecules.append(current_molecule)
                current_molecule = []

        return molecules, simulation


# --------------------------------------------------------------
#          --- Lippincott-Schroeder Potential ---
# --------------------------------------------------------------


# --- OH term (Using eV for energy) ---
def V_LS_bond1(r: float) -> float:
    """Potential for bond I (OH stretch) in eV"""
    if r <= 1e-6:
        return np.inf  # Avoid division by zero or negative r
    delta_r = r - r0
    exponent = -n * delta_r**2 / (2 * r)
    return D_OH * (1 - np.exp(exponent))


# --- O--H term (Using eV for energy) ---
def V_LS_bond2(r: float, R: float) -> float:
    """Potential for bond II (H---O interaction) in eV"""
    r_ho = R - r
    if r_ho <= 1e-6:
        return np.inf  # H cannot be beyond the second O
    delta_r_star = r_ho - r0_star  # How much H---O is stretched/compressed
    exponent = -n_star * delta_r_star**2 / (2 * r_ho)
    # Using the approximation V2 = -D* exp(...)
    return -D_star * np.exp(exponent)


# --- Total Potential for H motion at fixed R ---
def V_LS_hydrogen_motion(r: float, R: float = R_OO) -> float:
    """Total potential for H motion at fixed R in eV"""
    v1 = V_LS_bond1(r)
    v2 = V_LS_bond2(r, R)
    # Set minimum to zero for convenience in solver
    # This potential needs to be evaluated on the grid later
    return v1 + v2


# ----------------------------------------------------------------
# --- Numerical Schrödinger Solver (Finite Difference Method) ---
# ----------------------------------------------------------------


def solve_schrodinger_1d(
    potential_func: Callable[[float], float],
    r_grid: NDArray[np.float64],
    num_states: int = 2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Numerically solves the 1D TISE using finite difference method.

    Args:
        potential_func: Function V(r) defining the potential in eV.
        r_grid: 1D array of r values (Angstrom) for discretization.
        num_states: Number of lowest energy states to return.

    Returns:
        eigenvalues: Array of energy levels (E_n) in eV.
        eigenvectors: Array where columns are the numerical wavefunctions psi_n(r).
        Normalized such that integral |psi_n|^2 dr = 1.
    """

    if verbose_output is not None:
        print("\nSolving Schrödinger equation numerically...")

    dr = r_grid[1] - r_grid[0]
    N = len(r_grid)

    # Potential energy on the grid
    V_grid = np.array([potential_func(r) for r in r_grid])  # eV
    V_min = np.min(V_grid)
    V_grid -= V_min  # Shift minimum to zero (doesn't affect wavefunctions)

    # Kinetic energy term T = -hbar^2 / (2*mu) * d^2/dr^2
    # Factor hbar^2 / (2*mu) in eV * A^2
    hbar_sq_over_2mu_native = HBAR_amu_A2_fs**2 / (2 * MU_OH)
    hbar_sq_over_2mu_eV_A2 = hbar_sq_over_2mu_native * AMU_A2_fs2_to_eV

    # Finite difference matrix for -d^2/dr^2 (using centered difference)
    # Off-diagonal elements (-1) and Diagonal elements (2)
    diag = np.ones(N) * 2.0
    offdiag = np.ones(N - 1) * -1.0

    # Construct Hamiltonian matrix H = T + V
    # T matrix elements = (hbar^2 / (2*mu*dr^2)) * [2, -1, ...]
    # V matrix elements = V_grid on the diagonal
    H_diag = (hbar_sq_over_2mu_eV_A2 / dr**2) * diag + V_grid
    H_offdiag = (hbar_sq_over_2mu_eV_A2 / dr**2) * offdiag

    # Solve the eigenvalue problem H * psi = E * psi
    # Use eigh_tridiagonal for efficiency and numerical stability
    eigenvalues_raw, eigenvectors_raw = eigh_tridiagonal(
        H_diag, H_offdiag, select="i", select_range=(0, num_states - 1)
    )

    # Add back the potential minimum shift and ensure correct normalization
    eigenvalues = eigenvalues_raw + V_min  # Energies in eV
    eigenvectors = np.zeros_like(eigenvectors_raw)

    # Normalize eigenvectors: integral |psi|^2 dr = 1
    for i in range(num_states):
        psi = eigenvectors_raw[:, i]
        norm_sq = simpson(y=np.abs(psi) ** 2, x=r_grid)
        if norm_sq > 1e-9:
            eigenvectors[:, i] = psi / np.sqrt(norm_sq)
        else:
            eigenvectors[:, i] = psi  # Avoid division by zero if norm is tiny

    if verbose_output is not None:
        print("Schrödinger equation solved.\n")
    return eigenvalues, eigenvectors


# ------------------------------------------------------------
# --- Wigner Function Calculation (Numerical Integration) ---
# -----------------------------------------------------------


def calculate_wigner_function(
    psi_n: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    num_y_points: int = 201,
) -> NDArray[np.float64]:
    """
    Calculates the Wigner function W_n(r, p) on a grid.

    Args:
        psi_n: 1D array of the numerical wavefunction psi_n(r).
        r_grid: 1D array of r values (Angstrom) where psi_n is defined.
        p_grid: 1D array of p values (amu * A / fs) for the output grid.
        num_y_points: Number of points for numerical integration over y.

    Returns:
        wigner_grid: 2D array W_n(r, p) evaluated at r_grid[i], p_grid[j].
                     Units: 1 / (eV * fs) if p is in amu*A/fs and r in A.
    """
    Nr = len(r_grid)
    Np = len(p_grid)
    wigner_grid = np.zeros((Nr, Np))

    # Interpolate wavefunction for evaluation at r+y and r-y
    # Use linear interpolation, ensure bounds_error=False, fill_value=0
    psi_interp = interp1d(
        r_grid, psi_n, kind="linear", bounds_error=False, fill_value=0.0
    )

    # Determine integration range for y
    # Needs to cover the extent where psi_n is non-zero
    r_min_psi, r_max_psi = r_grid[0], r_grid[-1]
    # Heuristic: integrate y over the range of r_grid should be sufficient
    y_max = (r_max_psi - r_min_psi) / 2.0
    y_grid = np.linspace(-y_max, y_max, num_y_points)

    # Loop over r and p, calculate integral numerically
    for i, r_val in enumerate(r_grid):
        # Precompute psi values needed for this r_val
        psi_r_plus_y = psi_interp(r_val + y_grid)
        psi_r_minus_y = psi_interp(r_val - y_grid)  # psi is real, so psi* = psi

        for j, p_val in enumerate(p_grid):
            # Integrand: conj(psi(r+y)) * psi(r-y) * exp(2*i*p*y/hbar)
            # Since psi is real: psi(r+y) * psi(r-y) * exp(2*i*p*y/hbar)
            exponent_term = (2.0j * p_val / HBAR_amu_A2_fs) * y_grid
            integrand = psi_r_plus_y * psi_r_minus_y * np.exp(exponent_term)

            # Integrate using Simpson's rule (real part, as Wigner is real)
            # Factor 1 / (pi * hbar)
            integral_val = simpson(y=integrand, x=y_grid)
            # Wigner function is real, take real part (imaginary part should be ~0 due to symmetry)
            wigner_grid[i, j] = integral_val.real / (np.pi * HBAR_amu_A2_fs)

    return wigner_grid


# ----------------------------------------------------
#   --- Wigner Sampling (Acceptance-Rejection) ---
# ----------------------------------------------------


def sample_from_wigner(
    wigner_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    num_samples: int = 1,
) -> list[tuple[float, float]]:
    """
    Samples (r, p) pairs from the Wigner distribution using acceptance-rejection.
    Handles positive and negative values in Wigner function.

    Args:
        wigner_grid: 2D array W_n(r, p).
        r_grid: 1D r coordinates (Angstrom) for the grid.
        p_grid: 1D p coordinates (amu * A / fs) for the grid.
        num_samples: Number of samples to generate.

    Returns:
        samples: List of (r, p) tuples.
    """
    if verbose_output == 2:
        print(f"Sampling {num_samples} points from Wigner distribution...")
    samples = []
    Nr = len(r_grid)
    Np = len(p_grid)

    # Use absolute value for proposal distribution envelope
    wigner_abs = np.abs(wigner_grid)
    w_max = np.max(wigner_abs)

    if w_max < 1e-15:
        print("Warning: Wigner function seems to be zero everywhere. Cannot sample.")
        # Return samples at the center of the grid or raise error
        r_center = r_grid[Nr // 2]
        p_center = p_grid[Np // 2]
        return [(r_center, p_center)] * num_samples

    # Create interpolator for efficient lookup of W(r,p)
    # Use RectBivariateSpline for 2D interpolation
    wigner_interp = RectBivariateSpline(r_grid, p_grid, wigner_grid)

    count = 0
    max_attempts = num_samples * 1000  # Prevent infinite loop if sampling is hard
    attempts = 0

    while count < num_samples and attempts < max_attempts:
        attempts += 1
        # Sample uniformly within the grid range
        r_try = np.random.uniform(r_grid[0], r_grid[-1])
        p_try = np.random.uniform(p_grid[0], p_grid[-1])

        # Get Wigner value at the sampled point using interpolation
        w_val = wigner_interp(r_try, p_try, grid=False)  # Get value at single point

        # Acceptance-Rejection step
        # Compare |W(r,p)| with a random number scaled by max(|W|)
        acceptance_prob = np.abs(w_val) / w_max
        if np.random.rand() < acceptance_prob:
            # Accept the sample (r_try, p_try)
            samples.append((r_try, p_try))
            count += 1

    if attempts >= max_attempts:
        print(
            f"Warning: Reached max attempts ({max_attempts}) "
            f"but only generated {count}/{num_samples} samples."
        )

    if verbose_output == 2:
        print(f"Sampling complete. Generated {len(samples)} samples.")

    return samples


# -------------------------------------------------------
# --- Coordinate Remapping (Unchanged from previous) ---
# -------------------------------------------------------


def remap_coords_vels(
    pos_O: NDArray[np.float64],
    pos_H: NDArray[np.float64],
    vel_O: NDArray[np.float64],
    vel_H: NDArray[np.float64],
    r_new: float,
    p_new: float,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Remaps O, H positions and velocities for a target r_new, p_new."""

    # Current state
    vec_OH = pos_H - pos_O  # A
    r_old = np.linalg.norm(vec_OH)  # A
    if r_old < 1e-9:  # Avoid division by zero if atoms overlap
        raise ZeroDivisionError("Warning: O and H atoms are too close for sampling!\n")
    else:
        unit_vec_OH = vec_OH / r_old

    # Calculate changes needed
    delta_r = r_new - r_old  # A

    # Update positions (preserve center of mass)
    pos_O_new = pos_O - (MU_OH / MASSES["O"]) * delta_r * unit_vec_OH  # A
    pos_H_new = pos_H + (MU_OH / MASSES["H"]) * delta_r * unit_vec_OH  # A

    # Update velocities (preserve center of mass velocity)
    # p_new is relative momentum along the bond in amu * A / fs
    # Velocity change dv = p_rel / m
    vel_O_new = vel_O - (p_new / MASSES["O"]) * unit_vec_OH  # A / fs
    vel_H_new = vel_H + (p_new / MASSES["H"]) * unit_vec_OH  # A / fs

    return pos_O_new, pos_H_new, vel_O_new, vel_H_new


# -------------------------------------------------------
#       --- Molecules Excitation Function ---
# -------------------------------------------------------
def excite_molecules(
    molecules: List[List[Atom]], excite_perc: float, excite_lvl: int
) -> List[List[Atom]]:
    num_molecules = len(molecules)
    num_to_excite = int(num_molecules * excite_perc)
    exc_mol_idxs = np.random.choice(num_molecules, num_to_excite, replace=False)
    molecules_to_excite = [molecules[idx] for idx in exc_mol_idxs]

    r_min = 0.6
    r_max = 2.5  # Adjust based on potential shape
    Nr = 201  # Use odd number for Simpson's rule
    r_grid = np.linspace(r_min, r_max, Nr)

    energies, wavefunctions = solve_schrodinger_1d(
        V_LS_hydrogen_motion, r_grid, num_states=2
    )

    delta_E_eV = np.abs(energies[1] - energies[0])  # eV
    delta_E_native = delta_E_eV / AMU_A2_fs2_to_eV  # Convert back to amu*A^2/fs^2
    p_max = np.sqrt(2 * MU_OH * delta_E_native)  # amu * A / fs
    Np = 201
    p_grid = np.linspace(-p_max * 3, p_max * 3, Np)

    psi_0 = wavefunctions[:, 0]
    psi_1 = wavefunctions[:, 1]

    if verbose_output is not None:
        print(
            f"Energy levels: E0={energies[0]:.3f} eV ({(energies[0] * EV_TO_CM1):.2f} cm-1), E1={energies[1]:.3f} eV ({(energies[1] * EV_TO_CM1):.2f} cm-1)"
        )
        print(
            f"Energy gap E1-E0 = {(delta_E_eV):.3f} eV ({(delta_E_eV * EV_TO_CM1):.2f} cm-1)"
        )
        print(f"Corresponding reduced momentum: {p_max:.4f} amu*A/fs")
        print(f"\nCalculating Wigner function for n={excite_lvl} level...")

    if excite_lvl == 0:
        wigner_target = calculate_wigner_function(psi_0, r_grid, p_grid)
    elif excite_lvl == 1:
        wigner_target = calculate_wigner_function(psi_1, r_grid, p_grid)
    else:
        raise NotImplementedError("Target states above 1 are not implemented yet!")

    if verbose_output is not None:
        print("Wigner function calculated.")

    modified_indices = set()  # Keep track of modified atoms
    all_samples = []

    for exc_mol_idx, exc_molecule in enumerate(molecules_to_excite):
        Oatom = exc_molecule[0]
        Hatom = np.random.choice(exc_molecule[1:3])  # Randomly select one H atom
        Oatom_idx = Oatom.index
        Hatom_idx = Hatom.index

        # Check if atoms were already modified (e.g., if exciting both bonds)
        if (
            Oatom_idx in modified_indices or Hatom_idx in modified_indices
        ) and verbose_output is not None:
            print(
                f"Skipping molecule {exc_mol_idxs[exc_mol_idx]}, bond already modified."
            )
            continue
        if verbose_output == 2:
            print(
                f"\nExciting bond O({Oatom_idx})-H({Hatom_idx}) in molecule {exc_mol_idxs[exc_mol_idx]}..."
            )

        # Get current positions and velocities
        pos_O = Oatom.position
        pos_H = Hatom.position
        vel_O = Oatom.velocity
        vel_H = Hatom.velocity

        # Perform Wigner sampling for this bond
        samples_rp = sample_from_wigner(wigner_target, r_grid, p_grid, num_samples=1)
        if not samples_rp:
            print(
                f"Failed to sample for molecule {exc_mol_idxs[exc_mol_idx]}. Skipping."
            )
            continue

        all_samples.extend(samples_rp)
        r_sampled, p_sampled = samples_rp[0]
        modified_indices.add(Oatom_idx)
        modified_indices.add(Hatom_idx)

        pos_O_new, pos_H_new, vel_O_new, vel_H_new = remap_coords_vels(
            pos_O, pos_H, vel_O, vel_H, r_sampled, p_sampled
        )

        # Update positions and velocities in the molecule
        Oatom.position = pos_O_new
        Hatom.position = pos_H_new
        Oatom.velocity = vel_O_new
        Hatom.velocity = vel_H_new

        delta_R = np.linalg.norm((pos_H_new - pos_O_new) - (pos_H - pos_O))
        delta_V = np.linalg.norm((vel_H_new - vel_O_new) - (vel_H - vel_O))
        new_V_mod = np.linalg.norm(vel_H_new - vel_O_new)

        if verbose_output == 2:
            print(f"Sampled displacement: {r_sampled:.3f} A")
            print(f"Delta R: {delta_R:.4f} A")
            print(f"Sampled momentum: {p_sampled:.3f} amu*A/fs")
            print(f"New velocity modulus: {new_V_mod:.4f} A/fs")
            print(f"New velocity for H atom: {np.linalg.norm(vel_H_new):.3f} A/fs")
            print(f"New velocity for O atom: {np.linalg.norm(vel_O_new):.3f} A/fs")
            print(f"Delta V: {delta_V:.4f} A/fs")

    if plot:
        plot_wigner(
            r_grid,
            energies,
            wavefunctions,
            wigner_target,
            p_grid,
            all_samples,
            excite_lvl,
        )

    return molecules


# -------------------------------------------------------
#       --- Function to write XYZ data file ---
# -------------------------------------------------------
def writeXYZ(
    filename: str,
    molecules: List[List[Atom]],
    simulation: Simulation,
) -> None:
    lattice_string = f'Lattice="{simulation.lattice_string}"'
    properties_string = f"Properties={simulation.properties_string}"

    with open(filename, "w") as f:
        f.write(f"{simulation.n_atoms}\n")
        f.write('pbc="T T T"' + " " + lattice_string + " " + properties_string)

        for molecule in molecules:
            for atom in molecule:
                # charge = 0.5564 if atom.atom_type == 2 else -1.1128
                f.write(
                    f"{atom.atom_string} {atom.position[0]} {atom.position[1]} {atom.position[2]} {atom.mass} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}\n"
                )

    if verbose_output is not None:
        print(f"\nWrote XYZ data file: {filename}")


def writeLAMMPSdata(
    filename: str,
    molecules: List[List[Atom]],
    box_bounds: NDArray[np.float64],
    atom_style: str,
    units: str,
) -> None:
    Natoms = len(molecules) * 3
    Nbonds = len(molecules) * 2
    Nangles = len(molecules)

    with open(filename, "w") as f:
        f.write("LAMMPS data file for Wigner excitation\n\n")
        f.write(f"{Natoms} atoms\n")
        f.write("2 atom types\n")

        if atom_style == "full":
            f.write(f"{Nbonds} bonds\n")
            f.write("1 bond types\n")
            f.write(f"{Nangles} angles\n")
            f.write("1 angle types\n")

        f.write(f"\n-0.0 {box_bounds[0][0]} xlo xhi\n")
        f.write(f"-0.0 {box_bounds[1][1]} ylo yhi\n")
        f.write(f"-0.0 {box_bounds[2][2]} zlo zhi\n\n")

        f.write(f"Atoms # {atom_style}\n\n")

        if atom_style == "full":
            for mol_idx, molecule in enumerate(molecules):
                for atom in molecule:
                    if units == "metal":
                        atom.velocity *= 1e3  # Convert back from A/fs to A/ps
                    charge = 0.5564 if atom.atom_type == 2 else -1.1128
                    f.write(
                        f"{atom.index} {mol_idx + 1} {atom.atom_type} {charge} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
                    )

        else:
            for mol_idx, molecule in enumerate(molecules):
                for atom in molecule:
                    if units == "metal":
                        atom.velocity *= 1e3  # Convert back from A/fs to A/ps
                    f.write(
                        f"{atom.index} {atom.atom_type} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
                    )

        f.write("\nVelocities\n\n")
        for mol_idx, molecule in enumerate(molecules):
            for atom in molecule:
                f.write(
                    f"{atom.index} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}\n"
                )

        if atom_style == "full":
            f.write("\nBonds\n\n")
            bond_idx = 1
            for molecule in molecules:
                Oatom = molecule[0]
                Hatom1 = molecule[1]
                Hatom2 = molecule[2]
                f.write(f"{bond_idx} 1 {Oatom.index} {Hatom1.index}\n")
                bond_idx += 1
                f.write(f"{bond_idx} 1 {Oatom.index} {Hatom2.index}\n")
                bond_idx += 1

            f.write("\nAngles\n\n")
            for angle_idx, molecule in enumerate(molecules):
                Oatom = molecule[0]
                Hatom1 = molecule[1]
                Hatom2 = molecule[2]
                f.write(
                    f"{angle_idx + 1} 1 {Hatom1.index} {Oatom.index} {Hatom2.index}\n"
                )

    if verbose_output is not None:
        print(f"\nWrote LAMMPS data file: {filename}")


def plot_wigner(
    r_grid: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    wigner_grid: Optional[NDArray[np.float64]] = None,
    p_grid: Optional[NDArray[np.float64]] = None,
    samples: Optional[List[Tuple[float, float]]] = None,
    state_level: int = 1,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Optional import of scienceplots - we'll handle if it's not available
        try:
            import scienceplots

            plt.style.use(["science", "notebook"])
        except ImportError:
            pass

        # Calculate both ground and excited state Wigner functions
        if wigner_grid is not None and p_grid is not None:
            # Create a figure with a side-by-side layout
            fig = plt.figure(figsize=(16, 8))  # Wide figure for side-by-side layout

            # Create a grid with 2 rows, 2 columns
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

            ax1 = fig.add_subplot(
                gs[:, 0]
            )  # Left side: potential/wavefunctions (spans both rows)
            ax2 = fig.add_subplot(
                gs[0, 1]
            )  # Top-right: Ground state (n=0) Wigner function
            ax3 = fig.add_subplot(
                gs[1, 1]
            )  # Bottom-right: Excited state (n=1) Wigner function

            # Calculate ground state Wigner function
            psi_0 = eigenvectors[:, 0]
            wigner_0 = calculate_wigner_function(psi_0, r_grid, p_grid)

            # Calculate excited state Wigner function
            psi_1 = eigenvectors[:, 1]
            wigner_1 = calculate_wigner_function(psi_1, r_grid, p_grid)
        else:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            ax2 = None
            ax3 = None
            wigner_0 = None
            wigner_1 = None

        # Plot potential and wavefunctions
        V_grid = np.array([V_LS_hydrogen_motion(r) for r in r_grid])
        energies = eigenvalues
        scale = 1  # Scale factor for wavefunction visualization

        for i in range(len(eigenvalues)):
            ax1.plot(
                r_grid,
                energies[i] * EV_TO_CM1 + scale * eigenvectors[:, i] * EV_TO_CM1,
                label=f"psi_{i} (E={energies[i]:.3f} eV)",
                color=f"C{i}",
            )
            ax1.axhline(
                energies[i] * EV_TO_CM1 + scale * eigenvectors[0, i] * EV_TO_CM1,
                color=f"C{i}",
                linestyle="--",
                lw=1,
            )
            ax1.text(
                r_grid[-1] * 0.98,  # x-position (near the right edge)
                energies[i] * EV_TO_CM1 + 0.02,  # y-position (slightly above the line)
                f"E_{i} = {energies[i] * EV_TO_CM1:.0f} cm-1",
                color=f"C{i}",
                ha="right",  # Horizontal alignment: right-aligned
                va="bottom",  # Vertical alignment: below the anchor point
            )

        ax1.plot(r_grid, V_grid * EV_TO_CM1, color="black", label="V(r)", lw=2)
        ax1.set_xlabel("r [Å]")
        ax1.set_ylabel("Energy [cm-1]")
        ax1.set_xlim(r_grid[0], r_grid[-1])
        ax1.set_title("Lippincott-Schroeder Potential and Wavefunctions")
        ax1.grid()
        ax1.legend()

        # Plot Wigner distributions if we have the data
        if (
            wigner_grid is not None
            and p_grid is not None
            and ax2 is not None
            and ax3 is not None
        ):
            # Plot ground state (n=0) Wigner function
            im0 = ax2.contourf(r_grid, p_grid, wigner_0.T, levels=50, cmap="seismic")
            cbar0 = fig.colorbar(im0, ax=ax2, fraction=0.046, pad=0.04)
            cbar0.set_label("Wigner Function (n=0)")
            ax2.set_xlabel("r [Å]")
            ax2.set_ylabel("p [amu·Å/fs]")
            ax2.set_title("Ground State (n=0) Wigner Distribution")

            # Plot excited state (n=1) Wigner function
            im1 = ax3.contourf(r_grid, p_grid, wigner_1.T, levels=50, cmap="seismic")
            cbar1 = fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
            cbar1.set_label("Wigner Function (n=1)")
            ax3.set_xlabel("r ([Å]")
            ax3.set_ylabel("p [amu·Å/fs]")
            ax3.set_title("Excited State (n=1) Wigner Distribution")

            # Plot sampled points if provided
            if samples is not None and len(samples) > 0:
                r_samples, p_samples = zip(*samples)
                # Add samples to the appropriate plot based on state_level
                if state_level == 0:
                    ax2.scatter(
                        r_samples,
                        p_samples,
                        color="black",
                        marker="o",
                        s=10,
                        alpha=0.7,
                        label=f"Samples ({len(samples)} points)",
                    )
                    ax2.legend()
                else:  # state_level == 1 or higher
                    ax3.scatter(
                        r_samples,
                        p_samples,
                        color="black",
                        marker="o",
                        s=10,
                        alpha=0.7,
                        label=f"Samples ({len(samples)} points)",
                    )
                    ax3.text(
                        r_grid[-1] * 0.95,
                        p_grid[-1] * 0.8,
                        f"{len(samples)} samples",
                        color="black",
                        ha="right",
                        va="bottom",
                        fontdict={"size": 12, "weight": "bold"},
                    )

        plt.tight_layout()
        plt.show()
    except ImportError as e:
        print(f"Could not generate plot: {e}")
        print("Install matplotlib and scienceplots for visualization.")


def run_excitation(
    dumpfile: str,
    atom_per_molecule: int = 3,
    excite_perc: float = 0.1,
    excite_lvl: int = 1,
    datafile: str = "excited.xyz",
    plot_enabled: bool = False,
    verbosity: Optional[int] = None,
    keep_vels: bool = False,
) -> Tuple[List[List[Atom]], Simulation]:
    """
    Main function to run the Wigner sampling and excitation.

    Args:
        dumpfile: Path to the GPUMD dump file
        atom_per_molecule: Number of atoms per molecule
        excite_perc: Fraction of molecules to excite (0.0-1.0)
        excite_lvl: Excitation level (0 = ground state, 1 = first excited)
        datafile: Output data filename
        plot_enabled: Whether to generate plots
        verbosity: Verbosity level (None, 1, 2)
        keep_vels: Whether to keep original velocities
    """
    global plot, verbose_output

    plot = plot_enabled
    verbose_output = verbosity

    # Read molecules
    molecules, simulation = readGPUMDdump(dumpfile, atom_per_molecule, keep_vels)

    # Perform excitation
    exc_molecules = excite_molecules(molecules, excite_perc, excite_lvl)

    if verbose_output is not None:
        print(
            f"\nSuccessfully excited {int(len(molecules) * excite_perc)} out of {len(molecules)} molecules."
        )

    # Write output
    writeXYZ(datafile, exc_molecules, simulation)

    return exc_molecules, simulation


def main():
    # --- Argument parsing ---
    parser = ArgumentParser(
        prog="wignerGPUMD.py",
        description="Script to excite water molecules in a LAMMPS dump file using Wigner Sampling.",
    )
    parser.add_argument(
        "dumpfile",
        type=str,
        help="Input LAMMPS dump file (e.g., 'init.dump').",
    )
    parser.add_argument(
        "-a",
        "--atom_per_molecule",
        type=int,
        default=3,
        help="Number of atoms per molecule [default: 3].",
    )
    parser.add_argument(
        "-e",
        "--excite_perc",
        type=float,
        default=0.1,
        help="Fraction (0-1) of molecules to excite [default: 0.1 = 10 percent].",
    )
    parser.add_argument(
        "-l",
        "--excite_lvl",
        type=int,
        choices=[0, 1],
        default=1,
        help="Excitation level: 0 (ground) or 1 (first excited state) [default=1].",
    )
    parser.add_argument(
        "-d",
        "--datafile",
        type=str,
        default="excited.xyz",
        help="Output XYZ data file name [default: 'excited.xyz'].",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the Wigner function and wavefunctions.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Enable verbose output. Use -vv for more verbosity.",
    )
    parser.add_argument(
        "--keep_vels",
        action="store_true",
        help="Keep original Maxwell-Boltzmann velocities of the molecules.",
    )

    args = parser.parse_args()

    # Set global variables
    global plot, verbose_output
    plot = args.plot
    verbose_output = args.verbose

    # Run the main workflow
    _ = run_excitation(
        args.dumpfile,
        args.atom_per_molecule,
        args.excite_perc,
        args.excite_lvl,
        args.datafile,
        args.plot,
        args.verbose,
        args.keep_vels,
    )


if __name__ == "__main__":
    main()
