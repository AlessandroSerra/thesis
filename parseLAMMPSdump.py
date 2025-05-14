# parseLAMMPSdump.py
"""
Lettura di un file LAMMPS dump (un solo frame) e costruzione di:
    • molecules : list[list[Atom]]
    • simulation: Simulation
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from MDtools.dataStructures import Atom, Frame, Simulation


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readLAMMPSdump(
    filename: str,
    atom_per_molecule: int = 3,
    keep_vels: bool = True,
    has_mass: bool = False,
    units: str = "metal",
) -> Tuple[List[Frame], Simulation]:
    """
    Read multiple frames from a LAMMPS dump trajectory file.

    Parameters
    ----------
    filename : str
        Path to the dump file (.lammpstrj) containing multiple frames.
    atom_per_molecule : int
        Number of atoms per molecule (3 for H2O, etc.).
    keep_vels : bool
        If True, reads velocities; otherwise sets them to zero.
    units : str
        'metal' → converts velocities from Å/ps → Å/fs, otherwise leaves them.

    Returns
    -------
    frames : List[Frame]
        List of Frame objects, each containing timestep and molecules.
    simulation : Simulation
        Simulation object with cell information from the last frame.
    """
    frames = []
    simulation = None

    with open(filename, "r") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            # Start of a new frame
            timestep = int(lines[i + 1].strip())

            # Find n_atoms
            while i < len(lines) and "NUMBER OF ATOMS" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            n_atoms = int(lines[i + 1].strip())

            # Find box bounds
            while i < len(lines) and "BOX BOUNDS" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            box_bounds = np.array(
                [
                    list(map(float, lines[i + 1].split())),
                    list(map(float, lines[i + 2].split())),
                    list(map(float, lines[i + 3].split())),
                ],
                dtype=float,
            )

            # Find atoms section
            while i < len(lines) and "ITEM: ATOMS" not in lines[i]:
                i += 1
            if i >= len(lines):
                break

            # Parse atom data
            molecules: List[List[Atom]] = []
            current_molecule: List[Atom] = []

            for j in range(n_atoms):
                atom_data = list(map(float, lines[i + j + 1].split()))
                atom_id = int(atom_data[0])
                atom_type = int(atom_data[1])
                atom_string = "O" if atom_type == 1 else "H"
                position = np.array(atom_data[2:5], dtype=float)
                mass = float(atom_data[5]) if has_mass else None

                # --- velocities ---
                if keep_vels:
                    vel = (
                        np.array(atom_data[5:8], dtype=float)
                        if has_mass
                        else np.array(atom_data[4:7])
                    )
                    if units == "metal":  # Å/ps → Å/fs
                        vel *= 1e-3
                else:
                    vel = np.zeros(3, dtype=float)

                # --- Atom object ---
                atom = Atom(
                    index=atom_id,
                    atom_type=atom_type,
                    atom_string=atom_string,
                    mass=mass,
                    position=position,
                    velocity=vel,
                )
                current_molecule.append(atom)

                # Pack molecule
                if len(current_molecule) == atom_per_molecule:
                    molecules.append(current_molecule)
                    current_molecule = []

            # Update Simulation object (with data from the current frame)
            x_dim = np.abs(box_bounds[0][1] - box_bounds[0][0])
            y_dim = np.abs(box_bounds[1][1] - box_bounds[1][0])
            z_dim = np.abs(box_bounds[2][1] - box_bounds[2][0])
            cell_vectors = np.diag([x_dim, y_dim, z_dim])

            simulation = Simulation(
                n_atoms=n_atoms,
                lattice_string=None,
                cell_vectors=cell_vectors,
                properties_string=None,
            )

            # Create Frame object
            frame = Frame(index=len(frames), timestep=timestep, molecules=molecules)
            frames.append(frame)

            # Move to next line after atoms section
            i += n_atoms + 1
        else:
            i += 1

    if not frames:
        raise RuntimeError("No frames found in the LAMMPS dump file.")

    if simulation is None:
        raise RuntimeError("Could not create Simulation object from LAMMPS dump file.")

    return frames, simulation
