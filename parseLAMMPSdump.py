from __future__ import annotations

from typing import List, Tuple

import numpy as np

from MDtools.dataStructures import Atom, Frame, Simulation


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readLAMMPSdump(
    filename: str,
    units: str = "metal",
    atom_per_molecule: int = 3,
    keep_vels: bool = True,
) -> Tuple[List[Frame], Simulation | None]:
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
    Returns
    -------
    frames : List[Frame]
        List of Frame objects, each containing timestep and molecules.
    simulation : Simulation
        Simulation object with cell information from the last frame.
    """

    frames = []
    simulation = None  # Initialize as None in case there are no frames
    vels_factor = 1e-3 if units == "metal" else 1.0

    with open(f"{filename}", "r") as f:
        lines = f.readlines()

    line_index = 0
    frame_index = 0

    while line_index < len(lines):
        # skip first line (ITEM: TIMESTEP)
        line_index += 1

        # Extract timestep
        timestep = int(lines[line_index])

        # skip third line (ITEM: NUMBER OF ATOMS)
        line_index += 2

        # Extract number of atoms
        n_atoms = int(lines[line_index])

        # skip fourth line (ITEM: BOX BOUNDS)
        line_index += 2

        # Extract box bounds (xlo, xhi)
        xlo = float(lines[line_index].split()[0])
        xhi = float(lines[line_index].split()[1])

        line_index += 1

        yhi = float(lines[line_index].split()[0])
        ylo = float(lines[line_index].split()[1])

        line_index += 1

        zhi = float(lines[line_index].split()[0])
        zlo = float(lines[line_index].split()[1])

        line_index += 1

        properties_string = lines[line_index].strip()

        line_index += 1

        lattice_string = f"{xhi - xlo} 0.0 0.0 0.0 {yhi - ylo} 0.0 0.0 0.0 {zhi - zlo}"
        lattice_values = np.array([float(x) for x in lattice_string.split()])
        cell_vectors = lattice_values.reshape((3, 3))

        # Create Simulation object for the first frame only
        if frame_index == 0:
            simulation = Simulation(
                n_atoms=n_atoms,
                lattice_string=lattice_string,
                cell_vectors=cell_vectors,
                properties_string=properties_string,
            )

        current_molecule = []
        molecules = []
        if line_index >= len(lines):
            break

        for _ in range(n_atoms):
            line_split = lines[line_index].split()
            line_index += 1

            atom_index = int(line_split[0])
            atom_type = int(line_split[1])
            atom_string = "O" if atom_type == 1 else "H"
            atom_position = np.array([float(x) for x in line_split[2:5]])
            atom_unwrapped_position = np.array([float(x) for x in line_split[9:12]])
            atom_mass = float(line_split[5])
            atom_velocity = (
                np.array([float(x) * vels_factor for x in line_split[6:9]])
                if keep_vels
                else np.zeros(3)
            )  # A/fs

            atom = Atom(
                index=atom_index,
                atom_type=atom_type,
                atom_string=atom_string,
                mass=atom_mass,
                position=atom_position,
                unwrapped_position=atom_unwrapped_position,
                velocity=atom_velocity,
            )
            current_molecule.append(atom)

            if len(current_molecule) == 3:
                molecules.append(current_molecule)
                current_molecule = []

        # Create a frame and add to frames list
        frame = Frame(index=frame_index, timestep=timestep, molecules=molecules)
        frames.append(frame)
        frame_index += 1

    return frames, simulation


def writeXYZ(frames: list[Frame], simulation: Simulation, filename: str) -> None:
    """
    Scrive le coordinate unwrapped in un file XYZ.
    """
    with open(filename, "w") as f:
        for frame in frames:
            f.write(f"{simulation.n_atoms}\n")
            f.write(
                f'Time={frame.timestep} pbc="T T T" Lattice={simulation.lattice_string} Properties={simulation.properties_string}'
            )
            for molecule in frame.molecules:
                for atom in molecule:
                    if (
                        atom.unwrapped_position is not None
                    ):  # Scrive le coordinate unwrapped
                        f.write(
                            f"{atom.atom_string} {atom.position[0]} {atom.position[1]} {atom.position[2]} {atom.mass} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]} {atom.unwrapped_position[0]} {atom.unwrapped_position[1]} {atom.unwrapped_position[2]}\n"
                        )

                    else:
                        f.write(
                            f"{atom.atom_string} {atom.position[0]} {atom.position[1]} {atom.position[2]} {atom.mass} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}"
                        )

    print(f"File {filename} written successfully.")
