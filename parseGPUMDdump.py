import re
from typing import List, Tuple

import numpy as np

from MDtools.dataStructures import Atom, Frame, Simulation


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readGPUMDdump(
    filename: str, atom_per_molecule: int = 3, keep_vels: bool = True
) -> Tuple[List[Frame], Simulation | None]:
    frames = []
    simulation = None  # Initialize as None in case there are no frames

    with open(f"{filename}", "r") as f:
        lines = f.readlines()

        line_index = 0
        frame_index = 0

        while line_index < len(lines):
            # Read frame header
            n_atoms = int(lines[line_index])
            line_index += 1
            comment_line = lines[line_index]
            line_index += 1

            # Parse lattice and properties from comment line
            lattice_string_match = re.search(r'Lattice="([^"]*)"', comment_line)
            properties_string_match = re.search(r'Properties=([^"]*)', comment_line)
            timestep_match = re.search(r"Time=([0-9.]+)", comment_line)

            if lattice_string_match is None or properties_string_match is None:
                raise ValueError(
                    "Invalid file format: missing lattice or properties string."
                )

            lattice_string = lattice_string_match.group(1)
            properties_string = properties_string_match.group(1)
            lattice_values = np.array([float(x) for x in lattice_string.split()])
            cell_vectors = lattice_values.reshape((3, 3))

            # Extract timestep if available, default to frame_index if not
            timestep = (
                float(timestep_match.group(1)) if timestep_match else float(frame_index)
            )

            # Create Simulation object for the first frame only
            if frame_index == 0:
                simulation = Simulation(
                    n_atoms=n_atoms,
                    lattice_string=lattice_string,
                    cell_vectors=cell_vectors,
                    properties_string=properties_string,
                )

            # Process atoms
            current_molecule = []
            molecules = []

            for i in range(n_atoms):
                if line_index >= len(lines):
                    break

                line_split = lines[line_index].split()
                line_index += 1

                atom_string = line_split[0]
                atom_type = 1 if atom_string == "O" else 2
                atom_position = np.array([float(x) for x in line_split[1:4]])
                atom_unwrapped_position = np.array([float(x) for x in line_split[8:11]])
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
                    unwrapped_position=atom_unwrapped_position,
                    velocity=atom_velocity,
                )
                current_molecule.append(atom)

                if len(current_molecule) == atom_per_molecule:
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
