from __future__ import annotations

import re

import numpy as np

from MDtools.dataStructures import Atom, Frame, Simulation


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


def readGPUMDdump(
    filename: str, every: int = 1, atom_per_molecule: int = 3, keep_vels: bool = True
) -> tuple[list[Frame], Simulation | None]:
    """
    read multiple frames from a gpumd dump trajectory file.

    parameters
    ----------
    filename : str
        path to the dump file (.xyz) containing multiple frames.
    every : int
        read only every nth frame (default: 1, meaning read all frames).
    atom_per_molecule : int
        number of atoms per molecule (3 for h2o, etc.).
    keep_vels : bool
        if true, reads velocities; otherwise sets them to zero.
    returns
    -------
    frames : list[frame]
        list of frame objects, each containing timestep and molecules.
    simulation : simulation
        simulation object with cell information from the last frame.
    """
    # Pre-compila le regex
    lattice_pattern = re.compile(r'Lattice="([^"]*)"')
    properties_pattern = re.compile(r'Properties=([^"]*)')
    timestep_pattern = re.compile(r"Time=([0-9.]+)")

    frames = []
    simulation = None  # initialize as none in case there are no frames

    with open(f"{filename}", "r") as f:
        lines = f.readlines()

    line_index = 0
    frame_index = 0
    actual_frame_index = 0

    while line_index < len(lines):
        # read frame header
        n_atoms = int(lines[line_index])
        line_index += 1
        comment_line = lines[line_index]
        line_index += 1

        # parse lattice and properties from comment line
        lattice_string_match = lattice_pattern.search(comment_line)
        properties_string_match = properties_pattern.search(comment_line)
        timestep_match = timestep_pattern.search(comment_line)

        if lattice_string_match is None or properties_string_match is None:
            raise ValueError(
                "invalid file format: missing lattice or properties string."
            )

        lattice_string = lattice_string_match.group(1)
        properties_string = properties_string_match.group(1)
        lattice_values = np.array([float(x) for x in lattice_string.split()])
        cell_vectors = lattice_values.reshape((3, 3))

        # extract timestep if available, default to frame_index if not
        timestep = (
            float(timestep_match.group(1)) if timestep_match else float(frame_index)
        )

        # create simulation object for the first frame only
        if frame_index == 0:
            simulation = Simulation(
                n_atoms=n_atoms,
                lattice_string=lattice_string,
                cell_vectors=cell_vectors,
                properties_string=properties_string,
            )

        # check if we should process this frame based on the 'every' parameter
        process_frame = frame_index % every == 0

        # process atoms
        if process_frame:
            current_molecule = []
            molecules = []

            for i in range(n_atoms):
                if line_index >= len(lines):
                    break

                line_split = lines[line_index].split()
                line_index += 1

                atom_string = line_split[0]
                atom_type = 1 if atom_string == "o" else 2
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

            # create a frame and add to frames list
            frame = Frame(
                index=actual_frame_index, timestep=timestep, molecules=molecules
            )
            frames.append(frame)
            actual_frame_index += 1
        else:
            # skip atoms if we're not processing this frame
            line_index += n_atoms

        frame_index += 1

    return frames, simulation
