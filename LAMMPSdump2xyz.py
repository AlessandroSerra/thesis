#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from MDtools.dataStructures import Atom


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readLAMMPSdump(
    filename: str, atom_per_molecule: int, units: str
) -> Tuple[List[List[Atom]], NDArray[np.float64]]:
    with open(f"{filename}", "r") as f:
        lines = f.readlines()

    molecules = []
    atoms_processed = 0

    for i, line in enumerate(lines):
        if "NUMBER OF ATOMS" in line:
            n_atoms = int(lines[i + 1].split()[0])
            continue

        if "BOX BOUNDS" in line:
            box_bounds = np.array(
                [
                    list(map(float, lines[i + 1].split())),
                    list(map(float, lines[i + 2].split())),
                    list(map(float, lines[i + 3].split())),
                ]
            )
            continue

        if "ITEM: ATOMS" in line:
            current_molecule = []
            for j in range(n_atoms):
                atom_data = list(map(float, lines[i + j + 1].split()))
                atom_id = int(atom_data[0])
                atom_type = int(atom_data[1])
                position = np.array(atom_data[2:5])
                if units == "metal":
                    velocity = (
                        np.array(atom_data[5:8]) * 1e-3
                    )  # Convert from A/ps to A/fs
                else:
                    velocity = np.array(atom_data[5:8])  # already in A/fs
                atom = Atom(atom_id, atom_type, position, velocity)
                current_molecule.append(atom)
                atoms_processed += 1

                # When we reach 2 atoms, create a molecule and reset
                if len(current_molecule) == atom_per_molecule:
                    molecules.append(current_molecule)
                    current_molecule = []

    print(f"Processed {atoms_processed} atoms from the LAMMPS dump file.")
    print(
        f"\nRead {len(molecules)} molecules from {filename} LAMMPS dump file with {units} units."
    )

    return molecules, box_bounds


# -------------------------------------------------------
#       --- Function to write XYZ data file ---
# -------------------------------------------------------
def writeXYZ(
    filename: str,
    molecules: List[List[Atom]],
    box_bounds: NDArray[np.float64],
) -> None:
    x_lenght = np.abs(box_bounds[0][1] - box_bounds[0][0])
    y_lenght = np.abs(box_bounds[1][1] - box_bounds[1][0])
    z_lenght = np.abs(box_bounds[2][1] - box_bounds[2][0])
    total_atoms = sum(len(molecule) for molecule in molecules)

    lattice_string = f'Lattice="{x_lenght} 0 0 0 {y_lenght} 0 0 0 {z_lenght}"'
    properties_string = "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3"
    processed_atoms = 0

    with open(filename, "w") as f:
        f.write(f"{total_atoms}\n")
        f.write('pbc="T T T"' + " " + lattice_string + " " + properties_string + "\n")

        for molecule in molecules:
            for atom in molecule:
                atom_string = "O" if atom.atom_type == 1 else "H"
                mass = 15.999 if atom.atom_type == 1 else 1.008
                f.write(
                    f"{atom_string} {atom.position[0]} {atom.position[1]} {atom.position[2]} {mass} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}\n"
                )
                processed_atoms += 1

    print(f"\nWritten {processed_atoms} atoms in the XYZ file.")
    print(f"\nWrote {len(molecules)} molecules into XYZ data file: {filename}\n")


# -------------------------------------------------------
#                   --- MAIN ---
# -------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(
        prog="LAMMPSdump2xyz.py",
        description="Script to convert a LAMMPS dump file into a GPUMD xyz model file.",
    )
    parser.add_argument(
        "dumpfile",
        type=str,
        help="Input LAMMPS dump file (e.g., 'init.dump').",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default="model.xyz",
        help="Output GPUMD xyz model file [default: 'model.xyz'].",
    )
    parser.add_argument(
        "-u",
        "--units",
        type=str,
        choices=["real", "metal"],
        default="metal",
        help="Units of the LAMMPS dump file [default: metal]",
    )
    parser.add_argument(
        "-a",
        "--atom_per_molecule",
        type=int,
        default=3,
        help="Number of atoms per molecule [default: 3].",
    )

    args: Namespace = parser.parse_args()

    # Read the LAMMPS dump file
    molecules, box_bounds = readLAMMPSdump(
        filename=args.dumpfile,
        atom_per_molecule=args.atom_per_molecule,
        units=args.units,
    )
    writeXYZ(filename=args.outfile, molecules=molecules, box_bounds=box_bounds)


if __name__ == "__main__":
    main()
