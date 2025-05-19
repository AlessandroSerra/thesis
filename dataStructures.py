from dataclasses import dataclass
from typing import Generator, List, TypeVar

import numpy as np
from numpy.typing import NDArray


# --- Atom Object ---
@dataclass
class Atom:
    index: int
    atom_type: int
    atom_string: str | None
    mass: float | None
    position: np.ndarray
    unwrapped_position: np.ndarray | None
    velocity: np.ndarray

    def __str__(self) -> str:
        return f"Atom {self.index} of type {self.atom_string} at position {self.position} and velocity {self.velocity}"

    def __repr__(self) -> str:
        return f"Atom(index={self.index}, atom_type={self.atom_type}, atom_string={self.atom_string}, mass={self.mass}, position={self.position}, unwrapped_position={self.unwrapped_position}, velocity={self.velocity})"


@dataclass
class Frame:
    index: int
    timestep: float
    molecules: List[List[Atom]]

    def __str__(self) -> str:
        return f"Frame {self.index} at timestep {self.timestep} with {len(self.molecules)} molecules"

    def __repr__(self) -> str:
        return f"Frame(index={self.index}, timestep={self.timestep}, molecules={self.molecules})"

    def __len__(self) -> int:
        """Return the number of atoms in the frame."""
        return sum(len(molecule) for molecule in self.molecules)

    def __iter__(self):
        """Iterate over all atoms in the frame."""
        for molecule in self.molecules:
            for atom in molecule:
                yield atom

    def get_positions(self) -> Generator[np.ndarray, None, None]:
        """Yield positions of all atoms one by one (memory efficient)."""
        for atom in self:
            yield atom.position

    def get_all_positions(self) -> np.ndarray:
        """Return a numpy array with all atom positions."""
        return np.array(list(self.get_positions()))

    def get_unwrapped_positions(self) -> Generator[np.ndarray | None, None, None]:
        """Yield unwrapped positions of all atoms one by one (memory efficient)."""
        for atom in self:
            if atom.unwrapped_position:
                yield atom.unwrapped_position

    def get_all_unwrapped_positions(self) -> np.ndarray:
        """Return a numpy array with all unwrapped atom positions."""
        return np.array(list(self.get_unwrapped_positions()))

    def get_velocities(self) -> Generator[np.ndarray, None, None]:
        """Yield positions of all atoms one by one (memory efficient)."""
        for atom in self:
            yield atom.velocity

    def get_all_velocities(self) -> np.ndarray:
        """Return a numpy array with all atom positions."""
        return np.array(list(self.get_velocities()))


@dataclass
class Simulation:
    n_atoms: int
    lattice_string: str | None
    cell_vectors: NDArray[np.float64]
    properties_string: str | None

    def __str__(self) -> str:
        return f"Simulation with {self.n_atoms} atoms and lattice {self.lattice_string}"

    def __repr__(self) -> str:
        return f"Simulation(n_atoms={self.n_atoms}, lattice_string={self.lattice_string}, cell_vectors={self.cell_vectors}, properties_string={self.properties_string})"

    def __len__(self) -> int:
        """Return the number of atoms in the simulation."""
        return self.n_atoms

    def calculate_volume(self) -> float:
        """Calculate the volume of the simulation cell."""
        return np.linalg.det(self.cell_vectors)


AtomType = TypeVar("AtomType", bound="Atom")
SimulationType = TypeVar("SimulationType", bound="Simulation")
FrameType = TypeVar("FrameType", bound="Frame")
MoleculeType = List[AtomType]
