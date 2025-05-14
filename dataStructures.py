from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
from numpy.typing import NDArray


# --- Atom Object ---
@dataclass
class Atom:
    index: int
    atom_type: int
    atom_string: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class Frame:
    index: int
    timestep: float
    molecules: List[List[Atom]]


@dataclass
class Simulation:
    n_atoms: int
    lattice_string: str | None
    cell_vectors: NDArray[np.float64]
    properties_string: str | None


AtomType = TypeVar("AtomType", bound="Atom")
SimulationType = TypeVar("SimulationType", bound="Simulation")
MoleculeType = List[AtomType]
