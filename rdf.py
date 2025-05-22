from typing import Tuple

import numpy as np
from freud.box import Box
from freud.density import RDF

from MDtools.dataStructures import Frame, Simulation


def calculateRDF(
    traj: Frame,
    sim: Simulation,
    r_max: float = 7.0,
    partial: str | None = None,
    n_bins: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola il Radial Distribution Function (RDF) da una lista di Frame (Freud backend).
    Args:
        traj (list[Frame]): Lista di Frame contenenti le posizioni degli atomi.
        sim (Simulation): Oggetto Simulation contenente le informazioni sulla simulazione.
        r_max (float): Distanza massima per il calcolo dell'RDF.
        partial (Optional[str]): Se specificato, calcola l'RDF parziale per "O" o "H".
        n_bins (Optional[int]): Numero di bin per il calcolo dell'RDF. Se None, viene calcolato automaticamente.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Due array numpy contenenti i valori dell'RDF e delle distanze.
    """

    _n_bins = n_bins if n_bins is not None else int(r_max * 10)

    box = Box.from_matrix(sim.cell_vectors)
    rdf = RDF(bins=_n_bins, r_max=r_max)

    if partial is None:
        positions = traj.get_all_positions()
        rdf.compute(system=(box, positions), reset=True)

    else:
        if partial == "OO":
            positionsO = traj.get_all_positions(type="O")
            rdf.compute(system=(box, positionsO), reset=True)
        elif partial == "HH":
            positionsH = traj.get_all_positions(type="H")
            rdf.compute(system=(box, positionsH), reset=True)
        elif partial == "OH":
            positionsH = traj.get_all_positions(type="H")
            positionsO = traj.get_all_positions(type="O")
            rdf.compute(system=(box, positionsO), query_points=positionsH, reset=True)

        else:
            raise ValueError("Partial must be 'OO', 'HH' or 'OH.")

    return rdf.rdf, rdf.bin_centers
