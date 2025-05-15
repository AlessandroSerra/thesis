from typing import List, Optional, Union

import numpy as np

from MDtools.dataStructures import Atom, Frame, Simulation


# --- Funzione Ausiliaria per trovare O e H ---
def _get_oxygen_and_hydrogens_in_molecule(
    molecule_atom_list: List[Atom],
) -> tuple[Optional[Atom], List[Atom]]:
    """
    Identifica l'atomo di Ossigeno e gli atomi di Idrogeno in una molecola.
    Assume che ci sia al massimo un Ossigeno.
    """
    oxygen_atom: Optional[Atom] = None
    hydrogen_atoms: List[Atom] = []
    for atom in molecule_atom_list:
        if atom.atom_string == "O":
            if oxygen_atom is None:
                oxygen_atom = atom
            else:
                # Potenziale problema: più di un ossigeno identificato in una molecola d'acqua.
                # Per ora, si usa il primo trovato.
                print(
                    f"Attenzione: Più di un atomo 'O' trovato nella molecola che inizia con atomo idx {molecule_atom_list[0].index}. Si usa il primo 'O'."
                )
        elif atom.atom_string == "H":
            hydrogen_atoms.append(atom)
    return oxygen_atom, hydrogen_atoms


# --- Funzione Principale ---
def calculate_hydrogen_unwrapped_from_oxygen_reference(
    frames_input: Union[Frame, List[Frame]], simulation_input: Simulation
) -> None:
    """
    Calcola le coordinate unwrapped degli atomi di Idrogeno (H)
    basandosi sulla 'unwrapped_position' dell'atomo di Ossigeno (O)
    nella stessa molecola, che si assume sia GIA' STATA CORRETTAMENTE IMPOSTATA.

    Modifica 'atom.unwrapped_position' per gli atomi H in-place.
    Non modifica 'atom.unwrapped_position' per gli atomi O.

    Args:
        frames_input: Un singolo oggetto Frame o una lista di oggetti Frame da processare.
        simulation_input: L'oggetto Simulation, utilizzato per accedere a 'cell_vectors'.
    """

    if isinstance(frames_input, Frame):
        frames_to_process = [frames_input]
    elif isinstance(frames_input, list) and all(
        isinstance(f, Frame) for f in frames_input
    ):
        frames_to_process = frames_input
    else:
        print(
            "Errore: 'frames_input' deve essere un oggetto Frame o una lista di oggetti Frame."
        )
        return

    if not frames_to_process:
        print("Nessun frame fornito per il processamento.")
        return

    cell_vectors = simulation_input.cell_vectors
    # Estrae le lunghezze della cella dalla diagonale dei vettori di cella.
    # Questo è accurato per celle ortorombiche. Per altre, è un'approssimazione.
    cell_lengths = np.diag(cell_vectors)

    if not np.allclose(cell_vectors, np.diag(cell_lengths)):
        print(
            "Attenzione: i vettori della cella di simulazione non sono strettamente diagonali."
        )
        print(
            "Le correzioni per le condizioni periodiche al contorno (PBC) si baseranno "
            "sulle componenti diagonali dei vettori di cella, come se la cella fosse ortorombica."
        )

    for current_frame in frames_to_process:
        if not current_frame.molecules:
            # print(f"Info: Frame {current_frame.index} non contiene molecole.")
            continue

        for molecule_as_list_of_atoms in current_frame.molecules:
            if not molecule_as_list_of_atoms:
                # print(f"Attenzione: Trovata una 'molecola' (lista di atomi) vuota nel frame {current_frame.index}.")
                continue

            oxygen_atom, hydrogen_atoms = _get_oxygen_and_hydrogens_in_molecule(
                molecule_as_list_of_atoms
            )

            if not oxygen_atom:
                # print(f"Attenzione: Nessun atomo di Ossigeno ('O') trovato nella molecola che inizia con "
                #       f"l'atomo indice {molecule_as_list_of_atoms[0].index} nel frame {current_frame.index}.")
                continue

            if oxygen_atom.unwrapped_position is None:
                print(
                    f"CRITICO: L'atomo di Ossigeno (indice {oxygen_atom.index}) nella molecola "
                    f"che inizia con l'atomo indice {molecule_as_list_of_atoms[0].index} nel frame {current_frame.index} "
                    f"NON ha una 'unwrapped_position' preimpostata. "
                    "Gli atomi di Idrogeno di questa molecola non possono essere unwrappati correttamente."
                )
                # Come fallback, si potrebbe impostare unwrapped_position = position per gli H
                # per evitare che rimangano None, se questo è il comportamento desiderato.
                for h_atom_fallback in hydrogen_atoms:
                    if h_atom_fallback.unwrapped_position is None:
                        h_atom_fallback.unwrapped_position = np.copy(
                            h_atom_fallback.position
                        )
                continue

            # Ora, oxygen_atom.unwrapped_position è la posizione di riferimento nota.
            #      oxygen_atom.position è la posizione PBC dell'ossigeno.

            for h_atom in hydrogen_atoms:
                # Calcola il vettore differenza tra la posizione PBC dell'Idrogeno e la posizione PBC dell'Ossigeno
                vector_H_O_in_pbc = h_atom.position - oxygen_atom.position

                # Correggi questo vettore differenza per le condizioni periodiche al contorno
                corrected_vector_H_O = np.copy(vector_H_O_in_pbc)
                for i in range(3):  # Itera sulle coordinate x, y, z
                    component_difference = corrected_vector_H_O[i]
                    dimension_length = cell_lengths[i]
                    half_dimension_length = dimension_length / 2.0

                    if component_difference > half_dimension_length:
                        corrected_vector_H_O[i] -= dimension_length
                    elif component_difference < -half_dimension_length:
                        corrected_vector_H_O[i] += dimension_length

                # La posizione unwrapped dell'Idrogeno è la 'unwrapped_position' (data) dell'Ossigeno
                # sommata al vettore differenza H-O corretto per il PBC.
                h_atom.unwrapped_position = (
                    oxygen_atom.unwrapped_position + corrected_vector_H_O
                )

    print(
        "Calcolo delle posizioni unwrapped degli Idrogeni (basato su Ossigeno di riferimento) completato."
    )
