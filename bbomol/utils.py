from ase import Atoms


def convert_rdkit_mol_to_symbols_positions(mol):
    """
    Converting a RDKit molecule object to tuple (list of symbols, list of positions)
    :param mol:
    :return:
    """
    symbols = []
    positions = []

    # Iterating over atoms to extract symbols and positions
    for at_idx in range(mol.GetNumAtoms()):
        symbols.append(mol.GetAtomWithIdx(at_idx).GetSymbol())
        pos_3d = mol.GetConformer(0).GetAtomPosition(at_idx)
        positions.append([pos_3d.x, pos_3d.y, pos_3d.z])

    return symbols, positions


def convert_rdkit_mol_to_ase_atoms(mol):
    """
    Converting a RDKit molecule object to a ase.Atoms object
    :param mol:
    :return:
    """

    symbols, positions = convert_rdkit_mol_to_symbols_positions(mol)

    # Converting molecule to ASE Atoms
    return Atoms(symbols=symbols, positions=positions)
