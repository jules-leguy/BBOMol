import os

import tqdm
from evomol.molgraphops.molgraph import MolGraph
from rdkit.Chem import MolFromSmiles

dataset_path = os.environ["DATA"] + "/00_datasets/DFT/QM9/QM9.smi"

with open(dataset_path, "r") as f:
    smiles_list = [line.split()[0] for line in f.readlines()]


for smi in tqdm.tqdm(smiles_list):
    if MolGraph(MolFromSmiles(smi)).to_aromatic_smiles() != smi:
        print("different !")
        print(smi)
        print(MolGraph(MolFromSmiles(smi)).to_aromatic_smiles())