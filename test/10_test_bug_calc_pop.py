import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from bbomol import run_optimization

output_path = '10_test_bug_calc_pop'
sillywalks_threshold = 1
dataset_path = os.environ["DATA"] + "/00_datasets/DFT/QM9/QM9.smi"


def load_random(path, size):
    with open(path, "r") as f:
        smiles_list = [line.split()[0] for line in f.readlines()]
    return list(np.random.choice(smiles_list, size))


def run(i):
    run_optimization({
        "io_parameters": {
            "results_path": output_path + str(i) + "/",
            "MM_program": "rdkit",
            "smiles_list_init": load_random(dataset_path, 1000)
        },
        "obj_function": "qed",
        "merit_optim_parameters": {
            "evomol_parameters": {
                "optimization_parameters": {
                    "max_steps": 10,
                },
                "action_space_parameters": {
                    "max_heavy_atoms": 50,
                    "atoms": "C,N,O,F,P,S,Cl,Br",
                    "sillywalks_threshold": sillywalks_threshold
                },
                "io_parameters": {
                    "silly_molecules_reference_db_path": os.environ[
                                                             "DATA"] + "/00_datasets/ChEMBL25/complete_ChEMBL_ecfp4_dict.json"
                }
            },
            "noise_based": False,
            "init_pop_zero_EI": False,
        },
        "bbo_optim_parameters": {
            "max_obj_calls": 2000
        },
        "surrogate_parameters": {
            "GPR_instance": GaussianProcessRegressor(1.0 * RBF(1.0) + WhiteKernel(1.0),
                                                     normalize_y=True),
            "descriptor": {
                "type": "shingles"
            }
        }
    })


run(0)
