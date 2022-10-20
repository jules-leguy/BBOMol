import os

import numpy as np
from evomol import run_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from bbomol import run_optimization
from bbomol.postprocessing.plot import plot_best_so_far
from bbomol.postprocessing.postprocessing import load_complete_input_results

dataset_path = os.environ["DATA"] + "/00_datasets/DFT/QM9/QM9.smi"


def load_random(path, size):
    with open(path, "r") as f:
        smiles_list = [line.split()[0] for line in f.readlines()]
    return list(np.random.choice(smiles_list, size))


def run_test(path, desc):
    run_optimization({
        "io_parameters": {
            "results_path": path,
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
                },
            },
            "merit_type": "EI",
            "noise_based": True,
            "init_pop_zero_EI": False,
            "merit_xi": 0.0
        },
        "bbo_optim_parameters": {
            "max_obj_calls": 100
        },
        "surrogate_parameters": {
            "GPR_instance": GaussianProcessRegressor(1.0 * RBF(1.0) + WhiteKernel(1.0),
                                                     normalize_y=True),
            "descriptor": {
                "type": desc
            }
        }
    })


# run_test("15_shingles_desc", "shingles")
run_test("15_random_desc", "random")

# run_model({
#     "obj_function": "qed",
#     "io_parameters": {
#         "model_path": "13_test_EvoMol",
#         "record_all_generated_individuals": True
#     },
#     "optimization_parameters":{
#         "max_obj_calls": 100
#     },
#     "action_space_parameters": {
#         "max_heavy_atoms": 50,
#         "atoms": "C,N,O,F,P,S,Cl,Br",
#     },
# })


results_dict = load_complete_input_results(
    BBO_experiments_dict={"shingles": "15_shingles_desc/", "random": "15_random_desc/"},
    EvoMol_experiments_dict={"EvoMol": "13_test_EvoMol/"},
    include_dataset_init_step=True
)

plot_best_so_far(results_dict, metric="both", plot_last_common_data_all_runs=True)


