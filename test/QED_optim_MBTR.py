import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from bbomol import run_optimization


def run(i):

    run_optimization({
        "io_parameters":{
            "results_path": "exp_QED_MBTR_RBF/" + str(i),
            "MM_program": "rdkit"
        },
        "obj_function": "qed",
        "merit_optim_parameters": {
            "evomol_parameters": {
                "optimization_parameters": {
                    "max_steps": 10,
                },
                "action_space_parameters": {
                    "max_heavy_atoms": 38,
                    "atoms": "C,N,O,F"
                }
            }
        },
        "bbo_optim_parameters": {
            "max_obj_calls": 10000
        },
        "surrogate_parameters": {
            "GPR_instance": GaussianProcessRegressor(1.0*RBF(1.0) + WhiteKernel(1.0), normalize_y=True),
            "descriptor": {
                "type": "MBTR"
            }
        },
        "parallelization": {
            "n_jobs_merit_optim_restarts": 10,
            "n_jobs_desc_comput": 10,
        }
    })


for i in range(1, 6):
    run(i)
