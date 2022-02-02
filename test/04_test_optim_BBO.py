import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from bbomol import run_optimization

sillywalks_threshold = 1
output_path = "04_test_optim"

def run(i):

    run_optimization({
        "io_parameters":{
            "results_path": output_path + str(i) + "/",
            "MM_program": "rdkit",
            "dft_working_dir": os.environ["DFT_COMPUT_RDKIT_MM"]
        },
        "obj_function": "homo",
        "merit_optim_parameters": {
            "evomol_parameters": {
                "optimization_parameters": {
                    "max_steps": 10,
                },
                "action_space_parameters": {
                    "max_heavy_atoms": 9,
                    "atoms": "C,N,O,F",
                    "sillywalks_threshold": sillywalks_threshold
                },
                "io_parameters":{
                    "silly_molecules_reference_db_path": os.environ["DATA"] + "/00_datasets/ChEMBL25/complete_ChEMBL_ecfp4_dict.json"
                }
            }
        },
        "bbo_optim_parameters": {
            "max_obj_calls": 10
        },
        "surrogate_parameters": {
            "GPR_instance": GaussianProcessRegressor(1.0*RBF(1.0) + WhiteKernel(1.0),
                                                     normalize_y=True),
            "descriptor": {
                "type": "MBTR"
            }
        }
    })

run(0)
