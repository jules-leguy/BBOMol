from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from bbomol import run_optimization

run_optimization({
    "obj_function": "homo",
    "merit_optim_parameters": {
        "evomol_parameters": {
            "optimization_parameters": {
                "max_steps": 10,
            },
            "action_space_parameters": {
                "max_heavy_atoms": 9,
                "atoms": "C,N,O,F"
            }
        }
    },
    "surrogate_parameters": {
        "GPR_instance": GaussianProcessRegressor(1.0*RBF(1.0)+WhiteKernel(1.0), normalize_y=True),
        "descriptor": {
            "type": "MBTR"
        }
    }
})
