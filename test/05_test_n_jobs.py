from bbomol import run_optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

results_path = "05_test_n_jobs"

run_optimization({
    "obj_function": "homo",
    "merit_optim_parameters": {
        "merit_type": "EI",
    },
    "surrogate_parameters": {
        "GPR_instance": GaussianProcessRegressor(1.0 * RBF(1.0) + WhiteKernel(1.0), normalize_y=True),
        "descriptor": {
            "type": "MBTR"
        }
    },
    "bbo_optim_parameters": {
        "max_obj_calls": 20
    },
    "io_parameters": {
        "dft_n_jobs": 4,
        "results_path": results_path
    }
})