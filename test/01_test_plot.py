import numpy as np

from bbomol.postprocessing.plot import plot_best_so_far, plot_ecdf, display_ert, plot_stable_dynamics
from bbomol.postprocessing.postprocessing import load_complete_input_results

BBO_experiments_dict = {
    "Baseline": "test_input_results/01.01_MBTR_RBF_alpha_0_dot_1",
    "White kernel": "test_input_results/01.04_MBTR_RBF_whitekernel",
}

EvoMol_experiments_dict = {
    "EvoMol": "test_input_results/01.01_EvoMol_from_methane_optim_HOMO_rdkit_nocache"
}

results_dict = load_complete_input_results(BBO_experiments_dict, EvoMol_experiments_dict,
                                           [str(i) for i in range(1, 11)])

plot_best_so_far(results_dict, exp_list_plot=["Baseline", "White kernel", "EvoMol"], metric="both",
                 output_dir_path=".", plot_name="test", classes_dashes=[0, 0, 1])

plot_ecdf(results_dict, ecdf_targets=np.arange(-10, -1, 1e-2), xunit="calls", output_dir_path=".")

plot_ecdf(results_dict, ecdf_targets=np.arange(-10, -1, 1e-2), xunit="time", output_dir_path=".")

display_ert(results_dict, ert_targets=np.arange(-10, -1, 1), xunit="calls",
            exp_list_plot=["Baseline", "White kernel", "EvoMol"], plot_title="ERT")

display_ert(results_dict, ert_targets=np.arange(-10, -1, 1), xunit="time",
            exp_list_plot=["Baseline", "White kernel", "EvoMol"], plot_title="ERT")

plot_stable_dynamics(results_dict, output_dir_path=".")
