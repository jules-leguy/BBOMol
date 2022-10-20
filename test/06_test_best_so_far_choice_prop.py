import os

from evomol import QEDEvaluationStrategy

from bbomol.postprocessing.plot import plot_best_so_far
from bbomol.postprocessing.postprocessing import load_complete_input_results

paths = {
    "TTF": os.environ["DATA"] + "/07_BBO/03_bbo_optim/post_paper/02.01_TTF/",
    "TTF sillywalks": os.environ["DATA"] + "/07_BBO/03_bbo_optim/post_paper/02.04_TTF_sillywalks_CHEMBL_0/"
}

results_dict = load_complete_input_results(paths, sub_experiment_names=["1"])

print(results_dict["TTF sillywalks"].keys())

plot_best_so_far(results_dict)
plot_best_so_far(results_dict, prop="homo")
plot_best_so_far(results_dict, prop=QEDEvaluationStrategy())