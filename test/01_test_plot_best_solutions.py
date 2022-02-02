import os

from evomol import SAScoreEvaluationStrategy, CLScoreEvaluationStrategy

from bbomol.postprocessing.plot import draw_best_solutions
from bbomol.postprocessing.postprocessing import load_complete_input_results

paths = {
    "TTF": os.environ["DATA"] + "/07_BBO/03_bbo_optim/post_paper/02.01_TTF/",
    "TTF sillywalks": os.environ["DATA"] + "/07_BBO/03_bbo_optim/post_paper/02.04_TTF_sillywalks_CHEMBL_0/"
}

results_dict = load_complete_input_results(paths, sub_experiment_names=["1"])

print(results_dict["TTF sillywalks"].keys())
draw_best_solutions(results_dict, properties=["obj_value", SAScoreEvaluationStrategy(), CLScoreEvaluationStrategy()])