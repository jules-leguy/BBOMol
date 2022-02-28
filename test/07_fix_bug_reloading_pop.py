from bbomol.postprocessing.plot import draw_best_solutions
from bbomol.postprocessing.postprocessing import load_complete_input_results

pop_path = "/home/jleguy/Downloads/"

results_dict = load_complete_input_results(BBO_experiments_dict={}, EvoMol_experiments_dict={"exp_evomol": pop_path})
print(list(results_dict["exp_evomol"].keys()))


for k, v in results_dict["exp_evomol"].items():
    try:
        print(k + " : " + str(len(v[0])))
    except Exception as e:
        print(k + " : " + str(v[0]) + "(unique value)")

draw_best_solutions(results_dict, output_dir_path=pop_path, properties=["obj_value", "homo"])
