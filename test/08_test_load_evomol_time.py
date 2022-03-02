from evomol import run_model

from bbomol.postprocessing.postprocessing import load_complete_input_results

model_path = "08_test_record_time"
model_path2 = "exp_QED_EvoMol/1"

# run_model({
#     "obj_function": {
#         "type": "linear_combination",
#         "functions": ["homo",
#                       "entropy_ifg",
#                       {
#                           "type": "mean",
#                           "functions": ["qed", "sascore"]
#
#                       }],
#         "coef": [1, 1, 1]
#     },
#     "io_parameters": {
#         "model_path": model_path,
#         "record_all_generated_individuals": True,
#         "smiles_list_init": ["C", "N"]
#     },
#     "optimization_parameters": {
#         "max_steps": 4
#     }
# })

results_dict = load_complete_input_results({}, EvoMol_experiments_dict={"evomol_exp1": model_path,
                                                                        "evomol_exp2": model_path2})

print(list(results_dict["evomol_exp1"].keys()))
print(list(results_dict["evomol_exp1"]["dataset_success_success_obj_computation_time"]))
print(list(results_dict["evomol_exp1"]["dataset_failed_success_obj_computation_time"]))


print(list(results_dict["evomol_exp2"].keys()))
print(list(results_dict["evomol_exp2"]["dataset_success_success_obj_computation_time"]))
print(list(results_dict["evomol_exp2"]["dataset_failed_success_obj_computation_time"]))
print(results_dict["evomol_exp2"])