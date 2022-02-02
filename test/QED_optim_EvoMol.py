from evomol import run_model
from os.path import join
from evomol.evaluation_dft import OPTEvaluationStrategy


def run(i):
    run_model({
        "obj_function": "qed",
        "optimization_parameters": {
            "max_steps": float("inf"),
            "max_obj_calls": 10000,
            "pop_max_size": 300,
            "k_to_replace": 10,
            "problem_type": "max",
        },
        "io_parameters": {
            "model_path": "exp_QED_EvoMol/" + str(i),
            "smiles_list_init": ["C"],
            "save_n_steps": 1,
            "record_all_generated_individuals": True
        },
        "action_space_parameters": {
            "atoms": "C,N,O,F",
            "max_heavy_atoms": 38,
        }
    })

for i in range(1, 5):
    run(i)