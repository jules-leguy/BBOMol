import csv
import glob
from os.path import join, exists
from pathlib import Path
import pandas as pd
import numpy as np


def convert_timestamps_best_to_homogeneous_mapping(timestamps_list, best_scores_list, max_time=1e6):

    timestamps_arr = np.array(timestamps_list)
    best_scores_arr = np.array(best_scores_list)

    # Converting timestamp to predefined values
    homogeneous_timestamps = np.arange(0, max_time, 100)
    best_score_homogeneous_timestamps = []

    for t in homogeneous_timestamps:

        # Computing list of scores obtained in time lesser or equal than current timestamp
        scores_curr_timestamp = best_scores_arr[timestamps_arr <= t]

        if len(scores_curr_timestamp) > 0:

            best_score_homogeneous_timestamps.append(
                np.max(scores_curr_timestamp)
            )

        else:
            best_score_homogeneous_timestamps.append(np.nan)

    return homogeneous_timestamps, np.array(best_score_homogeneous_timestamps)


def extract_time_best_BBO(bbo_exp_root, max_time=1e6):
    """
    Returning vectors of homogeneous timestamps with corresponding best scores for a BBO experiment
    :param bbo_exp_root:
    :param max_time:
    :return:
    """

    step_to_time = {}

    # Reading steps.csv as a pandas DataFrame
    steps_df = pd.read_csv(join(bbo_exp_root, "steps.csv"))

    try:
        # Extracting total time and corresponding steps vectors
        times = steps_df["alg_total_time"]
        steps = steps_df["step"]

        # Writing data as a dict
        for i, step in enumerate(steps):
            step_to_time[step] = times[i]

        # Reading dataset of solutions
        dataset_df = pd.read_csv(join(bbo_exp_root, "dataset.csv"))

        timestamps_list = []
        best_scores_list = []
        curr_max = -float("inf")

        # Building timestamp/best score mapping
        for step in np.unique(dataset_df["step"]):

            curr_step_best_score = np.max(dataset_df["obj_value"][dataset_df["step"] == step])

            if curr_step_best_score > curr_max:
                curr_max = curr_step_best_score

            timestamps_list.append(step_to_time[step])
            best_scores_list.append(curr_max)

        homogeneous_timestamps, best_score_homogeneous_timestamps = convert_timestamps_best_to_homogeneous_mapping(timestamps_list, best_scores_list, max_time=max_time)
        return {
            "timestamps": homogeneous_timestamps,
            "best_scores_timestamps": best_score_homogeneous_timestamps,
            "effective_last_timestamp": timestamps_list[-1]
        }

    except KeyError as e:
        return {
            "timestamps": [],
            "best_scores_timestamps": [],
            "effective_last_timestamp": None
        }


def extract_time_best_EvoMol(evomol_exp_root, max_time=1e6):
    """
    Returning vectors of homogeneous timestamps with corresponding best scores for an EvoMol experiment
    :param evomol_exp_root:
    :param max_time:
    :return:
    """

    # Reading steps.csv as a pandas DataFrame
    steps_df = pd.read_csv(join(evomol_exp_root, "steps.csv"))

    timestamps_list = []
    best_scores_list = []

    for i, timestamp in enumerate(steps_df["timestamps"]):
        timestamps_list.append(timestamp)
        best_scores_list.append(steps_df["total_max"][i])

    homogeneous_timestamps, best_score_homogeneous_timestamps = convert_timestamps_best_to_homogeneous_mapping(timestamps_list, best_scores_list, max_time=max_time)

    return {
        "timestamps": homogeneous_timestamps,
        "best_scores_timestamps": best_score_homogeneous_timestamps,
        "effective_last_timestamp": timestamps_list[-1]
    }


def extract_BBO_dataset(bbo_exp_root, include_dataset_init_step=False):
    """
    Extracting the data contained in a BBO dataset.csv file.
    :param dataset_path: path to the dataset.csv file
    :param include_dataset_init_step: whether to include DOE SMILES (SMILES at step 0)
    :return:
    """

    success_dataset_introduction_step = []
    success_dataset_smiles = []
    success_dataset_obj_value = []
    success_dataset_n_calls = []

    failed_step = []
    failed_smiles = []
    failed_descriptors = []
    failed_objective = []
    failed_n_calls = []

    contains_success_col = None

    with open(join(bbo_exp_root, "dataset.csv"), "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):

            if i == 0:

                # Checking if contains success column
                contains_success_col = "success" in row

                # Extracting additional columns
                row_size = len(row)
                row_idx_to_key = {}
                additional_data_dict_dataset = {}
                for j in range(4, row_size):
                    row_idx_to_key[j] = row[j]
                    additional_data_dict_dataset["dataset_success_" + row_idx_to_key[j]] = []
                    additional_data_dict_dataset["dataset_failed_" + row_idx_to_key[j]] = []
            else:

                # Extracting current row values
                step, smiles, obj_value, n_calls = row[0], row[1], row[2], row[3]
                if contains_success_col:
                    success = row[4]
                else:
                    success = True

                # Recording data if conditions are met
                if include_dataset_init_step or int(step) > 0:

                    if success:
                        success_dataset_introduction_step.append(int(step))
                        success_dataset_smiles.append(smiles)
                        success_dataset_obj_value.append(float(obj_value))
                        success_dataset_n_calls.append(int(n_calls))

                        # Extracting data in additional columns
                        for j in range(4, row_size):
                            additional_data_dict_dataset["dataset_success_" + row_idx_to_key[j]].append(row[j])

                    else:
                        failed_step.append(int(step))
                        failed_smiles.append(smiles)
                        failed_descriptors.append(None)
                        failed_objective.append(None)
                        failed_n_calls.append(int(n_calls))

                        # Extracting data in additional columns
                        for j in range(4, row_size):
                            additional_data_dict_dataset["dataset_failed_" + row_idx_to_key[j]].append(row[j])

    # Extracting data in failed_smiles.csv file if exists
    if exists(join(bbo_exp_root, "failed_dataset.csv")):
        with open(join(bbo_exp_root, "failed_dataset.csv"), "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):

                # Extracting additional columns
                if i == 0:
                    row_size = len(row)
                    row_idx_to_key = {}
                    additional_data_dict_failed_dataset = {}
                    for j in range(5, row_size):
                        row_idx_to_key[j] = row[j]
                        additional_data_dict_failed_dataset["dataset_failed_"+row_idx_to_key[j]] = []

                if i > 0:

                    # Recording data if conditions are met
                    if include_dataset_init_step or int(step) > 0:

                        # Extracting data in deterministic columns
                        step, smiles, n_calls, failed_desc, failed_obj = row[0], row[1], row[2], row[3], row[4]
                        failed_step.append(int(step))
                        failed_smiles.append(smiles)
                        failed_descriptors.append(failed_desc == "True")
                        failed_objective.append(failed_obj == "True")
                        failed_n_calls.append(int(n_calls))

                        # Extracting data in additional columns
                        for j in range(5, row_size):
                            additional_data_dict_failed_dataset["dataset_failed_"+row_idx_to_key[j]].append(row[j])

    output_dict = {
        "dataset_success_step": success_dataset_introduction_step,
        "dataset_success_smiles": success_dataset_smiles,
        "dataset_success_obj_value": success_dataset_obj_value,
        "dataset_success_n_calls": success_dataset_n_calls,
        "dataset_failed_step": failed_step,
        "dataset_failed_smiles": failed_smiles,
        "dataset_failed_descriptor": failed_descriptors,
        "dataset_failed_objective": failed_objective,
        "dataset_failed_n_calls": failed_n_calls
    }

    output_dict.update(additional_data_dict_dataset)
    output_dict.update(additional_data_dict_failed_dataset)

    return output_dict

def extract_BBO_test_MAE(bbo_exp_root):
    """
    Extracting the test MAE values vs. the steps
    If there are multiple test datasets, recording the values in distinct lists
    :param bbo_exp_root:
    :return:
    """

    # Finding path of all datasets values
    files = glob.glob(join(bbo_exp_root, "surrogate/") + "test_predictions_*.csv")

    results_dict = {}

    for filepath in files:

        name = Path(filepath).name.replace(".csv", "")

        results_dict["test_step"] = []
        results_dict[name] = []

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i > 0:
                    results_dict["test_step"].append(int(row[0]))
                    results_dict[name].append(float(row[1]))

    return results_dict


def extract_BBO_steps(bbo_exp_root):
    """
    Extracting data from steps.csv file
    :param bbo_exp_root:
    :return:
    """

    data_dict = {}

    # Opening steps.csv file
    with open(join(bbo_exp_root, "steps.csv"), "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                keys = []
                for k in row:
                    keys.append(k)
                    data_dict["steps_" + k] = []
            else:
                for j in range(len(row)):
                    data_dict["steps_" + keys[j]].append(float(row[j]))

    return data_dict


def extract_BBO_experiment_data(bbo_exp_root, include_dataset_init_step=False):
    """
    Extracting all data stored for the given experiment and returning it in a dictionary
    :param bbo_exp_root:
    :param include_dataset_init_step:
    :return:
    """

    results_dict = {}

    # Extracting dataset data
    results_dict.update(extract_BBO_dataset(bbo_exp_root, include_dataset_init_step=include_dataset_init_step))

    # Extracting steps data
    results_dict.update(extract_BBO_steps(bbo_exp_root))

    # Extracting test data
    results_dict.update(extract_BBO_test_MAE(bbo_exp_root))

    # Extracting time data
    results_dict.update(extract_time_best_BBO(bbo_exp_root))

    return results_dict


def extract_multiple_BBO_experiments_data(bbo_multiple_data_root, experiments_folder_names,
                                          include_dataset_init_step=False):
    """
    Extracting the data of multiple experiments. All the experiments must be as different folders in the given root path.
    Merging the results of the call to extract_experiment_data function. The keys are the same, but the values are a
    list of corresponding values extracted from the distinct experiments.
    :param bbo_multiple_data_root:
    :param include_dataset_init_step:
    :return:
    """

    # Creation of output dictionary
    output_dict = {}

    # Initialization of keys based on the results of the first folder
    keys = extract_BBO_experiment_data(join(bbo_multiple_data_root, experiments_folder_names[0])).keys()
    for k in keys:
        output_dict[k] = []

    # Extracting all results
    for folder_name in experiments_folder_names:
        exp_path = join(bbo_multiple_data_root, folder_name)
        if exists(exp_path):
            curr_model_data = extract_BBO_experiment_data(exp_path,
                                                          include_dataset_init_step=include_dataset_init_step)

            for k in keys:
                output_dict[k].append(curr_model_data[k])

    return output_dict


def extract_evomol_experiment_data(evomol_exp_root):
    """
    Extracting the data of an EvoMol experiment with the same output as extract_BBO_experiment_data function
    :param evomol_exp_root:
    :return:
    """

    output_dict = {
        "dataset_success_step": [],
        "dataset_success_smiles": [],
        "dataset_success_obj_value": [],
        "dataset_success_n_calls": [],
        "dataset_failed_step": [],
        "dataset_failed_smiles": [],
        "dataset_failed_objective": [],
        "dataset_failed_n_calls": []
    }

    # Initialization of the variable that represents whether the data contains information about solutions that failed
    # the objective computation
    contains_failed_obj_data = None

    # Reading data
    with open(join(evomol_exp_root, "all_generated.csv"), "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):

            # Checking if file contains information about the failed solutions
            if i == 0:
                contains_failed_obj_data = "success_obj_computation" in row
            else:

                # Checking whether the current row is a success
                if not contains_failed_obj_data or row[5] == "True":
                    success_or_failed_key = "success"
                else:
                    success_or_failed_key = "failed"

                # Data common to success and failure case
                output_dict["dataset_" + success_or_failed_key + "_step"].append(int(row[0]))
                output_dict["dataset_" + success_or_failed_key + "_smiles"].append(row[1])
                output_dict["dataset_" + success_or_failed_key + "_n_calls"].append(int(row[2]))

                # Data specific to success
                if success_or_failed_key == "success":
                    output_dict["dataset_success_obj_value"].append(float(row[3]))

                # Data specific to failure
                if success_or_failed_key == "failed":
                    output_dict["dataset_failed_objective"].append(row[5] == "False")

    # Adding timestamps data
    output_dict.update(extract_time_best_EvoMol(evomol_exp_root))

    return output_dict


def extract_multiple_evomol_experiments_data(evomol_multiple_data_root, experiments_folder_names):
    """
    Extracting the data of multiple EvoMol experiments in a way that is consistent with
    extract_multiple_BBO_experiments_data function.
    :param evomol_multiple_data_root:
    :param experiments_folder_names:
    :return:
    """

    # Creation of output dictionary
    output_dict = {}

    # Initialization of keys based on the results of the first folder
    keys = extract_evomol_experiment_data(join(evomol_multiple_data_root, experiments_folder_names[0])).keys()
    for k in keys:
        output_dict[k] = []

    # Extracting all results
    for folder_name in experiments_folder_names:

        if exists(join(evomol_multiple_data_root, folder_name)):

            curr_model_data = extract_evomol_experiment_data(join(evomol_multiple_data_root, folder_name))

            for k in keys:
                output_dict[k].append(curr_model_data[k])

    return output_dict
