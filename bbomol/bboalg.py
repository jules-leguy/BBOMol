import csv
import time
from os import makedirs
from os.path import join

import numpy as np
from evomol import run_model
from joblib import dump, Parallel, delayed
from sklearn.metrics import r2_score
from .stop_criterion import MultipleStopCriterion, FileStopCriterion


def save_dict_to_csv(d, csv_path, ignore_keys=None):
    """
    Saving the content of a dictionary to a csv file at given path.
    Each column is represented by an entry (column name, list data).
    :param d: dictionary to be written in file
    :param csv_path: path in which the CSV file is written
    :param ignore_keys: list of keys that won't be written
    :return:
    """

    # Initialization of csv array
    csv_array = []

    # Filling array
    for k, v in d.items():
        if ignore_keys is None or k not in ignore_keys:
            csv_array.append([k] + list(v))

    # Saving csv file
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        for row in np.array(csv_array).T:
            writer.writerow(row)


def compute_descriptors(smiles_list, descriptor, pipeline, apply_pipeline=False):
    """
    Computing the descriptors of given SMILES list. The molecules that cause an error during the computation of the
    descriptors are ignored and absent of the resulting data. If the apply_pipeline parameter is set to True,
    the given instance of sklearn.pipeline.Pipeline is used to transform the resulting descriptors.
    :param smiles_list: smiles list from which descriptors will be computed
    :param descriptor: instance of bbo.descriptor.Descriptor
    :param pipeline: instance of sklearn.pipeline.Pipeline
    :param apply_pipeline: whether to apply the pipeline on the resulting descriptors
    :return: (descriptors matrix, list of SMILES after processing, success boolean filter on input smiles list)
    """

    # Computing descriptors
    X, success = descriptor.fit_transform(smiles_list)

    # Filtering errors from the descriptors and the from smiles list
    X, smiles_filtered = X[success], np.array(smiles_list)[success]

    if apply_pipeline and pipeline is not None:
        X = pipeline.transform(X)

    return X, smiles_filtered, success


def compute_evomol_init_pop(evomol_init_pop_strategy, dataset_smiles, dataset_y, evomol_init_pop_size):
    """
    Computing the initial population of an EvoMol execution to optimize the merit function. Depends on the strategy
    defined by the evomol_init_pop_strategy value
    :param evomol_init_pop_strategy: string representing the strategy to select the solutions of the dataset to be part
    of the EvoMol init population. "methane" : starting from methane molecule only. "best" : best objective values from
    dataset. "best_weighted": random selection from dataset weighted by objective function value. "random": uniform
    random selection from dataset.
    :param dataset_smiles: list of SMILES of the dataset
    :param dataset_y: list of objective values of solutions of the dataset
    :param evomol_init_pop_size: size of the EvoMol initial population to be built
    :return: list of SMILES
    """

    # Selecting only methane
    if evomol_init_pop_strategy == "methane":
        return ["C"]

    # Selecting the SMILES from the dataset randomly with a uniform law
    elif evomol_init_pop_strategy == "random":
        return np.random.choice(dataset_smiles, min(evomol_init_pop_size, len(dataset_smiles)),
                                replace=False)

    # Selecting the SMILES from the dataset randomly with a probability proportional to the objective value (the
    # higher the more probable)
    elif evomol_init_pop_strategy == "random_weighted":

        # If negative scores exist, shifting all values so that minimum is 0
        if np.min(dataset_y) < 0:
            dataset_y_shifted = dataset_y - np.min(dataset_y)
        else:
            dataset_y_shifted = dataset_y

        # Setting all zero objective values to a small number so that probability computation can be proceed
        dataset_y_corrected = dataset_y_shifted
        dataset_y_corrected[dataset_y_corrected == 0] = 1e-10

        return np.random.choice(dataset_smiles, min(evomol_init_pop_size, len(dataset_smiles)),
                                replace=False, p=dataset_y_corrected / dataset_y_corrected.sum())

    # Selecting the SMILES from the dataset with the highest objective values
    elif evomol_init_pop_strategy == "best":
        return np.array(dataset_smiles)[np.argsort(dataset_y)[::-1][:min(evomol_init_pop_size,
                                                                         len(dataset_smiles))]]


def run_evomol_merit_optimization(merit_function, results_path, evomol_parameters, evomol_init_pop_strategy,
                                  dataset_smiles, dataset_y, evomol_init_pop_size):
    """
    Running an EvoMol optimization of the merit function.
    The selected individuals from the resulting population are returned as a SMILES list.
    :param merit_function: bbo.merit.Merit instance (function to be optimized by EvoMol)
    :param results_path: path in which results will be stored
    :param evomol_parameters: parameters to be given to EvoMol for optimization. Only the "action_space_parameters"
    and "optimization_parameters" attributes are considered
    :param n_best_evomol_retrieved: number of (best) solutions to be selected
    :param evomol_init_pop_strategy: strategy to select solutions from the dataset to be part of the initial population
    of EvoMol (see the compute_evomol_init_pop function).
    :param dataset_smiles: list of SMILES of the dataset
    :param dataset_y: list of objective values of solutions of the dataset
    :param evomol_init_pop_size: size of the EvoMol initial population
    :return: instance of EvoMol optimization PopAlg
    """

    print("run evomol optimization")

    parameters = {
        "obj_function": merit_function,
        "io_parameters": {
            "model_path": join(results_path, "EvoMol_optimization"),
            "smiles_list_init": compute_evomol_init_pop(evomol_init_pop_strategy=evomol_init_pop_strategy,
                                                        dataset_smiles=dataset_smiles, dataset_y=dataset_y,
                                                        evomol_init_pop_size=evomol_init_pop_size),
            "external_tabu_list": dataset_smiles,
            "save_n_steps": 200,
            "print_n_steps": 1
        }
    }

    if "action_space_parameters" in evomol_parameters:
        parameters["action_space_parameters"] = evomol_parameters["action_space_parameters"]
    if "optimization_parameters" in evomol_parameters:
        parameters["optimization_parameters"] = evomol_parameters["optimization_parameters"]

    # Running EvoMol optimization
    optimized_evomol_instance = run_model(
        parameters
    )

    return optimized_evomol_instance


class BBOAlg:

    def __init__(self, init_dataset_smiles, descriptor, objective, merit_function, surrogate, stop_criterion,
                 evomol_parameters, evomol_init_pop_size, n_evomol_runs, n_best_evomol_retrieved,
                 evomol_init_pop_strategy, objective_init_pop=None, score_assigned_to_failed_solutions=None,
                 results_path="BBO_results", n_jobs=1, pre_dispatch='2 * n_jobs', batch_size='auto',
                 save_surrogate_model=False, period_compute_test_predictions=50, period_save=50,
                 save_pred_test_values=False, test_dataset_smiles_dict=None, test_objectives_dict=None, pipeline=None):
        """
        Main class performing BBO optimization.
        :param init_dataset_smiles: smiles list to be used as initial dataset
        :param test_dataset_smiles_dict: dictionary of smiles list to be used as surrogate function test sets
        :param test_objectives_dict : dictionary of bbo.objective.EvoMolEvaluationStrategyWrapper used to evaluate
        the objective values of each test set
        :param descriptor: bbo.descriptor.Descriptor instance building the ML descriptors
        :param objective: bbo.objective.EvoMolEvaluationStrategyWrapper wrapping an EvoMol EvaluationStrategy instance
        :param merit_function: bbo.merit.Merit instance to compute the values of merit function
        :param surrogate: bbo.model.SurrogateModel instance
        :param stop_criterion: bbo.stop_criterion.StopCriterion instance
        :param pipeline: sklearn.pipeline.Pipeline instance whose fit and transform methods are applied on the train set,
        and whose transform method is applied on the tet set at each training and prediction
        :param evomol_parameters: parameters to be given to EvoMol for optimization. Only the "action_space_parameters"
        and "optimization_parameters" attributes are considered by BBOAlg.
        :param evomol_init_pop_size: determines the initial population size of EvoMol optimization
        :param n_evomol_runs: number of EvoMol optimization runs at each step
        :param n_best_evomol_retrieved: number of (best) solutions to be selected to be introduced to the dataset
        from each EvoMol optimization run
        :param evomol_init_pop_strategy: strategy to initialize evomol population. "methane" : starting from methane
        molecule only. "best" : best objective values from dataset. "best_weighted": random selection from dataset
        weighted by objective function value. "random": uniform random selection from dataset.
        :param objective_init_pop: bbo.objective.EvoMolEvaluationStrategyWrapper wrapping an EvoMol EvaluationStrategy
        instance to be used on initial population. If not, set as objective instance
        :param score_assigned_to_failed_solutions: if None then default behaviour : solutions which fail the descriptor
        of objective are discarded. If not None, they are kept and assigned the given value as objective value.
        :param results_path: path in which results of the BBO optimization will be stored
        :param n_jobs: number of jobs to be used for parallel computations
        :param save_surrogate_model: whether the surrogate model should be saved along with results data
        :param period_compute_test_predictions: period in steps of computation and recording of predictions on the
        test dataset
        :param period_save: period in steps of saving data in the results directory;
        :param save_pred_test_values: whether to save the predicted values for all test samples in
        surrogate/test_predictions.csv
        """

        self.init_dataset_smiles = init_dataset_smiles
        self.test_dataset_smiles_dict = test_dataset_smiles_dict if test_dataset_smiles_dict is not None else {}
        self.test_objective_dict = test_objectives_dict if test_objectives_dict is not None else {}

        self.descriptor = descriptor
        self.objective = objective
        self.objective_init_pop = objective if objective_init_pop is None else objective_init_pop
        self.merit_function = merit_function
        self.surrogate = surrogate
        self.stop_criterion = stop_criterion
        self.pipeline = pipeline
        self.score_assigned_to_failed_solutions = score_assigned_to_failed_solutions

        self.evomol_parameters = evomol_parameters
        self.evomol_init_pop_size = evomol_init_pop_size
        self.n_evomol_runs = n_evomol_runs
        self.n_best_evomol_retrieved = n_best_evomol_retrieved
        self.evomol_init_pop_strategy = evomol_init_pop_strategy

        self.results_path = results_path
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.save_surrogate_model = save_surrogate_model
        self.period_compute_test_predictions = period_compute_test_predictions
        self.period_save = period_save
        self.save_pred_test_values = save_pred_test_values

        # Variable representing the current step
        self.curr_step = 0

        # Dictionary containing all information about the dataset of solutions
        self.dataset_dict = {
            "step": np.array([]),  # Step of introduction
            "smiles": np.array([]),  # SMILES
            "X": np.array([]),  # Descriptor values
            "obj_value": np.array([]),  # Objective values
            "n_obj_calls": np.array([]),  # Number of calls to the objective function at the time of evaluation
            "success": np.array([]),
            "success_obj_computation_time": np.array([])
        }
        for key in self.objective.keys():
            self.dataset_dict[key] = np.array([])
        for key in self.merit_function.keys() + ["value"]:
            self.dataset_dict["merit_" + key] = np.array([])

        # Dictionary containing all information about the proposed solutions that failed
        self.failed_dataset_dict = {
            "step": np.array([]),  # Step of generation
            "smiles": np.array([]),  # SMILES
            "n_obj_calls": np.array([]),  # Number of calls to the objective function at the time of evaluation
            "failed_descriptors": np.array([]),  # Whether the computation of descriptors has failed
            "failed_objective": np.array([])  # Whether the computation of an objective value has failed
        }
        for key in self.merit_function.keys() + ["value"]:
            self.failed_dataset_dict["merit_" + key] = np.array([])

        # Dictionary containing all information about the test datasets
        self.test_datasets_dicts = {}
        for k in self.test_dataset_smiles_dict.keys():
            self.test_datasets_dicts[k] = {
                "smiles": np.array([]),  # SMILES of solutions in the test dataset
                "X": np.array([]),  # Descriptor values
                "obj_value": np.array([]),  # Objective values
                "success": np.array([])
            }
            for obj_key in self.test_objective_dict[k].keys():
                self.test_datasets_dicts[k][obj_key] = np.array([])

        # Dictionary containing all information about the test dataset solutions that failed
        self.failed_test_datasets_dicts = {}
        for k in self.test_dataset_smiles_dict.keys():
            self.failed_test_datasets_dicts[k] = {
                "smiles": np.array([]),  # SMILES
                "failed_descriptors": np.array([]),  # Whether the computation of descriptors has failed
                "failed_objective": np.array([])  # Whether the computation of an objective value has failed
            }

        # Variable containing the trace of predictions of the surrogate on the test set
        self.test_predictions_traces_dict = {}

        # Variable containing the trace of objective values of solutions generated each step
        self.steps_trace = None

    def compute_descriptors_objective_values(self, smiles_list, objective):
        """
        Computing the descriptors and the values of the objective function for the given smiles list. The smiles causing
        an error are discarded from the resulting dataset. Also returning integer masks identifying the smiles that
        were successfully and unsuccessfully transformed (with the cause).
        :param smiles_list: list of smiles to be transformed
        :param objective : bbo.objective.EvoMolEvaluationStrategyWrapper instance computing the objective values
        :return: (success smiles, success descriptor values, success objective values, success all scores values,
        success objective computation times, success int mask, failed smiles, failed bool mask, failed because of
        descriptor bool mask, failed because of objective bool mask)
        """

        # Computing descriptors for all input SMILES
        tstart_desc = time.time()
        dataset_X_filtered, smiles_filtered, success_desc_bool_mask = compute_descriptors(smiles_list,
                                                                                          descriptor=self.descriptor,
                                                                                          pipeline=self.pipeline,
                                                                                          apply_pipeline=False)
        desc_comput_time = time.time() - tstart_desc

        # Computing objective function values for all input SMILES and filtering to keep only the valid values
        tstart_obj = time.time()
        obj_values, all_scores, success_obj_values_bool_mask, obj_computation_times = objective.fit_transform(
            smiles_list)
        obj_comput_time = time.time() - tstart_obj

        obj_values_filtered = obj_values[success_obj_values_bool_mask]
        all_scores_filtered = np.array(all_scores)[success_obj_values_bool_mask]
        obj_computation_times_filtered = np.array(obj_computation_times)[success_obj_values_bool_mask]

        # Computing boolean and integer masks of SMILES failing the descriptors and failing the objective function
        failed_desc_bool_mask = np.logical_not(success_desc_bool_mask)
        success_desc_int_mask = np.arange(len(smiles_list))[success_desc_bool_mask]
        failed_obj_bool_mask = np.logical_not(success_obj_values_bool_mask)
        success_obj_int_mask = np.arange(len(smiles_list))[success_obj_values_bool_mask]

        # Computing mask of smiles with both success for descriptors and success for objective function values
        success_bool_mask = np.logical_and(success_desc_bool_mask, success_obj_values_bool_mask)
        success_int_mask = np.arange(len(smiles_list))[success_bool_mask]

        # Computing output lists of SMILES, descriptor, objective values and times to compute the objective values
        # for successful data
        success_smiles = np.array(smiles_list)[success_bool_mask]
        success_desc_values = dataset_X_filtered[np.isin(success_desc_int_mask, success_int_mask)]
        success_obj_values = obj_values_filtered[np.isin(success_obj_int_mask, success_int_mask)]
        success_all_scores = all_scores_filtered[np.isin(success_obj_int_mask, success_int_mask)].reshape(
            (len(success_smiles), objective.get_all_scores_number()))
        success_obj_computation_times = obj_computation_times_filtered[np.isin(success_obj_int_mask, success_int_mask)]

        # Insuring scores are float
        success_all_scores = success_all_scores.astype(np.float)
        success_obj_values = success_obj_values.astype(np.float)

        # Computing mask of SMILES with at least one failure
        failed_bool_mask = np.logical_not(success_bool_mask)

        # Computing list of failing SMILES
        failed_smiles = np.array(smiles_list)[failed_bool_mask]

        return success_smiles, success_desc_values, success_obj_values, success_all_scores, \
               success_obj_computation_times, success_int_mask, failed_smiles, failed_bool_mask, failed_desc_bool_mask, \
               failed_obj_bool_mask, desc_comput_time, obj_comput_time

    def initialize_stop_criterion(self):
        """
        Performing the initialization relative to the stop criterion
        :return:
        """

        # Adding file stop criterion
        file_stop_criterion = FileStopCriterion(join(self.results_path, "stop_execution"))
        if isinstance(self.stop_criterion, MultipleStopCriterion):
            self.stop_criterion.set_additional_criterion(file_stop_criterion)
        else:
            self.stop_criterion = MultipleStopCriterion([self.stop_criterion, file_stop_criterion])

        # Setting current instance of BBOAlg
        self.stop_criterion.set_bboalg_instance(self)

    def initialize_dataset_transform_smiles(self, input_smiles, success_dict, failed_dict, objective,
                                            is_test_dataset=False):
        """
        Computing the (success) dataset and failure dataset from the transformation of a list of SMILES to a list
        of descriptors and a list of objective values. Data for successful (resp. failing) transformations is written in
        success_dict (resp. failed_dict). If self.score_assigned_to_failed_solutions, failing solutions are also
        inserted in the success dataset with the given objective value.
        :param input_smiles: list of SMILES to be transformed
        :param success_dict: dict that can contain following keys : smiles, X, obj_value, step, n_obj_calls
        :param failed_dict: dict that can contain following keys : smiles, step, n_obj_calls, failed_descriptors,
        failed_objective
        :param is_test_dataset: whether the given data is the test dataset. If so, failing data won't be inserted in the
        success dataset even if self.score_assigned_to_failed_solutions is not None.
        :param objective: bbo.objective.EvoMolEvaluationStrategyWrapper instance
        :return:
        """

        # Computing descriptors and objective function values of initial dataset
        success_dict["smiles"], success_dict["X"], success_dict["obj_value"], success_all_scores, \
        success_dict["success_obj_computation_time"], success_int_mask, failed_smiles_list, failed_bool_mask, \
        failed_desc_bool_mask, failed_obj_bool_mask, time_desc, time_obj = \
            self.compute_descriptors_objective_values(input_smiles, objective)

        # Recording the other scores values
        for i, key in enumerate(objective.keys()):
            success_dict[key] = success_all_scores.T[i].reshape(-1, )

        # Setting merit value to undefined for initial population solutions
        for i, key in enumerate(self.merit_function.keys() + ["value"]):
            success_dict["merit_" + key] = np.full(len(success_dict["smiles"], ), np.nan)

        # Setting introduction step
        if "step" in success_dict:
            success_dict["step"] = np.zeros((success_dict["smiles"].shape[0],), dtype=np.int)

        # Setting success boolean
        if "success" in success_dict:
            success_dict["success"] = np.full((success_dict["smiles"].shape[0],), True)

        # Setting objective calls
        if "n_obj_calls" in success_dict:
            success_dict["n_obj_calls"] = np.full((success_dict["smiles"].shape[0],), objective.calls_count)

        # Saving failed data SMILES, steps and calls
        failed_dict["smiles"] = failed_smiles_list

        if "step" in failed_dict:
            failed_dict["step"] = np.zeros((failed_dict["smiles"].shape[0]), dtype=np.int)

        if "n_obj_calls" in failed_dict:
            failed_dict["n_obj_calls"] = np.full((failed_dict["smiles"].shape[0],),
                                                 objective.calls_count)

        # Saving the failure cause
        failed_dict["failed_descriptors"] = failed_desc_bool_mask[failed_bool_mask]
        failed_dict["failed_objective"] = failed_obj_bool_mask[failed_bool_mask]

        # Setting merit value to undefined for initial population solutions
        for i, key in enumerate(self.merit_function.keys() + ["value"]):
            failed_dict["merit_" + key] = np.full(len(failed_smiles_list, ), np.nan)

        # If self.score_assigned_to_failed_solutions is not None, also adding the failed solutions in the dataset with
        # the given score value
        if self.score_assigned_to_failed_solutions is not None and not is_test_dataset:

            success_dict["smiles"] = np.concatenate([success_dict["smiles"], failed_smiles_list])
            success_dict["X"] = np.concatenate([success_dict["X"], np.zeros((failed_dict["smiles"].shape[0],
                                                                             success_dict["X"].shape[1]))])

            # Assigning the given value to failed solutions
            success_dict["obj_value"] = np.concatenate([success_dict["obj_value"],
                                                        np.full((failed_dict["smiles"].shape[0],),
                                                                self.score_assigned_to_failed_solutions)])
            # Recording the other scores values
            for i, key in enumerate(objective.keys()):
                success_dict[key] = np.concatenate([success_dict[key],
                                                    np.full((failed_dict["smiles"].shape[0],),
                                                            self.score_assigned_to_failed_solutions)])

            # Setting merit value to undefined for initial population solutions
            for i, key in enumerate(self.merit_function.keys() + "value"):
                success_dict["merit_" + key] = np.concatenate([success_dict[key],
                                                               np.full((failed_dict["smiles"].shape[0],), np.nan)])

            if "step" in success_dict:
                success_dict["step"] = np.concatenate([success_dict["step"],
                                                       np.zeros((failed_dict["smiles"].shape[0],), dtype=np.int)])
            if "success" in success_dict:
                success_dict["success"] = np.concatenate([success_dict["success"],
                                                          np.full((failed_dict["smiles"].shape[0],), False)])
            if "n_obj_calls" in failed_dict:
                success_dict["n_obj_calls"] = np.concatenate(
                    [success_dict["n_obj_calls"], np.full((failed_dict["smiles"].shape[0],),
                                                          objective.calls_count)])

    def initialize_data(self):
        """
        Computing the descriptors and objective function values for initial dataset and test set.
        All smiles leading to an error are not part of the resulting effective dataset.
        Also initializing the data structures necessary to the recording of the results.
        :return:
        """

        print("Initialization of dataset")

        # Disabling the count of calls to the objective function
        self.objective_init_pop.disable_calls_count()

        # Computing descriptors and objective function values for the dataset
        self.initialize_dataset_transform_smiles(self.init_dataset_smiles, self.dataset_dict,
                                                 self.failed_dataset_dict, objective=self.objective_init_pop)

        # Restarting the count of calls to the objective function
        self.objective_init_pop.enable_calls_count()

        print("Initialization of test dataset")

        # Computing descriptors and objective function values of test datasets
        for test_dataset_key in self.test_dataset_smiles_dict.keys():
            self.initialize_dataset_transform_smiles(self.test_dataset_smiles_dict[test_dataset_key],
                                                     self.test_datasets_dicts[test_dataset_key],
                                                     self.failed_test_datasets_dicts[test_dataset_key],
                                                     objective=self.test_objective_dict[test_dataset_key],
                                                     is_test_dataset=True)

        # Test predictions trace
        for test_dataset_key in self.test_dataset_smiles_dict.keys():
            prediction_keys = ["pred_" + str(i) for i in
                               range(len(self.test_datasets_dicts[test_dataset_key][
                                             "smiles"]))] if self.save_pred_test_values else []
            self.test_predictions_traces_dict[test_dataset_key] = {k: [] for k in
                                                                   (["step", "mae", "med_abs_err", "max_abs_err", "r2",
                                                                     "intersect_size"] + prediction_keys
                                                                    )}

        # Steps trace
        self.steps_trace = {k: [] for k in ["step", "obj_mean", "obj_med", "obj_min", "obj_max", "n_generated",
                                            "n_obj_calls", "step_duration", "fit_duration", "test_prediction_duration",
                                            "optim_duration", "desc_obj_comput_duration", "time_desc", "time_obj",
                                            "alg_total_time", "min_desc_row_size"]}

    def save(self, curr_step):
        """
        Saving the results of the execution to self.results_path
        :param curr_step: current step
        :return:
        """

        # Creating results directory if necessary
        makedirs(self.results_path, exist_ok=True)
        makedirs(join(self.results_path, "surrogate"), exist_ok=True)

        """
        Saving dataset.csv and failed_dataset.csv data
        """
        save_dict_to_csv(self.dataset_dict, csv_path=join(self.results_path, "dataset.csv"), ignore_keys=["X"])
        save_dict_to_csv(self.failed_dataset_dict, csv_path=join(self.results_path, "failed_dataset.csv"))
        """
        
        Saving steps.csv data
        Step number, min, max, median, mean objective function values of solutions generated at given step
        """
        save_dict_to_csv(
            self.steps_trace,
            join(self.results_path, "steps.csv")
        )

        """
        Saving test_data.csv and failed_test_data.csv
        Contains the smiles and objective function values of all solutions of the test set. Only saved once at the
        beginning of the algorithm as this data does not change.
        """
        if curr_step == 0:
            for test_dataset_key in self.test_dataset_smiles_dict.keys():
                save_dict_to_csv(self.test_datasets_dicts[test_dataset_key],
                                 join(self.results_path, "surrogate", "test_data_" + test_dataset_key + ".csv"),
                                 ignore_keys=["X"])
                save_dict_to_csv(self.failed_test_datasets_dicts[test_dataset_key],
                                 join(self.results_path, "surrogate", "failed_test_data_" + test_dataset_key + ".csv"))

        """
        Saving test.csv
        Contains some evaluation metrics and the predictions of all solutions of the test set at given step.
        """
        for test_dataset_key in self.test_dataset_smiles_dict.keys():
            save_dict_to_csv(
                self.test_predictions_traces_dict[test_dataset_key],
                join(self.results_path, "surrogate", "test_predictions_" + test_dataset_key + ".csv")
            )

        """
        Saving model.pkl
        Saving all surrogate model data so that it can be reloaded eventually. Only saving if self.save_surrogate_model
        is set to true.
        """
        if self.save_surrogate_model:
            dump(self.surrogate, join(self.results_path, "surrogate", "model.pkl"))

    def compute_test_values(self, curr_step):
        """
        Recording the test predictions and quality metrics at current step
        :return:
        """

        for test_dataset_key in self.test_dataset_smiles_dict.keys():

            # Computing mask of solutions of the test set that are also in the dataset
            mask_intersect = np.in1d(self.test_datasets_dicts[test_dataset_key]["smiles"], self.dataset_dict["smiles"])

            # Applying pipeline
            if self.pipeline is None:
                X_test_transformed = self.test_datasets_dicts[test_dataset_key]["X"]
            else:
                X_test_transformed = self.pipeline.transform(self.test_datasets_dicts[test_dataset_key]["X"])

            # Prediction
            y_test_pred = self.surrogate.predict(X_test_transformed)

            # Setting predictions of known solutions as np.nan
            y_test_pred[mask_intersect] = np.nan

            # Creating arrays of values for non intersecting solutions
            y_true_non_intersect = self.test_datasets_dicts[test_dataset_key]["obj_value"][
                np.logical_not(mask_intersect)]
            y_pred_non_intersect = y_test_pred[np.logical_not(mask_intersect)]

            # Recording step data
            self.test_predictions_traces_dict[test_dataset_key]["step"].append(curr_step)

            # Computing and recording scores data
            abs_errors = np.abs(y_true_non_intersect - y_pred_non_intersect)
            self.test_predictions_traces_dict[test_dataset_key]["mae"].append(np.mean(abs_errors))
            self.test_predictions_traces_dict[test_dataset_key]["med_abs_err"].append(np.median(abs_errors))
            self.test_predictions_traces_dict[test_dataset_key]["max_abs_err"].append(np.max(abs_errors))
            self.test_predictions_traces_dict[test_dataset_key]["r2"].append(
                r2_score(y_true_non_intersect, y_pred_non_intersect))
            self.test_predictions_traces_dict[test_dataset_key]["intersect_size"].append(np.sum(mask_intersect))

            # Writing prediction data
            if self.save_pred_test_values:
                for i in range(len(y_test_pred)):
                    self.test_predictions_traces_dict[test_dataset_key]["pred_" + str(i)].append(y_test_pred[i])

    def update_steps_data(self, curr_step, new_solutions_obj_values, time_step, time_fit, time_optim,
                          time_comput_desc_obj, time_desc, time_obj, time_test_prediction, alg_total_time,
                          min_desc_row_size):
        """
        Updating the self.step_trace data
        :param curr_step: id of current step
        :param new_solutions_obj_values: objective function values of newly generated solutions
        :param time_step: complete duration of the step
        :param time_fit: model fitting duration
        :param time_optim: merit function optimization duration
        :param time_comput_desc_obj: descriptor and objective values computation of newly generated solutions duration
        :param time_desc: time to compute descriptors of inserted solutions
        :param time_obj: time to compute objective values of inserted solutions
        :param time_test_prediction: duration of the test of the surrogate on the test set(s)
        :param alg_total_time: total time since the beginning of the algorithm
        :param min_desc_row_size: minimum possible value of descriptor size
        :return:
        """

        self.steps_trace["step"].append(curr_step)
        self.steps_trace["obj_mean"].append(np.mean(new_solutions_obj_values))
        self.steps_trace["obj_med"].append(np.median(new_solutions_obj_values))
        self.steps_trace["obj_min"].append(np.min(new_solutions_obj_values)
                                           if len(new_solutions_obj_values) > 0 else np.nan)
        self.steps_trace["obj_max"].append(
            np.max(new_solutions_obj_values) if len(new_solutions_obj_values) > 0 else np.nan)
        self.steps_trace["n_generated"].append(len(new_solutions_obj_values))
        self.steps_trace["n_obj_calls"].append(self.objective.calls_count)
        self.steps_trace["step_duration"].append(time_step)
        self.steps_trace["fit_duration"].append(time_fit)
        self.steps_trace["test_prediction_duration"].append(time_test_prediction)
        self.steps_trace["optim_duration"].append(time_optim)
        self.steps_trace["desc_obj_comput_duration"].append(time_comput_desc_obj)
        self.steps_trace["time_desc"].append(time_desc)
        self.steps_trace["time_obj"].append(time_obj)
        self.steps_trace["alg_total_time"].append(alg_total_time)
        self.steps_trace["min_desc_row_size"].append(min_desc_row_size)

    def run(self):
        """
        Main function running the algorithm until the stop criterion has been reached.
        :return:
        """

        alg_start_time = time.time()

        # Computing descriptors and objective values for datasets
        self.initialize_data()

        # Initialization of stop criterion
        self.initialize_stop_criterion()

        # Initialization of steps data
        self.update_steps_data(curr_step=0, new_solutions_obj_values=[], time_step=0, time_fit=0, time_optim=0,
                               time_comput_desc_obj=0, time_desc=0, time_obj=0, time_test_prediction=0,
                               alg_total_time=time.time() - alg_start_time,
                               min_desc_row_size=self.descriptor.min_row_size())

        # Saving results data
        self.save(curr_step=0)

        try:

            # Main loop
            while not self.stop_criterion.time_to_stop(self.results_path):

                saved_last_step = False

                tstart_step = time.time()

                # Updating current step
                self.curr_step += 1

                # Applying pipeline on dataset if defined
                if self.pipeline is not None:
                    X_dataset_transformed = self.pipeline.fit_transform(self.dataset_dict["X"])
                else:
                    X_dataset_transformed = self.dataset_dict["X"]

                # Training surrogate model
                self.surrogate.fit(X_dataset_transformed, self.dataset_dict["obj_value"])

                print("model fitted")

                # Computing time for fitting surrogate model
                time_fit_surrogate = time.time() - tstart_step

                # Notifying the merit function that the surrogate model has been retrained on the given dataset
                self.merit_function.call_method_on_leaves("update_training_dataset", {
                    "dataset_X": self.dataset_dict["X"],
                    "dataset_y": self.dataset_dict["obj_value"],
                    "dataset_smiles": self.dataset_dict["smiles"]
                })

                # Computing test data
                tstart_test_prediction = time.time()

                if self.curr_step == 1 or self.curr_step % self.period_compute_test_predictions == 0:
                    self.compute_test_values(curr_step=self.curr_step)

                time_test_prediction = time.time() - tstart_test_prediction

                tstart_optim = time.time()

                print("Starting EvoMol optimizations")

                # Performing EvoMol optimizations
                evomol_instances = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch,
                                            batch_size=self.batch_size)(
                    delayed(run_evomol_merit_optimization)(
                        merit_function=self.merit_function, results_path=self.results_path,
                        evomol_parameters=self.evomol_parameters,
                        evomol_init_pop_strategy=self.evomol_init_pop_strategy,
                        dataset_smiles=self.dataset_dict["smiles"],
                        dataset_y=self.dataset_dict["obj_value"], evomol_init_pop_size=self.evomol_init_pop_size
                    ) for i in range(self.n_evomol_runs))

                time_optim = time.time() - tstart_optim

                results_smiles = []
                results_merit_scores = []
                results_merit_sub_scores = []

                # Retrieving the best solutions from EvoMol optimizations
                for i in range(self.n_evomol_runs):
                    curr_smiles, curr_scores, curr_sub_scores = evomol_instances[i].get_k_best_individuals_smiles(
                        self.n_best_evomol_retrieved,
                        tabu_list=list(self.dataset_dict["smiles"]) + results_smiles)

                    results_smiles.extend(curr_smiles)
                    results_merit_scores.extend(curr_scores)
                    results_merit_sub_scores.extend(curr_sub_scores)

                # Flattening the list of smiles
                results_smiles = list(np.array(results_smiles).reshape(-1, ))

                print(results_smiles)

                tstart_desc_obj_comput = time.time()

                # Computing descriptors and objective function values of new solutions
                success_smiles, success_X, success_y, success_all_scores, success_obj_computation_times, \
                success_int_mask, failed_smiles_list, failed_bool_mask, failed_desc_bool_mask, failed_obj_bool_mask, \
                time_desc, time_obj = \
                    self.compute_descriptors_objective_values(results_smiles, objective=self.objective)

                time_comput_desc_obj = time.time() - tstart_desc_obj_comput

                # Computing the vectors of calls count for the smiles based on their order. These values are
                # artificial as the objective values of the batch were computed in parallel but this insures that a
                # solution is associated with a value of calls count (and conversely).
                success_obj_calls = self.objective.calls_count - (len(results_smiles) - 1 - success_int_mask)
                failed_obj_calls = self.objective.calls_count - (len(results_smiles) - 1 -
                                                                 np.arange(len(results_smiles))[failed_bool_mask])

                # Updating attributes for successful solutions
                self.dataset_dict["smiles"] = np.concatenate([self.dataset_dict["smiles"], success_smiles])
                self.dataset_dict["success"] = np.concatenate([self.dataset_dict["success"],
                                                               np.full((len(success_smiles),), True)])
                self.dataset_dict["X"] = np.concatenate([self.dataset_dict["X"], success_X])
                self.dataset_dict["obj_value"] = np.concatenate([self.dataset_dict["obj_value"], success_y])
                self.dataset_dict["step"] = np.concatenate(
                    [self.dataset_dict["step"], np.full((len(success_smiles),), self.curr_step)])
                self.dataset_dict["n_obj_calls"] = np.concatenate(
                    [self.dataset_dict["n_obj_calls"], success_obj_calls])
                self.dataset_dict["success_obj_computation_time"] = np.concatenate([
                    self.dataset_dict["success_obj_computation_time"], success_obj_computation_times])

                # Recording the other scores values for successful solutions
                for i, key in enumerate(self.objective.keys()):
                    self.dataset_dict[key] = np.concatenate(
                        [self.dataset_dict[key], success_all_scores.T[i].reshape(-1, )])

                # Recording the merit scores for successful solutions
                self.dataset_dict["merit_value"] = np.concatenate([
                    self.dataset_dict["merit_value"], np.array(results_merit_scores)[success_int_mask]])
                for i, key in enumerate(self.merit_function.keys()):
                    self.dataset_dict["merit_" + key] = np.concatenate([
                        self.dataset_dict["merit_" + key], np.array(results_merit_sub_scores)[success_int_mask].T[i].reshape(-1,)]
                    )

                # Updating attributes for failed solutions
                self.failed_dataset_dict["smiles"] = np.concatenate([self.failed_dataset_dict["smiles"],
                                                                     failed_smiles_list])
                self.failed_dataset_dict["step"] = np.concatenate([self.failed_dataset_dict["step"],
                                                                   np.full((len(failed_smiles_list),), self.curr_step)])
                self.failed_dataset_dict["n_obj_calls"] = np.concatenate([self.failed_dataset_dict["n_obj_calls"],
                                                                          failed_obj_calls])
                self.failed_dataset_dict["failed_descriptors"] = np.concatenate(
                    [self.failed_dataset_dict["failed_descriptors"], failed_desc_bool_mask[failed_bool_mask]])
                self.failed_dataset_dict["failed_objective"] = np.concatenate(
                    [self.failed_dataset_dict["failed_objective"], failed_obj_bool_mask[failed_bool_mask]])

                # Recording the merit scores for failed solutions
                self.failed_dataset_dict["merit_value"] = np.concatenate([
                    self.failed_dataset_dict["merit_value"], np.array(results_merit_scores)[failed_bool_mask]])
                for i, key in enumerate(self.merit_function.keys()):
                    self.failed_dataset_dict["merit_" + key] = np.concatenate([
                        self.failed_dataset_dict["merit_" + key], np.array(results_merit_sub_scores)[failed_bool_mask].T[i].reshape(-1,)
                    ])

                # If self.score_assigned_to_failed_solutions is not None, failing solutions are also inserted in the
                # success dataset with the given objective value
                if self.score_assigned_to_failed_solutions is not None:

                    self.dataset_dict["smiles"] = np.concatenate([self.dataset_dict["smiles"],
                                                                  failed_smiles_list])
                    self.dataset_dict["X"] = np.concatenate(
                        [self.dataset_dict["X"], np.zeros((len(failed_smiles_list),
                                                           self.dataset_dict["X"].shape[1]))])

                    # Assigning the given value to failed solutions
                    self.dataset_dict["obj_value"] = np.concatenate([self.dataset_dict["obj_value"],
                                                                     np.full((len(failed_smiles_list),),
                                                                             self.score_assigned_to_failed_solutions)])
                    # Recording the other scores values
                    for i, key in enumerate(self.objective.keys()):
                        self.dataset_dict[key] = np.concatenate([self.dataset_dict[key],
                                                                 np.full((len(failed_smiles_list),),
                                                                         self.score_assigned_to_failed_solutions)])

                    # Recording the merit scores values
                    self.dataset_dict["merit_value"] = np.concatenate([
                        self.dataset_dict["merit_value"], np.array(results_merit_scores)[failed_bool_mask]])
                    for i, key in enumerate(self.merit_function.keys()):
                        self.dataset_dict["merit_" + key] = np.concatenate([
                            self.dataset_dict["merit_" + key], np.array(results_merit_sub_scores)[failed_bool_mask].T[i].reshape(-1,)
                        ])

                    self.dataset_dict["step"] = np.concatenate([self.dataset_dict["step"],
                                                                np.full((len(failed_smiles_list),), self.curr_step)])

                    self.dataset_dict["success"] = np.concatenate([self.dataset_dict["success"],
                                                                   np.full((len(failed_smiles_list),), False)])

                    self.dataset_dict["n_obj_calls"] = np.concatenate([self.dataset_dict["n_obj_calls"],
                                                                       failed_obj_calls])

                time_step = time.time() - tstart_step

                # Updating steps data
                self.update_steps_data(curr_step=self.curr_step, new_solutions_obj_values=success_y,
                                       time_step=time_step, time_fit=time_fit_surrogate, time_optim=time_optim,
                                       time_comput_desc_obj=time_comput_desc_obj,
                                       time_desc=time_desc,
                                       time_obj=time_obj,
                                       time_test_prediction=time_test_prediction,
                                       alg_total_time=time.time() - alg_start_time,
                                       min_desc_row_size=self.descriptor.min_row_size())

                # Saving results data
                if self.curr_step % self.period_save == 0:
                    saved_last_step = True
                    self.save(curr_step=self.curr_step)

                print("step over")

        except InterruptedError as e:
            print("Stopping : interrupted by user")
        except Exception as e:
            with open(join(self.results_path, "error.txt"), "w") as f:
                f.write(str(e))
            raise e
        finally:

            if not saved_last_step:
                # Saving results data
                self.save(curr_step=self.curr_step)
