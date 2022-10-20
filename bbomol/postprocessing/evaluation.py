import numpy as np


def compute_best_so_far(obj_calls, obj_values, max_obj_calls, problem_type="max"):
    """
    Computing the vector of best solution found indexed on a normalized objective calls vector ranging from 1 to a given
    value (max_obj_calls parameter).
    :param obj_calls: vector of number of calls to the objective function (of same size as the obj_values parameter)
    :param obj_values: vector of objective values (of same size as the obj_calls parameter)
    :param max_obj_calls: last value of the returned normalized objective calls vector (first value is 1)
    :param problem_type: whether the problem is a maximization problem ("max") or minimization problem ("min")
    :return: normalized objective calls vector (from 1 to max_obj_calls), corresponding best found objective value
    """

    curr_best = None

    # Creating the normalized objective calls vector
    norm_obj_calls = np.arange(1, max_obj_calls + 1)

    # Initialization of the best_so_far vector
    best_so_far_vect = np.full(norm_obj_calls.shape, np.nan)

    if problem_type == "max":
        curr_best = float("-inf")
    elif problem_type == "min":
        curr_best = float("inf")

    # Iterating over all possible number of calls to the objective function
    for i, norm_obj_call in enumerate(norm_obj_calls):

        # Checking if the current number of objective calls is in the input vector
        if norm_obj_call in obj_calls:

            # Retrieving the objective value associated with the current number of calls to the objective function
            curr_obj_value = obj_values[np.argwhere(np.array(obj_calls) == norm_obj_call)[0][0]]

            # Checking if the objective value is the new best
            if (curr_obj_value > curr_best and problem_type == "max") or \
                    (curr_obj_value < curr_best and problem_type == "min"):
                curr_best = curr_obj_value

        # Recording the best objective value found at the current number of calls to the objective function
        best_so_far_vect[i] = curr_best

    # Returning the normalized vector of calls to the objective function and the best-so-far vector
    return norm_obj_calls, best_so_far_vect


def compute_best_so_far_matrix(obj_calls_list, obj_values_list, max_obj_calls, problem_type="max"):
    """
    Computing the matrix of best-so-far vectors for the given runs.
    :param obj_calls_list: list of vectors of number of calls to the objective function (matching the sizes of the
    obj_value_list parameter)
    :param obj_values_list: list of vectors of objective values (matching the sizes of the obj_calls_list parameter)
    :param max_obj_calls: last value of the returned normalized objective calls vector (first value is 1)
    :param problem_type: whether the problem is a maximization problem ("max") or minimization problem ("min")
    :return:
    """

    # Computing the number of runs
    n_runs = len(obj_calls_list)

    # Computing the complete vector of calls to the objective function
    complete_obj_calls = np.arange(1, max_obj_calls + 1)

    # Initialization of the best-so-far matrix for all runs
    best_so_far_matrix = np.zeros((n_runs, len(complete_obj_calls)))

    # Iterating over all runs
    for i in range(n_runs):
        # Computing the best-so-far vector for the current run
        _, curr_run_best_so_far_vect = compute_best_so_far(obj_calls_list[i], obj_values_list[i],
                                                           max_obj_calls=max_obj_calls,
                                                           problem_type=problem_type)

        # Setting best-so-far values in complete best-so-far array
        best_so_far_matrix[i] = curr_run_best_so_far_vect

    # Returning the best-so-far matrix
    return complete_obj_calls, best_so_far_matrix


def compute_ecdf(obj_calls_list, obj_values_list, targets, problem_type="max"):
    """
    Computing the ECDF (empirical cumulative distribution function) of the runtimes (=calls to the objective function)
    with multiple targets following the definition of [1]. The definition is extended to be used both on maximization
    and minimization problems (problem_type parameter).
    This functions aggregates input data given as lists of lists describing different runs. For each x value (number of
    runs), this function returns the proportion of satisfied targets (number of input targets * number of runs).

    [1] Nikolaus Hansen et al., “COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,”
    Optimization Methods and Software 36, no.1 (January 2, 2021): 114–44, https://doi.org/10.1080/10556788.2020.1808977.
    :param obj_calls_list: list of vectors of number of calls to the objective function (matching the sizes of the
    obj_value_list parameter)
    :param obj_values_list: list of vectors of objective values (matching the sizes of the obj_calls parameter)
    :param targets: list of targets
    :param problem_type: whether the problem is a maximization problem ("max") or minimization problem ("min")
    :return: x values (number of calls to the objective function), y values (ECDF : proportion of satisfied targets)
    """

    success_targets_matrix = None

    # Computing the number of runs
    n_runs = len(obj_calls_list)

    # Computing the max number of calls to the objective function
    max_obj_calls = np.max([np.max(obj_calls_vect) for obj_calls_vect in obj_calls_list])

    # Computing the complete vector of calls to the objective function
    complete_obj_calls = np.arange(1, max_obj_calls + 1)

    # Initialization of the ECDF vector
    ecdf_vect = np.full(complete_obj_calls.shape, np.nan)

    # Computing the best-so-far matrix
    _, best_so_far_matrix = compute_best_so_far_matrix(obj_calls_list, obj_values_list, max_obj_calls=max_obj_calls,
                                                       problem_type=problem_type)

    # Iterating over all numbers of calls to the objective function
    for i in range(len(complete_obj_calls)):

        # Extracting best-so-far values for all runs at current number of calls to the objective function
        best_so_far_curr_n_calls = best_so_far_matrix.T[i].reshape(-1, 1)

        # Computing the matrix of reached targets
        if problem_type == "max":
            success_targets_matrix = np.greater_equal(best_so_far_curr_n_calls, np.array(targets))
        elif problem_type == "min":
            success_targets_matrix = np.less_equal(best_so_far_curr_n_calls, np.array(targets))

        # Computing the current ECDF value (number of success / total number of targets)
        ecdf_vect[i] = np.sum(success_targets_matrix) / (n_runs * len(targets))

    return complete_obj_calls, ecdf_vect


def compute_timestamps_ecdf(timestamps_list, obj_values_list, targets):
    """
    ECDF computation for timestamps data
    :param timestamps_list:
    :param obj_values_list:
    :param targets:
    :return:
    """

    n_runs = len(timestamps_list)

    timestamps = timestamps_list[0]

    ecdf_vect = np.full((len(timestamps),), np.nan)

    # Iterating over all timestamps
    for i in range(len(timestamps)):

        curr_timestamp = timestamps[i]
        best_so_far_values = []

        # Iterating over all runs to find the best found objective value at given timestamp
        for j in range(n_runs):
            best_so_far_values.append(np.nanmax(obj_values_list[j][timestamps <= curr_timestamp]))

        # Computing boolean matrix of targets achieved for each run
        success_target_matrix = np.greater_equal(np.array(best_so_far_values).reshape(-1, 1), targets)

        # Computing current ECDF value
        ecdf_vect[i] = np.sum(success_target_matrix) / (n_runs * len(targets))

    return timestamps, ecdf_vect


def compute_ERT_timestamps(timestamps_list, obj_values_list, targets, effective_last_timestamp_list):
    n_runs = len(timestamps_list)

    timestamps = timestamps_list[0]
    best_so_far_matrix = np.full((len(timestamps), n_runs), np.nan)

    # Iterating over all timestamps
    for i in range(len(timestamps)):

        curr_timestamp = timestamps[i]
        best_so_far_values = []

        # Iterating over all runs to find the best found objective value at given timestamp
        for j in range(n_runs):
            best_so_far_values.append(np.nanmax(obj_values_list[j][timestamps <= curr_timestamp]))

        best_so_far_matrix[i] = best_so_far_values

    # Initialization of the result vector
    ERT_vect = np.zeros((len(targets),))
    n_success_vect = np.zeros((len(targets),))

    # Iterating over all targets
    for i, target_value in enumerate(targets):

        success_targets_matrix = best_so_far_matrix.T >= target_value

        calls_sum = 0
        n_success = 0
        for j, success_row in enumerate(success_targets_matrix):

            # Testing if the target was reached for the current run
            if np.sum(success_row) > 0:

                # Recording the success
                n_success += 1

                # Adding the number of calls corresponding to the first objective value above the target
                calls_sum += timestamps[np.argmax(success_row)]

            else:

                # If the target was not reached, adding the total number of calls to the objective function during the
                # run
                calls_sum += effective_last_timestamp_list[j]

        # Computing and recording the ERT value for current target
        if n_success > 0:
            ERT_vect[i] = calls_sum / n_success
        else:
            ERT_vect[i] = np.inf

        n_success_vect[i] = n_success

    return ERT_vect, n_success_vect


def compute_ERT(obj_calls_list, obj_values_list, targets, problem_type="max"):
    """
    Computing the ERT (estimated expected run time) values for the given targets.
    The ERT for a target is computed as the sum of all calls to the objective function to reach the target, divided by
    the number of runs that reached the target.
    This functions returns a vector containing the estimated ERT for all given targets.

    see :
    Nikolaus Hansen et al., “COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,”
    Optimization Methods and Software 36, no.1 (January 2, 2021): 114–44, https://doi.org/10.1080/10556788.2020.1808977.
    :param obj_calls_list: list of vectors of number of calls to the objective function (of same sizes as the obj_values
    parameter)
    :param obj_values_list: list of vectors of objective values (of same sizes as the obj_values parameter)
    :param targets: list of target values
    :param problem_type: whether it is a maximization ("max") or minimization ("min") problem
    :return: an array of same size as the target parameter containing the ERT values
    """

    success_targets_matrix = None

    # Initialization of the result vector
    ERT_vect = np.zeros((len(targets),))
    n_success_vect = np.zeros((len(targets),))

    # Computing the max number of calls to the objective function
    max_obj_calls = np.max([np.max(obj_calls_vect) for obj_calls_vect in obj_calls_list])

    # Computing the best-so-far matrix for all runs
    complete_obj_calls, best_so_far_matrix = compute_best_so_far_matrix(obj_calls_list, obj_values_list,
                                                                        max_obj_calls=max_obj_calls,
                                                                        problem_type=problem_type)

    # Iterating over all targets
    for i, target_value in enumerate(targets):

        # Computing the matrix of reached targets
        if problem_type == "max":
            success_targets_matrix = best_so_far_matrix >= target_value
        elif problem_type == "min":
            success_targets_matrix = best_so_far_matrix <= target_value

        calls_sum = 0
        n_success = 0
        for j, success_row in enumerate(success_targets_matrix):

            # Testing if the target was reached for the current run
            if np.sum(success_row) > 0:

                # Recording the success
                n_success += 1

                # Adding the number of calls corresponding to the first objective value above the target
                calls_sum += complete_obj_calls[np.argmax(success_row)]

            else:

                # If the target was not reached, adding the total number of calls to the objective function during the
                # run
                calls_sum += obj_calls_list[j][-1]

        # Computing and recording the ERT value for current target
        if n_success > 0:
            ERT_vect[i] = calls_sum / n_success
        else:
            ERT_vect[i] = np.inf

        n_success_vect[i] = n_success

    return ERT_vect, n_success_vect
