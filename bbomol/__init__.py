import csv

import evomol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from bbomol.bboalg import BBOAlg
from chemdesc import MBTRDesc, ShinglesVectDesc, SOAPDesc, RandomGaussianVectorDesc
from bbomol.merit import ExpectedImprovementMerit, SurrogateValueMerit, ProbabilityOfImprovementMerit
from bbomol.model import GPRSurrogateModelWrapper
from bbomol.objective import EvoMolEvaluationStrategyWrapper
from bbomol.stop_criterion import KObjFunCallsFunctionStopCriterion


def _extract_explicit_IO_parameters(parameters_dict):
    input_io_parameters = parameters_dict["io_parameters"] if "io_parameters" in parameters_dict else {}
    explicit_io_parameters = {
        "smiles_list_init": input_io_parameters["smiles_list_init"] if "smiles_list_init" in input_io_parameters else [
            "C"],
        "results_path": input_io_parameters[
            "results_path"] if "results_path" in input_io_parameters else "BBOMol_optim/",
        "save_surrogate_model": input_io_parameters[
            "save_surrogate_model"] if "save_surrogate_model" in input_io_parameters else False,
        "save_n_steps": input_io_parameters["save_n_steps"] if "save_n_steps" in input_io_parameters else 1,
        "dft_working_dir": input_io_parameters[
            "dft_working_dir"] if "dft_working_dir" in input_io_parameters else "/tmp",
        "dft_base": input_io_parameters["dft_base"] if "dft_base" in input_io_parameters else "3-21G*",
        "dft_method": input_io_parameters["dft_method"] if "dft_method" in input_io_parameters else "B3LYP",
        "dft_cache_files": input_io_parameters["dft_cache_files"] if "dft_cache_files" in input_io_parameters else [],
        "dft_n_jobs": input_io_parameters["dft_n_jobs"] if "dft_n_jobs" in input_io_parameters else 1,
        "dft_mem_mb": input_io_parameters["dft_mem_mb"] if "dft_mem_mb" in input_io_parameters else 512,
        "MM_program": input_io_parameters["MM_program"] if "MM_program" in input_io_parameters else "rdkit_mmff94"
    }

    for parameter in input_io_parameters:
        if parameter not in explicit_io_parameters:
            raise RuntimeWarning("Unrecognized parameter in 'io_parameters' : " + str(parameter))

    return explicit_io_parameters


def _extract_explicit_merit_optim_parameters(parameters_dict):
    input_merit_optim_parameters = parameters_dict[
        "merit_optim_parameters"] if "merit_optim_parameters" in parameters_dict else {}

    explicit_merit_optim_parameters = {
        "merit_type": input_merit_optim_parameters[
            "merit_type"] if "merit_type" in input_merit_optim_parameters else "EI",
        # For "merit_xi" parameter, accepting both "merit_EI_xi" and "merit_xi" keys for compatibility reasons.
        "merit_xi": input_merit_optim_parameters["merit_xi"] if "merit_xi" in input_merit_optim_parameters else input_merit_optim_parameters["merit_EI_xi"] if "merit_EI_xi" in input_merit_optim_parameters else 0.01,
        "evomol_parameters": input_merit_optim_parameters[
            "evomol_parameters"] if "evomol_parameters" in input_merit_optim_parameters else {
            "optimization_parameters": {
                "max_steps": 10,
            },
            "action_space_parameters": {
                "max_heavy_atoms": 9,
                "atoms": "C,N,O,F"
            }
        },
        "init_pop_size": input_merit_optim_parameters[
            "init_pop_size"] if "init_pop_size" in input_merit_optim_parameters else 10,
        "n_merit_optim_restarts": input_merit_optim_parameters[
            "n_merit_optim_restarts"] if "n_merit_optim_restarts" in input_merit_optim_parameters else 10,
        "n_best_retrieved": input_merit_optim_parameters[
            "n_best_retrieved"] if "n_best_retrieved" in input_merit_optim_parameters else 1,
        "init_pop_strategy": input_merit_optim_parameters[
            "init_pop_strategy"] if "init_pop_strategy" in input_merit_optim_parameters else "random_weighted",
        "noise_based": input_merit_optim_parameters[
            "noise_based"] if "noise_based" in input_merit_optim_parameters else False,
        "init_pop_zero_EI": input_merit_optim_parameters[
            "init_pop_zero_EI"] if "init_pop_zero_EI" in input_merit_optim_parameters else True
    }

    for parameter in input_merit_optim_parameters:
        if parameter not in explicit_merit_optim_parameters:
            raise RuntimeWarning("Unrecognized parameter in 'merit_optim_parameters' : " + str(parameter))

    return explicit_merit_optim_parameters


def _extract_explicit_obj_function(parameters_dict):
    return parameters_dict["obj_function"] if "obj_function" in parameters_dict else "homo"


def _extract_explicit_BBO_optim_parameters(parameters_dict):
    input_BBO_optim_parameters = parameters_dict[
        "bbo_optim_parameters"] if "bbo_optim_parameters" in parameters_dict else {}

    explicit_BBO_optim_parameters = {
        "max_obj_calls": input_BBO_optim_parameters[
            "max_obj_calls"] if "max_obj_calls" in input_BBO_optim_parameters else 1000,
        "failed_solutions_score": input_BBO_optim_parameters[
            "failed_solutions_score"] if "failed_solutions_score" in input_BBO_optim_parameters else None
    }

    for parameter in input_BBO_optim_parameters:
        if parameter not in explicit_BBO_optim_parameters:
            raise RuntimeWarning("Unrecognized parameter in 'bbo_optim_parameters' : " + str(parameter))

    return explicit_BBO_optim_parameters


def _extract_explicit_descriptor_parameters(descriptor_parameters_dict):
    explicit_descriptor_parameters = {
        "type": descriptor_parameters_dict["type"] if "type" in descriptor_parameters_dict else "MBTR",
        "species": descriptor_parameters_dict["species"] if "species" in descriptor_parameters_dict else ["H", "C", "O",
                                                                                                          "N", "F"],
        "n_atoms_max": descriptor_parameters_dict[
            "n_atoms_max"] if "n_atoms_max" in descriptor_parameters_dict else 100,
        "rcut": descriptor_parameters_dict["rcut"] if "rcut" in descriptor_parameters_dict else 6.0,
        "nmax": descriptor_parameters_dict["nmax"] if "nmax" in descriptor_parameters_dict else 8,
        "lmax": descriptor_parameters_dict["lmax"] if "lmax" in descriptor_parameters_dict else 6,
        "average": descriptor_parameters_dict["average"] if "average" in descriptor_parameters_dict else "inner",
        "atomic_numbers_n": descriptor_parameters_dict[
            "atomic_numbers_n"] if "atomic_numbers_n" in descriptor_parameters_dict else 10,
        "inverse_distances_n": descriptor_parameters_dict[
            "inverse_distances_n"] if "inverse_distances_n" in descriptor_parameters_dict else 25,
        "cosine_angles_n": descriptor_parameters_dict[
            "cosine_angles_n"] if "cosine_angles_n" in descriptor_parameters_dict else 25,
        "lvl": descriptor_parameters_dict["lvl"] if "lvl" in descriptor_parameters_dict else 1,
        "vect_size": descriptor_parameters_dict["vect_size"] if "vect_size" in descriptor_parameters_dict else 2000,
        "count": descriptor_parameters_dict["count"] if "count" in descriptor_parameters_dict else True,
        "mu": descriptor_parameters_dict["mu"] if "mu" in descriptor_parameters_dict else 0,
        "sigma": descriptor_parameters_dict["sigma"] if "sigma" in descriptor_parameters_dict else 1,
    }

    for parameter in descriptor_parameters_dict:
        if parameter not in explicit_descriptor_parameters:
            raise RuntimeWarning(
                "Unrecognized parameter in 'surrogate_parameters['descriptor']' : " + str(parameter))

    return explicit_descriptor_parameters


def _extract_explicit_surrogate_parameters(parameters_dict):
    input_surrogate_parameters = parameters_dict[
        "surrogate_parameters"] if "surrogate_parameters" in parameters_dict else {}

    explicit_surrogate_parameters = {
        "max_train_size": input_surrogate_parameters["max_train_size"] if "max_train_size" in input_surrogate_parameters else float("inf"),
        "GPR_instance": input_surrogate_parameters[
            "GPR_instance"] if "GPR_instance" in input_surrogate_parameters else GaussianProcessRegressor(
            1.0 * RBF(1.0) + WhiteKernel(1.0), normalize_y=True),
        "descriptor": _extract_explicit_descriptor_parameters(input_surrogate_parameters["descriptor"]) if \
            "descriptor" in input_surrogate_parameters else _extract_explicit_descriptor_parameters({})
    }

    for parameter in input_surrogate_parameters:
        if parameter not in explicit_surrogate_parameters:
            raise RuntimeWarning("Unrecognized parameter in 'surrogate_parameters' : " + str(parameter))

    return explicit_surrogate_parameters


def _extract_parallelization_parameters(parameters_dict):
    input_parallelization_parameters = parameters_dict[
        "parallelization"] if "parallelization" in parameters_dict else {}

    explicit_parallelization_parameters = {
        "n_jobs_merit_optim_restarts": input_parallelization_parameters[
            "n_jobs_merit_optim_restarts"] if "n_jobs_merit_optim_restarts" in input_parallelization_parameters else 1,
        "n_jobs_desc_comput": input_parallelization_parameters[
            "n_jobs_desc_comput"] if "n_jobs_desc_comput" in input_parallelization_parameters else 1,
        "n_jobs_obj_comput": input_parallelization_parameters[
            "n_jobs_obj_comput"] if "n_jobs_obj_comput" in input_parallelization_parameters else 1,
    }

    for parameter in input_parallelization_parameters:
        if parameter not in explicit_parallelization_parameters:
            raise RuntimeWarning("Unrecognized parameter in 'parallelization' : " + str(parameter))

    return explicit_parallelization_parameters


def _parse_objective_function(obj_fun_explicit, IO_explicit_parameters, merit_optim_explicit_parameters,
                              parallelization_explicit_parameters):
    """
    Converting the input objective function to a functioning evomol.evaluation.EvaluationStrategyComposant instance.
    Using the parser defined in evomol.__init__.py file.
    Then, wrapping it using bbomol.objective.EvomolEvaluationStrategyWrapper
    :param obj_fun_explicit:
    :param IO_explicit_parameters:
    :param merit_optim_explicit_parameters:
    :return: evomol.evaluation.EvaluationStrategyComposant instance
    """

    eval_strategy = evomol._parse_objective_function_strategy(
        parameters_dict={"obj_function": obj_fun_explicit},
        explicit_IO_parameters_dict={
            "dft_working_dir": IO_explicit_parameters["dft_working_dir"],
            "dft_cache_files": IO_explicit_parameters["dft_cache_files"],
            "dft_MM_program": IO_explicit_parameters["MM_program"],
            "dft_base": IO_explicit_parameters["dft_base"],
            "dft_method": IO_explicit_parameters["dft_method"],
            "dft_n_jobs": IO_explicit_parameters["dft_n_jobs"],
            "dft_mem_mb": IO_explicit_parameters["dft_mem_mb"]
        },
        # explicit_search_parameters_dict=evomol._extract_explicit_search_parameters(
        #     merit_optim_explicit_parameters["evomol_parameters"])
        explicit_search_parameters_dict={}  # Only used for entropy optimization. For now not supported with BBOMol.
    )

    return EvoMolEvaluationStrategyWrapper(eval_strategy,
                                           n_jobs=parallelization_explicit_parameters["n_jobs_obj_comput"])


def _parse_surrogate(surrogate_explicit_parameters):
    """
    Building a bbomol.model.GPRSurrogateModelWrapper instance
    :param surrogate_explicit_parameters:
    :return: bbomol.model.GPRSurrogateModelWrapper instance
    """

    return GPRSurrogateModelWrapper(surrogate_explicit_parameters["GPR_instance"])


def _parse_descriptor(surrogate_explicit_parameters, parallelization_explicit_parameters,
                      IO_explicit_parameters):
    """
    Building a bbomol.descriptor.Descriptor instance
    :param surrogate_explicit_parameters:
    :return: bbomol.descriptor.Descriptor instance
    """

    if surrogate_explicit_parameters["descriptor"]["type"] == "MBTR":

        desc = MBTRDesc(species=surrogate_explicit_parameters["descriptor"]["species"],
                        n_jobs=parallelization_explicit_parameters["n_jobs_desc_comput"],
                        atomic_numbers_n=surrogate_explicit_parameters["descriptor"]["atomic_numbers_n"],
                        inverse_distances_n=surrogate_explicit_parameters["descriptor"]["inverse_distances_n"],
                        cosine_angles_n=surrogate_explicit_parameters["descriptor"]["cosine_angles_n"],
                        MM_program=IO_explicit_parameters["MM_program"])

    elif surrogate_explicit_parameters["descriptor"]["type"] == "shingles":

        desc = ShinglesVectDesc(lvl=surrogate_explicit_parameters["descriptor"]["lvl"],
                                vect_size=surrogate_explicit_parameters["descriptor"]["vect_size"],
                                count=surrogate_explicit_parameters["descriptor"]["count"])

    elif surrogate_explicit_parameters["descriptor"]["type"] == "SOAP":

        desc = SOAPDesc(rcut=surrogate_explicit_parameters["descriptor"]["rcut"],
                        nmax=surrogate_explicit_parameters["descriptor"]["nmax"],
                        lmax=surrogate_explicit_parameters["descriptor"]["lmax"],
                        species=surrogate_explicit_parameters["descriptor"]["species"],
                        average=surrogate_explicit_parameters["descriptor"]["average"],
                        n_jobs=parallelization_explicit_parameters["n_jobs_desc_comput"],
                        MM_program=IO_explicit_parameters["MM_program"])

    elif surrogate_explicit_parameters["descriptor"]["type"] == "random":

        desc = RandomGaussianVectorDesc(mu=surrogate_explicit_parameters["descriptor"]["mu"],
                                        sigma=surrogate_explicit_parameters["descriptor"]["sigma"],
                                        vect_size=surrogate_explicit_parameters["descriptor"]["vect_size"])

    return desc


def _parse_merit_function(merit_optim_explicit_parameters, descriptor, surrogate):
    """
    Building bbomol.merit.Merit instance
    :param merit_optim_explicit_parameters:
    :param descriptor:
    :param surrogate:
    :return: bbomol.merit.Merit instance
    """
    if merit_optim_explicit_parameters["merit_type"] == "EI":
        return ExpectedImprovementMerit(descriptor=descriptor, surrogate=surrogate, pipeline=None,
                                        xi=merit_optim_explicit_parameters["merit_xi"],
                                        noise_based=merit_optim_explicit_parameters["noise_based"],
                                        init_pop_zero_EI=merit_optim_explicit_parameters["init_pop_zero_EI"])
    elif merit_optim_explicit_parameters["merit_type"] == "POI":
        return ProbabilityOfImprovementMerit(descriptor=descriptor, surrogate=surrogate, pipeline=None,
                                             xi=merit_optim_explicit_parameters["merit_xi"],
                                             noise_based=merit_optim_explicit_parameters["noise_based"],
                                             init_pop_zero_EI=merit_optim_explicit_parameters["init_pop_zero_EI"])
    elif merit_optim_explicit_parameters["merit_type"] == "surrogate":
        return SurrogateValueMerit(descriptor=descriptor, pipeline=None, surrogate=surrogate)


def _create_BBOAlg_instance(IO_explicit_parameters, BBO_optim_explicit_parameters, merit_optim_explicit_parameters,
                            parallelization_explicit_parameters, surrogate_explicit_parameters, desc, objective,
                            merit_function, surrogate):
    """
    Creating bbomol.bboalg.BBOAlg instance matching all given parameters
    :param IO_explicit_parameters:
    :param BBO_optim_explicit_parameters:
    :param merit_optim_explicit_parameters:
    :param parallelization_explicit_parameters:
    :param desc:
    :param objective:
    :param merit_function:
    :param surrogate:
    :return: bbomol.bboalg.BBOAlg instance
    """

    # Extracting the smiles if IO_explicit_parameters["smiles_list_init"] rather than a list
    if isinstance(IO_explicit_parameters["smiles_list_init"], str):
        smiles_list = []
        with open(IO_explicit_parameters["smiles_list_init"], "r") as f:
            reader = csv.reader(f)
            for row in reader:
                smiles_list.append(row[0])
    else:
        smiles_list = IO_explicit_parameters["smiles_list_init"]

    return BBOAlg(
        init_dataset_smiles=smiles_list,
        descriptor=desc,
        objective=objective,
        merit_function=merit_function,
        surrogate=surrogate,
        stop_criterion=KObjFunCallsFunctionStopCriterion(BBO_optim_explicit_parameters["max_obj_calls"]),
        evomol_parameters=merit_optim_explicit_parameters["evomol_parameters"],
        evomol_init_pop_size=merit_optim_explicit_parameters["init_pop_size"],
        n_evomol_runs=merit_optim_explicit_parameters["n_merit_optim_restarts"],
        n_best_evomol_retrieved=merit_optim_explicit_parameters["n_best_retrieved"],
        evomol_init_pop_strategy=merit_optim_explicit_parameters["init_pop_strategy"],
        score_assigned_to_failed_solutions=BBO_optim_explicit_parameters["failed_solutions_score"],
        results_path=IO_explicit_parameters["results_path"],
        n_jobs_merit_optim=parallelization_explicit_parameters["n_jobs_merit_optim_restarts"],
        save_surrogate_model=IO_explicit_parameters["save_surrogate_model"],
        period_save=IO_explicit_parameters["save_n_steps"],
        max_train_size=surrogate_explicit_parameters["max_train_size"]
    )


def run_optimization(parameters_dict):
    """
    Running the BBO optimization described by the given dictionary of parameters
    :param parameters_dict: dictionary of parameters
    :return: instance of bbomol.bboalg.BBOAlg after optimization
    """

    # Extracting the objective function
    obj_fun_explicit = _extract_explicit_obj_function(parameters_dict)

    # Extracting merit optimization parameters
    merit_optim_explicit_parameters = _extract_explicit_merit_optim_parameters(parameters_dict)

    # Extracting BBO optimization parameters
    BBO_optim_explicit_parameters = _extract_explicit_BBO_optim_parameters(parameters_dict)

    # Extracting IO parameters
    IO_explicit_parameters = _extract_explicit_IO_parameters(parameters_dict)

    # Extracting surrogate parameters
    surrogate_explicit_parameters = _extract_explicit_surrogate_parameters(parameters_dict)

    # Parallelization parameters
    parallelization_explicit_parameters = _extract_parallelization_parameters(parameters_dict)

    # Parsing objective function
    obj_function = _parse_objective_function(obj_fun_explicit, IO_explicit_parameters, merit_optim_explicit_parameters,
                                             parallelization_explicit_parameters)

    # Parsing surrogate model
    surrogate = _parse_surrogate(surrogate_explicit_parameters)

    # Parsing descriptor
    desc = _parse_descriptor(surrogate_explicit_parameters, parallelization_explicit_parameters, IO_explicit_parameters)

    # Parsing merit function
    merit_function = _parse_merit_function(merit_optim_explicit_parameters, desc, surrogate)

    # Creating BBOAlg instance
    bboalg_instance = _create_BBOAlg_instance(IO_explicit_parameters, BBO_optim_explicit_parameters,
                                              merit_optim_explicit_parameters, parallelization_explicit_parameters,
                                              surrogate_explicit_parameters, desc, obj_function, merit_function,
                                              surrogate)

    # Running algorithm
    bboalg_instance.run()

    return bboalg
