import json
import os
import time
from os.path import join

import cclib
from evomol.evomol.evaluation_dft import smi_to_filename, rdkit_mmff94_xyz, load_obabel_smi, remove_files, \
    write_input_file
from joblib import Memory, Parallel, delayed
from bbo.objective import Objective
import numpy as np

# TODO delete this file

def _compute_dft_energy_values(working_dir_path, smi):
    """
    Computing HOMO/LUMO using Gaussian09 and a B3LYP/3-21G* optimization
    :param working_dir_path: path in which temporary data will be written
    :param smi: smiles to be computed
    :return: (homo value, lumo value, whether the optimization is a success, short message)
    """

    print("computing dft for " + str(smi))

    homo = None
    lumo = None
    success = False

    # Converting the smiles to file name compatible
    filename = smi_to_filename(smi)

    # Computing paths
    post_MM_smi_path = join(working_dir_path, filename + ".MM.smi")
    post_opt_smi_path = join(working_dir_path, filename + ".opt.smi")
    xyz_path = join(working_dir_path, filename + ".xyz")
    opt_input_path = join(working_dir_path, filename + "_OPT.inp")
    opt_log_path = join(working_dir_path, filename + "_OPT.log")

    # Converting SMILES to XYZ after computing MM (RDKit MMFF94)
    xyz_str, success_MM = rdkit_mmff94_xyz(smi, max_iterations=500)

    if success_MM:

        # Writing optimized XYZ to file
        with open(xyz_path, "w") as f:
            f.writelines(xyz_str)

        # Converting XYZ to smi
        command_obabel = join(os.getenv("OPT_LIBS"), "obabel/openbabel-2.4.1/bin/obabel") + " -ixyz " + xyz_path \
                         + " -osmi -O " + post_MM_smi_path
        os.system(command_obabel + " > /dev/null 2>&1")

        try:
            post_MM_smi = load_obabel_smi(post_MM_smi_path)
            smiles_read_correctly = True
        except Exception as e:
            smiles_read_correctly = False

        if smiles_read_correctly:

            if post_MM_smi == smi:

                # Creating input file for OPT
                write_input_file(opt_input_path, xyz_path, smi, 2)

                # Calculate OPT in the working directory
                command_opt = "cd " + working_dir_path + "; " + join(os.environ["OPT_LIBS"],
                                                                     "dft.sh") + " " + opt_input_path
                print("Starting OPT")
                start = time.time()
                os.system(command_opt + " > /dev/null 2>&1")
                stop = time.time()
                print("Execution time OPT: " + repr(int(stop - start)) + "s")

                # Checking that normal termination occurred
                with open(opt_log_path, "r") as log:
                    last_line = log.readlines()[-1]

                # if the OTP end up well
                if "Normal termination" in last_line:

                    # Extracting the smiles from the log file
                    command_obabel = join(os.getenv("OPT_LIBS"),
                                          "obabel/openbabel-2.4.1/bin/obabel") + " -ilog " + opt_log_path \
                                     + " -ocan -O " + post_opt_smi_path
                    os.system(command_obabel + " > /dev/null 2>&1")

                    try:

                        post_opt_smi_rdkit = load_obabel_smi(post_opt_smi_path)
                        smiles_read_correctly = True

                    except Exception as e:
                        smiles_read_correctly = False

                    if smiles_read_correctly:

                        # If before/after SMILES are identical
                        if smi == post_opt_smi_rdkit:

                            with open(opt_log_path, "r") as log:
                                data = cclib.io.ccread(log, optdone_as_list=True)
                                print("There are %i atoms and %i MOs" % (data.natom, data.nmo))
                                homos = data.homos
                                energies = data.moenergies

                            if len(homos) == 1:

                                homo = energies[0][homos[0]]
                                lumo = energies[0][homos[0] + 1]
                                success = True
                                msg = "Success"

                            else:
                                msg = "DFT error : |homos| > 1"
                        else:
                            msg = "DFT error : Different SMILES"
                    else:
                        msg = "Cannot build molecule from SMILES after DFT"
                else:
                    msg = "DFT error : Error during OPT"
            else:
                msg = "MM error : Different SMILES"
        else:
            msg = "Cannot build molecule from SMILES after MM"
    else:
        msg = "MM error"

    # Removing files
    remove_files([post_MM_smi_path, post_opt_smi_path, xyz_path, opt_input_path])

    # Returning values
    return homo, lumo, success, msg


class DFTEnergyObjective(Objective):

    def __init__(self, property, cache_location, json_cache_location_list=None, n_jobs=1, working_dir="/tmp"):
        """
        :param property: Key of the property to be evaluated ("homo", "lumo")
        :param cache_location: location of the joblib.Memory cache
        :param json_cache_location_list: list of json cache files
        :param n_jobs: number of jobs for parallel computation
        :param working_dir: path in which temporary files will be written
        """

        super().__init__(cache_location)
        self.property = property
        self.n_jobs = n_jobs
        self.json_cache_location_list = json_cache_location_list
        self.working_dir = working_dir

        self.cache_dft_fun = Memory(location=cache_location,
                                    verbose=0).cache(_compute_dft_energy_values, ignore=["working_dir_path"])

        self.cache_json_dict = {}
        if json_cache_location_list is not None:
            # Reversing the list so that values of first files are taken primarily if there exists an intersection of
            # SMILES keys
            json_cache_location_list.reverse()
            for json_cache_location in json_cache_location_list:
                with open(json_cache_location, "r") as f:
                    self.cache_json_dict.update(json.load(f))

    def transform(self, X):
        """
        Overriding Objective.transform method so that only the DFT calculation and the joblib.Memory access is
        parallelized. The access to the JSON cache is done with a single worker for reasons of performance.
        :param X:
        :return:
        """

        # Computing the mask of results in JSON cache
        json_cache_mask = np.full((len(X),), False)
        for i, smiles in enumerate(X):
            json_cache_mask[i] = smiles in self.cache_json_dict

        # Extracting the values from json cache sequentially
        values_json = []
        successes_json = []
        for smiles in np.array(X)[json_cache_mask]:
            value, success = self.transform_row(smiles)
            values_json.append(value)
            successes_json.append(success)

        # Performing a parallel computation of results that are not in JSON cache
        results_parallel = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.transform_row)(smiles) for smiles in np.array(X)[np.logical_not(json_cache_mask)])

        # Retrieving all parallel results
        values_parallel = []
        successes_parallel = []
        for value, success in results_parallel:
            values_parallel.append(value)
            successes_parallel.append(success)

        # Aggregating results
        values = np.zeros((len(X),))
        successes = np.full((len(X),), False)
        values[json_cache_mask] = np.array(values_json)
        successes[json_cache_mask] = np.array(successes_json)
        values[np.logical_not(json_cache_mask)] = np.array(values_parallel)
        successes[np.logical_not(json_cache_mask)] = np.array(successes_parallel)

        # Reshaping arrays
        values = np.array(values).reshape(-1, )
        successes = np.array(successes).reshape(-1, )

        # Updating the calls count
        if self.do_count_calls:
            self.calls_count += len(X)

        return values, successes

    def transform_row(self, smiles):
        super().transform_row(smiles)

        # If the value is in the JSON cache
        if self.cache_json_dict is not None and smiles in self.cache_json_dict:

            entry = self.cache_json_dict[smiles]

            # Computing the success value if the "success" key in in the entry
            if "success" in entry:
                success = entry["success"]
            # Otherwise the success is computed as whether the property is not None
            else:
                success = entry[self.property] is not None

            return entry[self.property], success

        # If the value is in the joblib.Memory cache or needs to be computed
        else:

            try:
                homo, lumo, success, msg = self.cache_dft_fun(self.working_dir, smiles)
            except Exception as e:
                homo = None
                lumo = None
                success = False
                msg = "DFT caused exception " + str(e)

            print("DFT results : " + str(smiles))
            print(msg)
            print(homo)

            if self.property == "homo":
                return homo, success
            elif self.property == "lumo":
                return lumo, success