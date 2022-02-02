#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Paths
results_path = "output/02_BBO_MBTR_RBF"
dft_path = os.path.abspath("output/dft_files")

# BBO parameters
period_save = 1
period_compute_test_predictions = 1
max_obj_calls = 1000

# EA parameters
evomol_pop_max_size = 300
evomol_max_steps = 10
evomol_k_to_replace = 10
evomol_init_pop_size = 10
evomol_n_runs = 10
evomol_n_best_retrieved = 1
evomol_init_pop_strategy = "random_weighted"

# Chemical space
max_heavy_atoms = 9
heavy_atoms = "C,N,O,F"

# GPR parameters
gpr_alpha = 1e-1
gpr_optimizer = 'fmin_l_bfgs_b'

# Initial dataset initialization
init_dataset_smiles = ["C"]

# Merit parameters
EI_xi = 0.01

# QM objective and MM optimization
prop = "homo"
MM_program = "rdkit"

# Parallelization (None)
n_jobs_dft = 1
n_jobs_per_model = 1
dft_n_threads = 1

# In[2]:


from sklearn.gaussian_process.kernels import RBF

# Kernel
kernel = 1.0 * RBF(1.0)

# In[3]:


from chemdesc import MBTRDesc

# Descriptor
descriptor = MBTRDesc(cache_location=None, n_jobs=n_jobs_per_model, cosine_angles_n=25,
                      atomic_numbers_n=10, inverse_distances_n=25, species=["C", "H", "O", "N", "F"],
                      MM_program=MM_program)

# In[4]:


from sklearn.gaussian_process import GaussianProcessRegressor

# GPR model
surrogate = GaussianProcessRegressor(kernel, optimizer=gpr_optimizer, alpha=gpr_alpha)

# In[5]:


from bbomol.merit import ExpectedImprovementMerit

# Merit function
merit = ExpectedImprovementMerit(descriptor=descriptor, surrogate=surrogate, xi=EI_xi, pipeline=None)

# In[6]:


from evomol.evaluation_dft import OPTEvaluationStrategy
from bbomol.objective import EvoMolEvaluationStrategyWrapper

# Objective function
objective = EvoMolEvaluationStrategyWrapper(
    OPTEvaluationStrategy(
        prop=prop,
        n_jobs=dft_n_threads,
        working_dir_path=dft_path,
        MM_program=MM_program,
    ),
    n_jobs=n_jobs_dft
)

# In[7]:


from os.path import join
from bbomol.bboalg import BBOAlg
from bbomol.stop_criterion import KObjFunCallsFunctionStopCriterion


def run(run_id):
    model_path = join(results_path, str(run_id))

    alg = BBOAlg(
        init_dataset_smiles=init_dataset_smiles,
        descriptor=descriptor,
        objective=objective,
        merit_function=merit,
        surrogate=surrogate,
        stop_criterion=KObjFunCallsFunctionStopCriterion(max_obj_calls),
        evomol_parameters={
            "optimization_parameters": {
                "pop_max_size": evomol_pop_max_size,
                "max_steps": evomol_max_steps,
                "k_to_replace": evomol_k_to_replace
            },
            "action_space_parameters": {
                "max_heavy_atoms": max_heavy_atoms,
                "atoms": heavy_atoms
            }
        },
        evomol_init_pop_size=evomol_init_pop_size,
        n_evomol_runs=evomol_n_runs,
        n_best_evomol_retrieved=evomol_n_best_retrieved,
        evomol_init_pop_strategy=evomol_init_pop_strategy,
        results_path=model_path,
        n_jobs_merit_optim=n_jobs_per_model,
        period_save=period_save,
        period_compute_test_predictions=period_compute_test_predictions

    )

    alg.run()


# In[ ]:


# Running a single execution

run(0)

# In[ ]:


# Running 10 executions

# for i in range(10):
#     run(i)
