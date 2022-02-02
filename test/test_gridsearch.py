from evomol import OPTEvaluationStrategy, ProductEvaluationStrategy, IsomerGuacaMolEvaluationStrategy, \
    SigmLinWrapperEvaluationStrategy, SAScoreEvaluationStrategy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV

from bbomol.bboalg import BBOAlg
from chemdesc import MBTRDesc
from bbomol.merit import ExpectedImprovementMerit
from bbomol.model import GPRSurrogateModelWrapper
from bbomol.objective import EvoMolEvaluationStrategyWrapper
from bbomol.stop_criterion import KObjFunCallsFunctionStopCriterion

formula = "C4N3"

obj = EvoMolEvaluationStrategyWrapper(OPTEvaluationStrategy("homo", MM_program="rdkit", n_jobs=1, working_dir_path="/tmp"), n_jobs=10)


model = GaussianProcessRegressor(1.0*RBF(1.0), optimizer="fmin_l_bfgs_b")
model = GridSearchCV(model, param_grid={"alpha": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]}, cv=5)


descriptor = MBTRDesc(cache_location=None, n_jobs=10, cosine_angles_n=25,
                      atomic_numbers_n=10, inverse_distances_n=25,
                      species=["C", "H", "O", "N", "F"], MM_program="rdkit")

surrogate = GPRSurrogateModelWrapper(model)

merit = ExpectedImprovementMerit(descriptor, None, surrogate)

alg = BBOAlg(
    init_dataset_smiles=["C"],
    descriptor=descriptor,
    objective=obj,
    merit_function=merit,
    surrogate=surrogate,
    stop_criterion=KObjFunCallsFunctionStopCriterion(100),
    evomol_parameters={
        "optimization_parameters": {
            "pop_max_size": 300,
            "max_steps": 10,
            "k_to_replace": 10
        },
        "action_space_parameters": {
            "max_heavy_atoms": 15,
            "atoms": "C,N,O,F"
        }
    },
    evomol_init_pop_size=10,
    n_evomol_runs=10,
    n_best_evomol_retrieved=1,
    evomol_init_pop_strategy="random_weighted",
    results_path="test/test_grid_search",
    n_jobs_merit_optim=10,
    period_save=1,
    period_compute_test_predictions=1
)

alg.run()