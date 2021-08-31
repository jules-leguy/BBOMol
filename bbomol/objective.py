import concurrent
import time
from concurrent.futures.process import ProcessPoolExecutor
import numpy as np
from evomol.molgraphops.molgraph import MolGraph
from rdkit.Chem.rdmolfiles import MolFromSmiles
from sklearn.base import BaseEstimator, TransformerMixin


class EvoMolEvaluationStrategyWrapper(TransformerMixin, BaseEstimator):
    """
    Wrapper transforming an evomol.evaluation.EvaluationStrategyComposant instance into an objective function class
    that can be used for BBO.
    Adding fit method and parallel computation.
    """

    def __init__(self, evaluation_strategy, n_jobs=1):
        """
        :param evaluation_strategy: evomol.evaluation.EvaluationStrategyComposant instance
        :param n_jobs: number of jobs for parallel computation of values
        """
        self.evaluation_strategy = evaluation_strategy
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def get_all_scores_number(self):
        """
        Returning the number of values that will be returned in the all_scores variable. In case of single objective
        it is equal to 1. If there are multiple objective combined, it is equal to the number of sub-objectives.
        :return:
        """
        return len(self.evaluation_strategy.keys())

    def disable_calls_count(self):
        """
        Method disabling the count of calls to the objective function. Used to ignore the computation of training or
        testing datasets in the total count
        :return:
        """
        self.evaluation_strategy.disable_calls_count()

    def enable_calls_count(self):
        """
        Enabling the count of calls to the objective function.
        :return:
        """
        self.evaluation_strategy.enable_calls_count()

    @property
    def calls_count(self):
        """
        Property returning the number of calls to the objective function
        :return:
        """
        return self.evaluation_strategy.n_calls

    def keys(self):
        """
        Method returning the keys (names) of computed properties
        :return:
        """
        return self.evaluation_strategy.keys()

    def evaluate_individual(self, individual):
        """
        Wrapping the call to the evaluation to record the computation time
        :param individual:
        :return: score, other_scores, computation time
        """

        tstart = time.time()

        score, other_scores = self.evaluation_strategy.evaluate_individual(individual)

        computation_time = time.time() - tstart

        return score, other_scores, computation_time

    def transform(self, X):

        # Computing objective values in parallel using concurrent.futures so that Exceptions are handled properly when
        # the computation fails
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(len(X)):
                futures.append(executor.submit(self.evaluate_individual, MolGraph(MolFromSmiles(X[i]))))

        scores = []
        all_scores_list = []
        successes = []
        computation_times = []

        # Retrieving all parallel results
        for future in futures:
            try:
                score, other_scores, computation_time = future.result()
                scores.append(score)
                all_scores_list.append(other_scores)
                successes.append(True)
                computation_times.append(computation_time)
            except Exception:
                scores.append(None)
                all_scores_list.append(np.full((len(self.keys()),), None))
                successes.append(False)
                computation_times.append(None)

        # Reshaping arrays
        scores = np.array(scores).reshape(-1, )
        successes = np.array(successes).reshape(-1, )

        return scores, all_scores_list, successes, computation_times
