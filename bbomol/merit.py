import time
from abc import abstractmethod

from .bboalg import compute_descriptors
from evomol.evaluation import EvaluationStrategy
import numpy as np
from scipy.stats import norm


class Merit(EvaluationStrategy):
    """
    Base class of merit functions that are EvoMol EvaluationStrategy instances
    """

    def __init__(self, descriptors, pipeline, surrogate):
        super().__init__()
        self.descriptor = descriptors
        self.pipeline = pipeline
        self.surrogate = surrogate

        # Variables declared to store the version of the training dataset transformed using the pipeline and the list
        # of SMILES at any time (cf. self.update_training_dataset_method)
        self.dataset_X_transformed = None
        self.dataset_smiles_list = None

    def update_training_dataset(self, dataset_X, dataset_y, dataset_smiles):
        """
        Method called by the BBOAlg instance to notify the merit function strategy that the surrogate was just retrained
        on the given dataset.
        Computing the transformed version of the descriptors using the pipeline and recording the dataset's SMILES
        :return:
        """
        if self.pipeline is not None:
            self.dataset_X_transformed = self.pipeline.fit_transform(dataset_X)
        else:
            self.dataset_X_transformed = dataset_X

        self.dataset_smiles_list = dataset_smiles

    def evaluate_individual(self, individual, to_replace_idx=None):
        """
        Returning the evaluation of the given individual.

        If the descriptor cannot be built, returning -inf.

        :param individual:
        :param to_replace_idx:
        :return:
        """

        X, smiles_list, success = compute_descriptors(smiles_list=[individual.to_aromatic_smiles()],
                                                      descriptor=self.descriptor, pipeline=self.pipeline,
                                                      apply_pipeline=True)

        if success[0]:
            score = self.compute_merit_value(X)
            return score, [score]

        else:
            return float("-inf"), [float("-inf")]

    @abstractmethod
    def compute_merit_value(self, X):
        """
        Computing the merit value of the individual described by the X matrix
        :param X:
        :return:
        """
        pass

    def compute_record_scores_init_pop(self, population):
        """
        Computation of the scores of all individuals in EvoMol population at initialization.
        This method is overriden because descriptors of the individuals of the population were already computed
        during the BBOAlg surrogate training.
        :param population:
        :return:
        """

        self.scores = []
        self.comput_time = []
        for idx, ind in enumerate(population):
            if ind is not None:

                tstart = time.time()

                # Extracting SMILES of current individual
                smi = ind.to_aromatic_smiles()

                # Masking the extracting SMILES
                mask_smi = np.array(self.dataset_smiles_list) == smi

                # Extracting the transformed descriptors of given SMILES
                X = self.dataset_X_transformed[mask_smi]

                score = self.compute_merit_value(X)

                # Computing score
                self.scores.append(score)
                self.comput_time.append(time.time() - tstart)


class SurrogateValueMerit(Merit):

    def __init__(self, descriptor, pipeline, surrogate):
        """
        Merit function simply returning the value predicted by the surrogate function.
        :param descriptor:
        :param pipeline:
        :param surrogate:
        """
        super().__init__(descriptors=descriptor, pipeline=pipeline, surrogate=surrogate)

    def compute_merit_value(self, X):

        return self.surrogate.predict(X)[0]

    def keys(self):
        return ["Surrogate"]


class ExpectedImprovementMerit(Merit):

    def __init__(self, descriptor, pipeline, surrogate, xi=0.01, noise_based=False, init_pop_zero_EI=True):
        """
        Expected improvement merit function. Based on http://krasserm.github.io/2018/03/21/bayesian-optimization/
        :param xi: xi exploration parameter
        :param noise_based: whether the maximum known value is computed as the max of predictions (noise case) or as
        the max of dataset (genuine expected improvement).
        :param init_pop_zero_EI: whether the EI value is set to zero for individuals of initial population to gain
        computation time for solutions of the dataset
        See http://krasserm.github.io/2018/03/21/bayesian-optimization/ and
        https://arxiv.org/pdf/1012.2599.pdf and https://arxiv.org/abs/1012.2599
        """
        super().__init__(descriptors=descriptor, pipeline=pipeline, surrogate=surrogate)
        self.xi = xi
        self.noised_based = noise_based
        self.init_pop_zero_EI = init_pop_zero_EI

        print("EI xi : " + str(self.xi))
        print("noise based : " + str(noise_based))
        print("init_pop_zero_EI : " + str(init_pop_zero_EI))

        self.y_max = None

    def keys(self):
        return ["EI"]

    def update_training_dataset(self, dataset_X, dataset_y, dataset_smiles):
        super().update_training_dataset(dataset_X, dataset_y, dataset_smiles)

        print("UPDATING TRAINING DATASET")

        # See http://krasserm.github.io/2018/03/21/bayesian-optimization/ and
        # https://arxiv.org/pdf/1012.2599.pdf and https://arxiv.org/abs/1012.2599 for the noise-based case
        if self.noised_based:

            # Applying pipeline on training data
            if self.pipeline is None:
                X_dataset_transformed = dataset_X
            else:
                X_dataset_transformed = self.pipeline.transform(dataset_X)

            # Predicting training data
            mu_sample = self.surrogate.predict(X_dataset_transformed)

            self.y_max = np.max(mu_sample)

        else:
            self.y_max = np.max(dataset_y)

    def compute_merit_value(self, X):
        """
        From : http://krasserm.github.io/2018/03/21/bayesian-optimization/

        Computing the EI value of X that represents the descriptors of a single point.
        X is of shape (1, descriptors dimension)

        Returns:
            Expected improvement value of the point X.
        """

        # Predicting value
        mu = self.surrogate.predict(X)

        # Computing uncertainty
        sigma = self.surrogate.uncertainty(X)
        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='warn'):
            imp = mu - self.y_max - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei[0][0]

    def compute_record_scores_init_pop(self, population):
        """
        Setting EI of all defined individuals of the initial population to 0 to gain time.
        :param population:
        :return:
        """

        if self.init_pop_zero_EI:

            self.scores = []
            self.comput_time = []
            for i, ind in enumerate(population):
                if ind is not None:
                    self.scores.append(0)
                    self.comput_time.append(0)
        else:
            print("CALL TO SUPER")
            super(ExpectedImprovementMerit, self).compute_record_scores_init_pop(population)
