import joblib
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.ensemble import BaggingRegressor
import numpy as np


class SurrogateModel(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def uncertainty(self, X, smiles_list=None):
        raise NotImplementedError()

    def set_training_smiles(self, training_smiles):
        pass

    def fit(self, X, y, training_smiles=None):
        pass


class DummySurrogateModelWrapper(SurrogateModel):

    def __init__(self, base_model):
        """
        Dummy model wrapper calling fit and predict methods on the contained model and not defining an uncertainty
        method
        :param base_model:
        """
        self.model = base_model

    def fit(self, X, y, training_smiles=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def uncertainty(self, X, smiles_list=None):
        pass


class GPRSurrogateModelWrapper(SurrogateModel):

    def __init__(self, base_model):
        """
        Wrapper to use sklearn.gaussian_proccess.GaussianProcessRegressor as models providing an uncertainty estimation
        :param base_model: sklearn.gaussian_proccess.GaussianProcessRegressor model
        """

        self.model = base_model
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def fit(self, X, y=None, training_smiles=None):
        self.model.fit(X, y)
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def predict(self, X):

        y, std = self.model.predict(X, return_std=True)
        self.last_X_predicted = X
        self.std_last_X_predicted = std
        return y

    def uncertainty(self, X, smiles_list=None):

        if np.array_equal(X, self.last_X_predicted):
            return self.std_last_X_predicted
        else:
            _, std = self.model.predict(X, return_std=True)
            return std


def _get_fingerprint(smiles):
    """
    Computing the ECFP4 fingerprint of the given SMILES
    :param target_fp:
    :return:
    """
    return GetMorganFingerprint(MolFromSmiles(smiles), 2)


class TanimotoDistanceSurrogateModel(SurrogateModel):

    def __init__(self, model, cache_location=None, n_jobs=1):
        """
        Model defining a distance to points of the training set as an uncertainty measure. The distance is computed as
        the Tanimoto distance of ECFP4 fingerprints. The uncertainty is the minimal distance with points of the training
        set
        :param model: base model
        :param n_jobs: number of jobs to compute fingerprints
        """

        self.model = model
        self.n_jobs = n_jobs
        self.X_fingerprints = None
        self.X_smiles = None

        self.get_fingerprint_cached = joblib.Memory(location=cache_location, verbose=0).cache(_get_fingerprint)

    def fit(self, X, y=None, training_smiles=None):
        self.model.fit(X, y)

        self.X_fingerprints = []
        self.X_smiles = training_smiles
        for smi in training_smiles:
            self.X_fingerprints.append(self.get_fingerprint_cached(smi))

    def predict(self, X):
        return self.model.predict(X)

    def compute_distance_tanimoto(self, target_fp, fp):
        return 1 - TanimotoSimilarity(target_fp, fp)

    def compute_min_distance_tanimoto(self, fp, smi):
        # distances = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self.compute_distance)(self.X_fingerprints[i], fp) for i in range(len(self.X_fingerprints)))

        distances = []
        for i in range(len(self.X_fingerprints)):
            dist = self.compute_distance_tanimoto(fp, self.X_fingerprints[i])
            distances.append(dist)

        return np.min(distances)

    def uncertainty(self, X, smiles_list=None):

        min_distances = []

        for i, smi in enumerate(smiles_list):
            curr_smi_fp = _get_fingerprint(smi)
            min_distances.append(self.compute_min_distance_tanimoto(curr_smi_fp, smi))

        return np.array(min_distances)


class DescDistanceSurrogateModel(SurrogateModel):

    def __init__(self, model):
        """
        Model defining a distance to points of the training set as an uncertainty measure. The distance is computed as
        the euclidean distance of their descriptor. The uncertainty is the minimal distance with points of the training
        set
        :param model: base model
        :param n_jobs: number of jobs to compute fingerprints
        """

        self.model = model
        self.X_train = None

    def fit(self, X, y=None, training_smiles=None):
        self.model.fit(X, y)
        self.X_train = X

    def predict(self, X):
        return self.model.predict(X)

    def compute_distance(self, d1, d2):
        return np.linalg.norm(d1 - d2)

    def compute_min_distance(self, desc_value):
        distances = []

        for desc_training in self.X_train:
            distances.append(self.compute_distance(desc_training, desc_value))

        return np.min(distances)

    def uncertainty(self, X, smiles_list=None):

        min_distances = []

        for desc_value in X:
            min_distances.append(self.compute_min_distance(desc_value))

        return np.array(min_distances)


class BaggingSurrogateModel(SurrogateModel):

    def __init__(self, base_model, n_models, n_jobs=1):
        """
        Ensemble model using bagging providing an estimation of the value (the mean of the predictions of the models)
        and an uncertainty estimation (the std. deviation of the predictions of the models)
        :param base_model: base model
        :param n_models: number of models
        :param n_jobs: number of jobs for parallel training of the models
        """
        self.model = base_model
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.ensemble_models = BaggingRegressor(base_estimator=self.model, n_estimators=self.n_models,
                                                n_jobs=self.n_jobs, max_samples=1.0)

        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def fit(self, X, y=None, training_smiles=None):
        self.ensemble_models.fit(X, y)
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def predict(self, X):

        predictions = np.zeros((self.n_models, len(X)))

        for i, estimator in enumerate(self.ensemble_models.estimators_):
            predictions[i] = estimator.predict(X)

        self.last_X_predicted = X
        self.std_last_X_predicted = np.std(predictions, axis=0)

        return np.mean(predictions, axis=0)

    def uncertainty(self, X, smiles_list=None):

        if np.array_equal(X, self.last_X_predicted):
            return self.std_last_X_predicted
        else:
            predictions = np.zeros((self.n_models, len(X)))

            for i, estimator in enumerate(self.ensemble_models.estimators_):
                predictions[i] = estimator.predict(X)

            return np.std(predictions, axis=0)
