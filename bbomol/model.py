import joblib
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.ensemble import BaggingRegressor
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import GridSearchCV


def compute_surrogate_predictions(surrogate_model, X):
    """
    Using the given surrogate model to predict the given input. Returning both prediction and uncertainty vectors
    :param surrogate_model: bbomol.model.SurrogateModel instance
    :param X: Descriptors for given input solutions
    :return: (prediction vector, uncertainty vector)
    """

    return surrogate_model.predict(X), surrogate_model.uncertainty(X)


class SurrogateModel(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def uncertainty(self, X, smiles_list=None):
        raise NotImplementedError()

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
        The model can either be a sklearn.gaussian_proccess.GaussianProcessRegressor instance of a
        sklearn.model_selection.GridSearchCV instance wrapping a sklearn.gaussian_proccess.GaussianProcessRegressor
        instance
        :param base_model: sklearn.gaussian_proccess.GaussianProcessRegressor or sklearn.model_selection.GridSearchCV
        instance
        """

        self.model = base_model
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def fit(self, X, y=None, training_smiles=None):

        # Fix in case of GridsearchCV with the starting population : copying several times the samples in initial step
        # if there are less samples that requested cross validations
        if isinstance(self.model, GridSearchCV):
            if len(X) < self.model.cv:
                X = np.tile(X, (self.model.cv, 1))
                y = np.tile(y, self.model.cv)

        self.model.fit(X, y)
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def predict(self, X):

        if isinstance(self.model, GaussianProcessRegressor):
            y, std = self.model.predict(X, return_std=True)
        elif isinstance(self.model, GridSearchCV):
            y, std = self.model.best_estimator_.predict(X, return_std=True)

        self.last_X_predicted = X
        self.std_last_X_predicted = std
        return y

    def uncertainty(self, X, smiles_list=None):

        if np.array_equal(X, self.last_X_predicted):
            return self.std_last_X_predicted
        else:
            _, std = self.get_model_instance().predict(X, return_std=True)
            return std

    def get_model_instance(self):

        if isinstance(self.model, GaussianProcessRegressor):
            return self.model
        elif isinstance(self.model, GridSearchCV):
            return self.model.best_estimator_

    def get_kernel_instance(self):
        return self.get_model_instance().kernel_

    def get_kernel_noise_values(self):
        """
        Returning a tuple corresponding to the noise values estimated by the model
        alpha_gpr is the noise parameterised in the GPR model
        (see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
        whitek_gpr is the noise estimated by the white kernel if defined
        (see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html)
        :return (alpha_gpr, whitek_gpr)
        """

        # Extracting alpha value
        alpha_gpr = self.get_model_instance().get_params()["alpha"]

        # Extracting kernel
        kernel = self.get_kernel_instance()

        whitekernel_idx = None
        # Iterating over all kernels to find a WhiteKernel
        for i in range(1, 10):
            if "k" + str(i) in kernel.get_params() and isinstance(kernel.get_params()["k" + str(i)], WhiteKernel):
                whitekernel_idx = i

        # Extracting whitekernel noise value if exists
        whitek_gpr = kernel.get_params()[
            "k" + str(whitekernel_idx) + "__noise_level"] if whitekernel_idx is not None else np.nan

        return alpha_gpr, whitek_gpr


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
