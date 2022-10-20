from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
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
        self.base_model = base_model

    def fit(self, X, y, training_smiles=None):
        self.base_model.fit(X, y)

    def predict(self, X):
        return self.base_model.predict(X)

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

        self.base_model = base_model
        self.last_X_predicted = None
        self.std_last_X_predicted = None

        print("GPR model : " + str(self.base_model))
        print("GPR model params : " + str(base_model.get_params()))

    def fit(self, X, y=None, training_smiles=None):

        # Fix in case of GridsearchCV with the starting population : copying several times the samples in initial step
        # if there are less samples that requested cross validations
        if isinstance(self.base_model, GridSearchCV):
            if len(X) < self.base_model.cv:
                X = np.tile(X, (self.base_model.cv, 1))
                y = np.tile(y, self.base_model.cv)

        self.base_model.fit(X, y)
        self.last_X_predicted = None
        self.std_last_X_predicted = None

    def predict(self, X):

        y = None
        std = None

        if isinstance(self.base_model, GaussianProcessRegressor):
            y, std = self.base_model.predict(X, return_std=True)
        elif isinstance(self.base_model, GridSearchCV):

            try:
                y, std = self.base_model.best_estimator_.predict(X, return_std=True)
            except AttributeError:
                # Happens when the GridSearchCV has not been trained yet
                y, std = np.full((X.shape[0],), np.nan), np.full((X.shape[0],), np.nan)

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

        if isinstance(self.base_model, GaussianProcessRegressor):
            return self.base_model
        elif isinstance(self.base_model, GridSearchCV):
            return self.base_model.best_estimator_

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
