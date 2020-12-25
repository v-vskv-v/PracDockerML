import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed


def root_mean_squared_error(y_true, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = y_pred[np.newaxis, :]
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 base_estimator=DecisionTreeRegressor, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.base_estimator = base_estimator
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = np.ceil(X.shape[1] / 3).astype(np.int_)

        self.estimators = Parallel(n_jobs=-1)(delayed(self._task)(X, y)
                                              for _ in range(self.n_estimators))
        train_loss = []
        test_loss = []

        y_pred = 0
        y_pred_val = 0
        for i, estimator in enumerate(self.estimators):
            y_pred += estimator.predict(X)
            train_loss.append(root_mean_squared_error(y, y_pred / (i + 1)))
            if X_val is not None:
                y_pred_val += estimator.predict(X_val)
                test_loss.append(root_mean_squared_error(y_val, y_pred_val / (i + 1)))
        return self, train_loss, test_loss

    def _task(self, X, y):
        estimator = self.base_estimator(max_depth=self.max_depth,
                                        max_features=self.feature_subsample_size,
                                        **self.trees_parameters)
        idxs = np.random.randint(X.shape[0], size=X.shape[0])
        return estimator.fit(X[idxs], y[idxs])

    def predict(self, X, all=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if all:
            predictions = np.zeros((self.n_estimators, X.shape[0]))
        prediction = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.estimators):
            prediction += estimator.predict(X)
            if all:
                predictions[i] = prediction / (i + 1)
        if all:
            return predictions
        return prediction / self.n_estimators


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 base_estimator=DecisionTreeRegressor, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.base_estimator = base_estimator
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.estimators = []
        self.coefs = []
        train_loss = []
        test_loss = []
        y_pred = np.zeros(X.shape[0])
        if self.feature_subsample_size is None:
            self.feature_subsample_size = np.ceil(X.shape[1] / 3).astype(np.int_)
        for _ in range(self.n_estimators):
            estimator = self.base_estimator(max_depth=self.max_depth,
                                            max_features=self.feature_subsample_size,
                                            **self.trees_parameters)
            self.estimators.append(estimator.fit(X, 2 * (y - y_pred) / X.shape[0]))
            prediction = self.estimators[-1].predict(X)
            self.coefs.append(minimize_scalar(lambda x, y=y, h=y_pred, b=prediction:
                                              ((y - h - x * b) ** 2).mean()).x)
            y_pred += self.learning_rate * self.coefs[-1] * prediction
            train_loss.append(root_mean_squared_error(y, y_pred))
            if X_val is not None:
                y_pred_val = self.predict(X_val)
                test_loss.append(root_mean_squared_error(y_val, y_pred_val))
        return self, train_loss, test_loss

    def predict(self, X, all=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if all:
            predictions = np.zeros((self.n_estimators, X.shape[0]))
        prediction = np.zeros(X.shape[0])
        for i in range(len(self.estimators)):
            prediction += self.coefs[i] * self.estimators[i].predict(X)
            if all:
                predictions[i] = prediction
        if all:
            return self.learning_rate * predictions
        return self.learning_rate * prediction
