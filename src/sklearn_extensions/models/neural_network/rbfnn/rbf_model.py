from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from numbers import Integral, Real
import scipy as sp
import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions, Options
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.cluster import KMeans

# pylint: disable=W0201


class RBFNNModel(ABC, BaseEstimator):
    """ """

    _parameter_constraints = {
        "n_units": [Interval(Integral, 1, None, closed="left")],
        "linear_layer": [BaseEstimator, None],
        "cluster_model": [BaseEstimator, None],
        "std_from_clusters": ["boolean"],
        "classification": ["boolean"],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        n_units: int,
        linear_layer,
        cluster_model,
        std_from_clusters,
        classification,
        random_state,
    ):
        self.n_units = n_units

        self.linear_layer = linear_layer
        self.cluster_model = cluster_model
        self.std_from_clusters = std_from_clusters
        self.classification = classification
        self.random_state = random_state

    def _fit_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Cluster input data into 'n_units' clusters
        self.cluster_model_ = self.cluster_model_.fit(X)
        cluster_idx = self.cluster_model_.predict(X)

        # Group points by cluster
        cluster_idx_sorted = cluster_idx.copy()
        cluster_idx_sorted.sort()
        _, cluster_idx_pos = np.unique(cluster_idx_sorted, return_index=True)
        X_sorted = X[cluster_idx.argsort()]
        X_grouped = np.split(X_sorted, cluster_idx_pos[1:])

        # Obtain centers and widths
        centers = self.cluster_model_.cluster_centers_
        widths = np.empty(self.n_units)
        for idx, X_cluster in enumerate(X_grouped):
            if self.std_from_clusters:
                widths[idx] = 1 / X_cluster.var()
            else:
                distance_to_centroid = sp.spatial.distance.cdist(X_cluster, centers[[idx], :])
                widths[idx] = np.sqrt(2 * self.n_units) / distance_to_centroid.max()

        return centers, widths

    def _rbf_layer(self, X: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """ """

        distances = sp.spatial.distance.cdist(X, centers) ** 2
        X_rbf = np.exp(-distances * widths)
        return X_rbf

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> RBFNNModel:
        """ """

        X, y = check_X_y(X, y)
        self._validate_params()
        if self.classification:
            self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        if not hasattr(self, "linear_layer_"):
            self.linear_layer_ = LinearRegression() if self.linear_layer is None else self.linear_layer

        self.cluster_model_ = KMeans(n_clusters=self.n_units, random_state=self.random_state) if self.cluster_model is None else self.cluster_model

        self.centers_, self.widths_ = self._fit_clustering(X)
        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)
        self.linear_layer_ = self.linear_layer_.fit(X_rbf, y)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ """

        check_is_fitted(self)
        X = check_array(X)

        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)

        return self.linear_layer_.predict(X_rbf)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ """

        check_is_fitted(self)
        X = check_array(X)

        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)

        return self.linear_layer_.predict_proba(X_rbf)


class RBFNNClassifier(RBFNNModel, ClassifierMixin):
    """ """

    _parameter_constraints = {
        **RBFNNModel._parameter_constraints,
        "return_probability": ["boolean"],
    }

    def __init__(
        self,
        n_units: int = 100,
        return_probability: bool = True,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        self.return_probability = return_probability

        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=True,
            random_state=random_state,
        )

    def fit(self, X, y):
        if self.linear_layer is None:
            if self.return_probability:
                self.linear_layer_ = LogisticRegression(random_state=self.random_state)
            else:
                self.linear_layer_ = RidgeClassifier(alpha=0, solver="svd", random_state=self.random_state)
        else:
            self.linear_layer_ = self.linear_layer

        return super().fit(X, y)


class RBFNNRegressor(RBFNNModel, RegressorMixin):
    """ """

    def __init__(
        self,
        n_units: int = 100,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=False,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.linear_layer_ = LinearRegression() if self.linear_layer is None else self.linear_layer
        return super().fit(X, y)
