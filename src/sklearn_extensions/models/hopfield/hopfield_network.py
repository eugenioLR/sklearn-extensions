from __future__ import annotations
from numbers import Integral, Real 
import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, OneToOneFeatureMixin


# pylint: disable=W0201
class BernoulliHopfieldNetwork(BaseEstimator, OneToOneFeatureMixin):
    """
    https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
    """

    _parameter_constraints: dict = {
        "n_iter": [Interval(Integral, 1, None, closed='left')],
        "synchronous": ["boolean"],
        "fit_intercept": ["boolean"],
        "bipolar_output": ["boolean"],
        "verbose": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(self, n_iter=1, synchronous=False, fit_intercept=False, bipolar_output=False, verbose=False, random_state=None):
        self.n_iter = n_iter
        self.synchronous = synchronous
        self.fit_intercept = fit_intercept
        self.bipolar_output = bipolar_output
        self.verbose = verbose
        self.random_state = random_state
    
    def energy(self, X):
        energy = -0.5 * np.einsum("ni,ji,nj->n", X, self.coef_, X)
        if self.fit_intercept:
            energy += -np.einsum("i,ni->n", self.intercept_, X)

        return energy

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self._random_state = check_random_state(self.random_state)

        # Ensure we get a bipolar input
        if 0 in X:
            X = 2 * X - 1

        n_samples = X.shape[0]
        self.coef_ = 1/n_samples * X.T @ X
        if self.fit_intercept:
            self.intercept_ = 1/n_samples * np.mean(X, axis=0)
        np.fill_diagonal(self.coef_, 0)

        self.is_fitted_ = True
        return self

    def _update_network(self, X):

        if self.synchronous:
            activation_weight = np.einsum("ij,nj->ni", self.coef_, X)
            if self.fit_intercept:
                activation_weight += self.intercept_
            X = 2 * (activation_weight > 0).astype(int) - 1
        else:
            idx_to_update = self._random_state.randint(X.shape[1], size=X.shape[0])
            activation_weight = np.einsum("ij,nj->ni", self.coef_, X)
            if self.fit_intercept:
                activation_weight += self.intercept_
            activation_weight = activation_weight[np.arange(X.shape[0]), idx_to_update]
            X[np.arange(X.shape[0]), idx_to_update] = 2 * (activation_weight > 0).astype(int) - 1

        return X

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Ensure we get a bipolar input
        if 0 in X:
            X = 2 * X - 1

        for i in range(self.n_iter):
            if self.verbose:
                print(self.energy(X))
            X = self._update_network(X)

        # If needed, convert to binary output
        if not self.bipolar_output:
            X = (X + 1) // 2

        return X
