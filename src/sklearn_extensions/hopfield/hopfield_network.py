from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin


class BernoulliHopfieldNetwork(BaseEstimator, OneToOneFeatureMixin):
    """
    https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
    """

    def __init__(self, iterations=1, synchronous=False, verbose=False, bipolar_output=False):
        self.iterations = iterations
        self.synchronous = synchronous
        self.verbose = verbose
        self.bipolar_output = bipolar_output

    def energy(self, X):
        return -0.5 * np.einsum('ni,ji,nj->n', X, self.coef_, X)

    def fit(self, X):
        if 0 in X:
            X = 2*X - 1

        self.is_fitted_ = True

        self.coef_ = (1 / X.shape[0]) * X.T @ X
        np.fill_diagonal(self.coef_, 0)

        return self

    def _update_network(self, X):
        if self.synchronous:
            activation_weight = np.einsum('ij, nj -> ni', self.coef_, X)
            if self.bipolar_output:
                X = 2*(activation_weight > 0).astype(int) - 1
            else:
                X = (activation_weight > 0).astype(int)
        else:
            idx_to_update = np.random.randint(X.shape[1], size=X.shape[0])
            activation_weight = np.einsum('ij, nj -> ni', self.coef_, X)[np.arange(X.shape[0]), idx_to_update]
            if self.bipolar_output:
                X[np.arange(X.shape[0]), idx_to_update] = 2*(activation_weight > 0).astype(int) - 1
            else:
                X[np.arange(X.shape[0]), idx_to_update] = (activation_weight > 0).astype(int)

        return X

    def transform(self, X):
        X_new = X.copy()

        for _ in range(self.iterations):
            if self.verbose:
                print(self.energy(X_new))
            X_new = self._update_network(X_new)

        return X_new