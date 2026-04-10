from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, OneToOneFeatureMixin


# pylint: disable=W0201
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
        return -0.5 * np.einsum("ni,ji,nj->n", X, self.coef_, X)

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Ensure we get a bipolar input
        if 0 in X:
            X = 2 * X - 1

        self.coef_ = (1 / X.shape[0]) * X.T @ X
        np.fill_diagonal(self.coef_, 0)

        self.is_fitted_ = True
        return self

    def _update_network(self, X):
        if self.synchronous:
            activation_weight = np.einsum("ij, nj -> ni", self.coef_, X)
            X = 2 * (activation_weight > 0).astype(int) - 1
        else:
            idx_to_update = np.random.randint(X.shape[1], size=X.shape[0])
            activation_weight = np.einsum("ij, nj -> ni", self.coef_, X)[np.arange(X.shape[0]), idx_to_update]
            X[np.arange(X.shape[0]), idx_to_update] = 2 * (activation_weight > 0).astype(int) - 1

        return X

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Ensure we get a bipolar input
        if 0 in X:
            X = 2 * X - 1

        for _ in range(self.iterations):
            if self.verbose:
                print(self.energy(X))
            X = self._update_network(X)

        # If needed, convert to binary output
        if not self.bipolar_output:
            X = (X + 1) // 2

        return X
