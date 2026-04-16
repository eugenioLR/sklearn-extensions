from __future__ import annotations
from abc import ABC
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, OneToOneFeatureMixin

# pylint: disable=all


class BernoulliBoltzmannMachine(BaseEstimator, OneToOneFeatureMixin):
    """
    https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
    """

    def __init__(self, iterations=1, learning_rate=0.1, hidden_units=10, k=1, syncronous=False):
        self.iterations = iterations
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.k = k
        self.syncronous = syncronous

    def energy(self, X_state):
        return -0.5 * np.einsum("ni,ji,nj->n", X_state, self.coef_, X_state) - np.einsum("i,ni->n", self.intercept_, X_state)

    def _log_prob_gradient(self, X_i):
        raise NotImplementedError
        input_dim = X_i.shape[1]
        activation = self.coef_ @ X_i

    def _gibbs_step(self, X_state):
        """Return probability of unit i being +1 and a sampled state."""
        activation = np.einsum("ij,nj->ni", self.coef_, X_state) + self.intercept_
        prob = 1 / (1 + np.exp(-2 * activation))  # for bipolar {-1,1}

        new_state = 2*(np.random.uniform(0, 1, prob.shape) < prob) - 1
        return new_state


    def fit(self, X):
        # raise NotImplementedError
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Ensure we get a bipolar input
        unique_values = set(np.unique(X))
        if unique_values.issubset({0, 1}):
            X = 2 * X - 1
        elif not unique_values.issubset({-1, 1}):
            raise ValueError("Input must be either binary ({0,1}) or bipolar ({-1, 1}).")
        
        self.network_size_ = self.n_features_in_ + self.hidden_units

        self.coef_ = np.random.normal(0, 0.01, (self.network_size_, self.network_size_))
        np.fill_diagonal(self.coef_, 0)
        self.intercept_ = np.zeros(self.network_size_)


        self.is_fitted_ = True
        return self

    def _update_network(self, X):
        raise NotImplementedError
        X_new = X

        for data_new_i, X_i in enumerate(X):
            X_i_bias = np.concatenate([X_i, np.ones(self.hidden_units)])
            idx_to_update = np.random.randint(X_i.shape[0])
            activation_weight = self.coef_[idx_to_update, :] @ X_i_bias
            X_new[data_new_i, idx_to_update] = 2 * (activation_weight > 0).astype(int) - 1

        return X_new

    def transform(self, X):
        X_new = X.copy()

        for _ in range(self.iterations):
            X_new = self._update_network(X_new)

        return X_new
