from __future__ import annotations
from numbers import Integral, Real 
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neural_network import BernoulliRBM

# pylint: disable=W0201


class BernoulliDBN(BaseEstimator, TransformerMixin):
    _parameter_constraints: dict = {
        "hidden_layer_sizes": ["array_like", Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }
    def __init__(
        self,
        hidden_layer_sizes=None,
        learning_rate=1e-3,
        batch_size=10,
        n_iter=10,
        verbose=False,
        random_state=None
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (256, 256)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
    
    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        X_next = X
        self.layers_ = []
        for idx, layer_size in enumerate(self.hidden_layer_sizes):
            rbm = BernoulliRBM(
                n_components=layer_size,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                n_iter=self.n_iter,
                verbose=self.verbose,
                random_state=self.random_state
            )

            X_next = rbm.fit_transform(X_next)
            self.layers_.append(rbm)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)

        for rbm in self.layers_:
            X = rbm.transform(X)
        
        return X
