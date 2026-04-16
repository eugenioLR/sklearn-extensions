from abc import ABC, abstractmethod
import numpy as np
from numbers import Integral, Real
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, Options
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, RidgeClassifier

class BaseExtremeLearningMachine(ABC, BaseEstimator):
    _parameter_constraints: dict = {
        "hidden_layer_size": [Interval(Integral, 1, None, closed="left")],
        "activation": [StrOptions({"identity", "logistic", "tanh", "relu"})],
        "alpha": [Interval(Real, 0, None, closed="left"), None],
        "linear_layer": [BaseEstimator, None],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        hidden_layer_size,
        activation,
        alpha,
        linear_layer,
        random_state
    ):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.linear_layer = linear_layer
        self.random_state = random_state

    def _init_coef(self, fan_in, fan_out):
        """
        Adapted from sklearn.neural_network._multilayer_perceptron
        """

        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        return coef_init, intercept_init

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self._random_state = check_random_state(self.random_state)
        self._validate_params()

        if hasattr(self, 'linear_layer_'):
            self.linear_layer_ = self.linear_layer_
        else:
            self.linear_layer_ = LinearRegression() if self.linear_layer is None else self.linear_layer

        hidden_activation = ACTIVATIONS[self.activation]

        self.hidden_coef_, self.hidden_intercept_ = self._init_coef(X.shape[1], self.hidden_layer_size)

        activations = safe_sparse_dot(X, self.hidden_coef_,)
        activations += self.hidden_intercept_
        hidden_activation(activations)
        self.linear_layer_ = self.linear_layer_.fit(activations, y)

        if hasattr(self.linear_layer_, 'classes_'):
            self.classes_ = self.linear_layer_.classes_

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X)

        hidden_activation = ACTIVATIONS[self.activation]

        activations = safe_sparse_dot(X, self.hidden_coef_,)
        activations += self.hidden_intercept_
        hidden_activation(activations)
        pred = self.linear_layer_.predict(activations)

        return pred

class ELMClassifier(BaseExtremeLearningMachine, ClassifierMixin):
    _parameter_constraints = {
        **BaseExtremeLearningMachine._parameter_constraints,
        "return_probability": ["boolean"]
    }

    def __init__(
        self,
        hidden_layer_size = 100,
        activation = "relu",
        alpha = None,
        return_probability = True,
        linear_layer = None,
        random_state = None
    ):
        self.return_probability = return_probability
        super().__init__(
            hidden_layer_size,
            activation,
            alpha,
            linear_layer,
            random_state,
        )
    
    def fit(self, X, y):
        if self.linear_layer is None:
            if self.return_probability:
                if self.alpha is None:
                    self.linear_layer_ = LogisticRegression(random_state=self.random_state)
                elif self.alpha == 0:
                    self.linear_layer_ = LogisticRegression(C=np.inf, random_state=self.random_state)
                else:
                    self.linear_layer_ = LogisticRegression(C=1/self.alpha, random_state=self.random_state)
            else:
                if self.alpha is None:
                    self.linear_layer_ = RidgeClassifier(alpha=0, solver="svd", random_state=self.random_state)
                else:
                    self.linear_layer_ = RidgeClassifier(alpha=self.alpha, solver="svd", random_state=self.random_state)
        else:
            self.linear_layer_ = self.linear_layer
        
        return super().fit(X, y)
    
    def predict_proba(self, X):
        if not self.return_probability:
            raise AttributeError("This classifier was created with return_probability=False")

        X = check_array(X)

        hidden_activation = ACTIVATIONS[self.activation]

        activations = safe_sparse_dot(X, self.hidden_coef_,)
        activations += self.hidden_intercept_
        hidden_activation(activations)
        pred = self.linear_layer_.predict_proba(activations)

        return pred

class ELMRegressor(BaseExtremeLearningMachine, RegressorMixin):
    def __init__(
        self,
        hidden_layer_size = 100,
        activation = "relu",
        alpha = None,
        linear_layer = None,
        random_state = None
    ):
        super().__init__(
            hidden_layer_size,
            activation,
            alpha,
            linear_layer,
            random_state,
        )
    
    def fit(self, X, y):
        if self.linear_layer is None:
            if self.alpha is None:
                self.linear_layer_ = LinearRegression()
            else:
                self.linear_layer_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            self.linear_layer_ = self.linear_layer
        return super().fit(X, y)

if __name__ == "__main__":
    import sklearn
    X, y = sklearn.datasets.make_regression(n_features=10)
    model = ELMRegressor(activation="relu", alpha=0)
    model.fit(X, y)
    pred = model.predict(X)

    print(model.score(X, y))

    X, y = sklearn.datasets.make_classification(n_features=10)
    model = ELMClassifier(activation="relu", return_probability=False)
    model.fit(X, y)
    pred = model.predict(X)

    print(model.score(X, y))

    X, y = sklearn.datasets.make_classification(n_features=10)
    model = ELMClassifier(activation="relu", return_probability=True)
    model.fit(X, y)
    pred = model.predict(X)

    print(model.score(X, y))