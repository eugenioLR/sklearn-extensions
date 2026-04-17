from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import _is_fitted
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels


class MultiplexedRegressor(BaseEstimator, RegressorMixin):
    _parameter_constraints = {"regressors": [list], "classifier": [BaseEstimator], "fit_on_predictions": ["boolean"], "verbose": ["verbose"]}

    def __init__(self, regressors, classifier, fit_on_predictions=False, verbose=False):
        self.regressors = regressors
        self.classifier = classifier
        self.fit_on_predictions = fit_on_predictions
        self.verbose = verbose

    def fit(self, X, y_class, y_reg):
        X, y_class = check_X_y(X, y_class)
        self.classes_ = unique_labels(y_class)
        X, y_reg = check_X_y(X, y_reg, y_numeric=True)
        self._validate_params()

        self.n_features_in_ = X.shape[1]

        if not _is_fitted(self.classifier):
            self.classifier_ = self.classifier.fit(X, y_class)
        else:
            self.classifier_ = self.classifier

        if self.fit_on_predictions:
            y_class_reference = self.classifier_.predict(X)
        else:
            y_class_reference = y_class

        self.regressors_ = [clone(reg) for reg in self.regressors]
        for idx, reg_model in enumerate(self.regressors_):
            mask = y_class_reference == idx
            if not _is_fitted(self.regressors[idx]):
                self.regressors_[idx] = reg_model.fit(X[mask], y_reg[mask])

            if self.verbose:
                print(f"{np.count_nonzero(mask)} points for class {idx}")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        pred_vector = np.empty(X.shape[0])

        pred_class = self.predict_class(X)
        for idx, reg_model in enumerate(self.regressors_):
            mask = pred_class == idx
            if np.count_nonzero(pred_class == idx) > 0:
                pred_vector[mask] = reg_model.predict(X[mask])

            if self.verbose:
                print(f"{np.count_nonzero(mask)} points for class {idx}")

        return pred_vector

    def predict_class(self, X):
        X = check_array(X)
        return self.classifier_.predict(X)

    def score_class(self, X, y):
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict_class(X))

    def score(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        return r2_score(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight)

    def score_report(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        return {
            "R2": r2_score(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight),
            "RMSE": root_mean_squared_error(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight),
            "MAE": mean_absolute_error(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight),
        }


if __name__ == "__main__":
    from sklearn.datasets import *
    from sklearn.linear_model import RidgeClassifier, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    X, y = make_regression(n_samples=30, n_features=4)
    y = y + np.random.normal(0, np.abs(y).mean() * 1e-3, y.shape)
    model = MultiplexedRegressor(
        classifier=RidgeClassifier(),
        regressors=[KNeighborsRegressor(), LinearRegression()],
    )
    model.fit(X, y_class=1 * (y > 0), y_reg=y)

    print(model.score(X, y))
