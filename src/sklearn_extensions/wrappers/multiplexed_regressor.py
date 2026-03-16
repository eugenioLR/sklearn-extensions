from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import _is_fitted
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, accuracy_score, f1_score

class MultiplexedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors, classifier, fit_on_predictions = False, verbose=False):
        self.regressors = regressors
        self.classifier = classifier
        self.fit_on_predictions = fit_on_predictions
        self.verbose = verbose

    def fit(self, X, y_class, y_reg):
        if not _is_fitted(self.classifier):
            self.classifier = self.classifier.fit(X, y_class)

        if self.fit_on_predictions:
            y_class_reference = self.classifier.predict(X)
        else:
            y_class_reference = y_class

        for idx, reg_model in enumerate(self.regressors):
            mask = y_class_reference == idx
            if not _is_fitted(self.regressors[idx]):
                self.regressors[idx] = reg_model.fit(X[mask], y_reg[mask])

            if self.verbose:
                print(f"{np.count_nonzero(mask)} points for class {idx}")

        return self
    
    def predict(self, X):
        pred_vector = np.empty(X.shape[0])

        pred_class = self.predict_class(X)
        for idx, reg_model in enumerate(self.regressors):
            mask = pred_class == idx
            if np.count_nonzero(pred_class == idx) > 0:
                pred_vector[mask] = reg_model.predict(X[mask])

            if self.verbose:
                print(f"{np.count_nonzero(mask)} points for class {idx}")

        return pred_vector
    
    def predict_class(self, X):
        return self.classifier.predict(X)

    def score_class(self, X, y):
        return accuracy_score(y_true = y, y_pred = self.predict_class(X))
    
    def score(self, X, y):
        return r2_score(y_true = y, y_pred = self.predict(X))

    def score_report(self, X, y):
        return {
            "R2": r2_score(y_true = y, y_pred = self.predict(X)),
            "RMSE": root_mean_squared_error(y_true = y, y_pred = self.predict(X)),
            "MAE": mean_absolute_error(y_true = y, y_pred = self.predict(X)),
        }

if __name__ == "__main__":
    from sklearn.datasets import *
    from sklearn.linear_model import RidgeClassifier, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    X, y = make_regression(n_samples=30, n_features=4)
    y = y + np.random.normal(0, np.abs(y).mean()*1e-3, y.shape)
    model = MultiplexedRegressor(classifier=RidgeClassifier(), regressors=[KNeighborsRegressor(), LinearRegression()])
    model.fit(X, y_class=1*(y > 0), y_reg=y)

    print(model.score(X, y))