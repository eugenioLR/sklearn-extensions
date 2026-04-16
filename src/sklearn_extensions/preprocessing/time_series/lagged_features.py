import numpy as np
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin

class LaggedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag_time, apply_mask=False):
        self.lag_time = lag_time
        self.apply_mask = apply_mask

    def fit(self, X, y=None, **fit_params):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = check_array(X)
        if self.apply_mask:
            X_result = X[: -self.lag_time, :]
        else:
            X_result = np.full_like(X, np.nan, dtype=float)
            X_result[self.lag_time :, :] = X[: -self.lag_time, :]

        return X_result

    def transform_target(self, y):
        if self.apply_mask:
            y_result = y[self.lag_time :]
        else:
            y_result = np.full_like(y, np.nan, dtype=float)
            y_result[self.lag_time :, :] = y[self.lag_time :, :]

        return y_result

    def transform_dataset(self, X, y):
        return self.transform(X), self.transform_target(y)
