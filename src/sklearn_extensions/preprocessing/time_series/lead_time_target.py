import numpy as np
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin

class LeadFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lead_time, flatten_output=True, apply_mask=False):
        self.lead_time = lead_time
        self.flatten_output = flatten_output
        self.apply_mask = apply_mask

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        ntimesteps = X.shape[0]
        nfeatures = X.shape[1]

        self.mask_ = np.arange(ntimesteps) >= self.lead_time

        windowed_features = np.full((ntimesteps, nfeatures, self.lead_time), np.nan, dtype=float)
        for idx, _ in enumerate(X[: -self.lead_time]):
            windowed_features[idx + self.lead_time, :] = X[idx : idx + self.lead_time, :].T

        if self.flatten_output:
            windowed_features = windowed_features.reshape((ntimesteps, nfeatures * self.lead_time))
        if self.apply_mask:
            windowed_features = windowed_features[self.mask_, :]

        return windowed_features

    def transform_target(self, y):
        if self.apply_mask:
            y_result = y[self.lead_time :]
        else:
            y_result = np.full_like(y, np.nan, dtype=float)
            y_result[self.lead_time :] = y[: -self.lead_time]

        return y_result

    def transform_dataset(self, X, y):
        return self.transform(X), self.transform_target(y)
