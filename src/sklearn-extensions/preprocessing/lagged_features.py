from sklearn.base import BaseEstimator, TransformerMixin

class LaggedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag_time, apply_mask=False):
        self.lag_time = lag_time
        self.apply_mask = apply_mask

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if self.apply_mask:
            X_result = X[:-self.lead_time, :]
        else:
            X_result = self.full_like(X, np.nan)
            X_result[self.lead_time:, :] = X[:-self.lead_time, :]

        return y_result
    
    def transform_target(self, y):
        if self.apply_mask:
            y_result = y[self.lead_time:]
        else:
            y_result = self.full_like(y, np.nan)
            y_result[self.lead_time:, :] = y[self.lead_time:, :]

        return y_result
    
    def transform_dataset(self, X, y):
        return self.transform(X), self.transform_target(y)