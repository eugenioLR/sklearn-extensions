import pytest
import numpy as np
import torch

@pytest.fixture
def random_data():
    """Generate random regression/classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_class = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 0] * 2 + np.random.randn(100) * 0.1
    return X, y_class, y_reg

@pytest.fixture
def time_series_data():
    """Generate simple time series."""
    np.random.seed(42)
    t = np.arange(50)
    X = np.sin(0.2 * t).reshape(-1, 1) + np.random.randn(50, 1) * 0.1
    y = np.roll(X, -1)[:-1]  # next step prediction
    return X[:-1], y