import pytest
import numpy as np
from sklearn_extensions.preprocessing import LaggedFeatureTransformer, LeadFeatureTransformer

# ----- LaggedFeatureTransformer -----
def test_lagged_transformer_init():
    lt = LaggedFeatureTransformer(lag_time=3)
    assert lt.lag_time == 3
    assert lt.apply_mask is False

def test_lagged_transformer_fit():
    lt = LaggedFeatureTransformer(2)
    X = np.random.randn(10, 4)
    lt.fit(X)
    # No state changes, just returns self
    assert hasattr(lt, 'apply_mask')  # dummy

def test_lagged_transformer_transform_no_mask():
    lt = LaggedFeatureTransformer(2, apply_mask=False)
    X = np.arange(20).reshape(10, 2)
    Xt = lt.transform(X)
    print(Xt)
    # First 2 rows should be NaN, rest shifted
    assert np.all(np.isnan(Xt[:2]))
    np.testing.assert_array_equal(Xt[2:], X[:-2])

def test_lagged_transformer_transform_mask():
    lt = LaggedFeatureTransformer(2, apply_mask=True)
    X = np.arange(20).reshape(10, 2)
    Xt = lt.transform(X)
    assert Xt.shape == (8, 2)  # first 2 removed
    np.testing.assert_array_equal(Xt, X[:-2])

def test_lagged_transformer_transform_target():
    lt = LaggedFeatureTransformer(3, apply_mask=False)
    y = np.arange(10)
    yt = lt.transform_target(y)
    # First 3 NaN, rest copied from y
    assert np.all(np.isnan(yt[:3]))
    np.testing.assert_array_equal(yt[3:], y[3:])

    lt_mask = LaggedFeatureTransformer(3, apply_mask=True)
    yt_mask = lt_mask.transform_target(y)
    np.testing.assert_array_equal(yt_mask, y[3:])

def test_lagged_transformer_transform_dataset():
    lt = LaggedFeatureTransformer(2, apply_mask=False)
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xt, yt = lt.transform_dataset(X, y)
    assert Xt.shape == (10, 2)
    assert yt.shape == (10,)
    # Check that transformation matches individual calls
    np.testing.assert_array_equal(Xt, lt.transform(X))
    np.testing.assert_array_equal(yt, lt.transform_target(y))

# ----- LeadFeatureTransformer -----
def test_lead_transformer_init():
    lf = LeadFeatureTransformer(lead_time=3)
    assert lf.lead_time == 3
    assert lf.flatten_output is True
    assert lf.apply_mask is False

def test_lead_transformer_fit():
    lf = LeadFeatureTransformer(2)
    X = np.random.randn(10, 4)
    lf.fit(X)
    assert hasattr(lf, 'lead_time')

def test_lead_transformer_transform_no_mask_flatten():
    lf = LeadFeatureTransformer(lead_time=3, flatten_output=True, apply_mask=False)
    X = np.arange(30).reshape(10, 3)  # 10 steps, 3 features
    Xt = lf.transform(X)
    # Expected shape: (10, 3*3=9) because flattened
    assert Xt.shape == (10, 9)
    # First 3 rows should be NaN (since need history)
    assert np.all(np.isnan(Xt[:3]))
    # Check one non-NaN row manually
    # For row index 5, features should be X[2:5] flattened
    expected_row_5 = X[2:5].flatten()
    np.testing.assert_array_equal(Xt[5], expected_row_5)

def test_lead_transformer_transform_no_mask_no_flatten():
    lf = LeadFeatureTransformer(lead_time=2, flatten_output=False, apply_mask=False)
    X = np.arange(20).reshape(10, 2)
    Xt = lf.transform(X)
    assert Xt.shape == (10, 2, 2)  # (steps, features, lead_time)
    assert np.all(np.isnan(Xt[:2]))
    np.testing.assert_array_equal(Xt[5], X[3:5].T)  # shape (2,2)

def test_lead_transformer_transform_mask():
    lf = LeadFeatureTransformer(lead_time=2, flatten_output=True, apply_mask=True)
    X = np.arange(20).reshape(10, 2)
    Xt = lf.transform(X)
    assert Xt.shape == (8, 4)  # steps from 2 onward, flattened
    expected = np.array([X[i-2:i].flatten() for i in range(2, 10)])
    np.testing.assert_array_equal(Xt, expected)

def test_lead_transformer_transform_target():
    lf = LeadFeatureTransformer(lead_time=3, apply_mask=False)
    y = np.arange(10)
    yt = lf.transform_target(y)
    assert np.all(np.isnan(yt[:3]))
    np.testing.assert_array_equal(yt[3:], y[:-3])

    lf_mask = LeadFeatureTransformer(3, apply_mask=True)
    yt_mask = lf_mask.transform_target(y)
    np.testing.assert_array_equal(yt_mask, y[3:])

def test_lead_transformer_transform_dataset():
    lf = LeadFeatureTransformer(2, apply_mask=False)
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xt, yt = lf.transform_dataset(X, y)
    np.testing.assert_array_equal(Xt, lf.transform(X))
    np.testing.assert_array_equal(yt, lf.transform_target(y))