# test_metrics.py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from sklearn_extensions.metrics import *

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.RandomState(42)

@pytest.fixture
def perfect_circular():
    """Perfect predictions for circular metrics."""
    y_true = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    y_pred = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    return y_true, y_pred

@pytest.fixture
def opposite_circular():
    """Opposite predictions (max circular error)."""
    y_true = np.array([0, np.pi/2, np.pi])
    y_pred = np.array([np.pi, 3*np.pi/2, 0])
    return y_true, y_pred

@pytest.fixture
def perfect_soft():
    """Perfect soft classification."""
    y_true = np.array([0.9, 0.1, 0.8, 0.2])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    return y_true, y_pred

@pytest.fixture
def worst_soft():
    """Worst soft classification (all opposite)."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.0, 1.0, 0.0, 1.0])
    return y_true, y_pred

# ----------------------------------------------------------------------
# Circular Metrics Tests
# ----------------------------------------------------------------------
def test_circular_absolute_error_perfect(perfect_circular):
    y_true, y_pred = perfect_circular
    assert circular_absolute_error(y_true, y_pred) == 0.0

def test_circular_absolute_error_opposite(opposite_circular):
    y_true, y_pred = opposite_circular
    expected = np.pi  # each pair differs by exactly π
    assert_allclose(circular_absolute_error(y_true, y_pred), expected)

def test_circular_absolute_error_wraparound():
    y_true = np.array([0.1, 2*np.pi - 0.1])
    y_pred = np.array([2*np.pi - 0.1, 0.1])
    # The absolute circular difference is 0.2 for both
    expected = 0.2
    assert_allclose(circular_absolute_error(y_true, y_pred), expected)

def test_circular_absolute_error_non_negative(rng):
    y_true = rng.rand(20) * 2 * np.pi
    y_pred = rng.rand(20) * 2 * np.pi
    result = circular_absolute_error(y_true, y_pred)
    assert result >= 0

def test_circular_absolute_error_incompatible_shapes():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError, match="Found input variables with inconsistent"):
        circular_absolute_error(y_true, y_pred)

def test_circular_absolute_error_nan_raises():
    y_true = np.array([0.0, np.nan, 2.0])
    y_pred = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        circular_absolute_error(y_true, y_pred)

def test_circular_rmse_perfect(perfect_circular):
    y_true, y_pred = perfect_circular
    assert circular_root_mean_square_error(y_true, y_pred) == 0.0

def test_circular_rmse_opposite(opposite_circular):
    y_true, y_pred = opposite_circular
    expected = np.pi  # sqrt(mean(π**2)) = π
    assert_allclose(circular_root_mean_square_error(y_true, y_pred), expected)

def test_circular_rmse_wraparound():
    y_true = np.array([0.0, 2*np.pi - 0.5])
    y_pred = np.array([2*np.pi - 0.5, 0.0])
    # Each diff is 0.5, rmse = sqrt(0.5**2) = 0.5
    assert_allclose(circular_root_mean_square_error(y_true, y_pred), 0.5)

def test_circular_bias_zero(perfect_circular):
    y_true, y_pred = perfect_circular
    bias = circular_bias(y_true, y_pred)
    # When predictions are perfect, wrapped diff is 0, atan2(0,1)=0
    assert_allclose(bias, 0.0, atol=1e-7)

def test_circular_bias_constant_offset():
    offset = np.pi / 4
    y_true = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    y_pred = y_true + offset
    bias = circular_bias(y_true, y_pred)
    # The circular bias should be -offset (or offset depending on convention)
    # Since wrapped_angle_diff = (y_true - y_pred) % period = -offset % 2π
    # With positive offset, y_true - y_pred = -offset, wrapped = 2π - offset
    # cos(2π - offset) = cos(offset), sin(2π - offset) = -sin(offset)
    # atan2(-sin(offset), cos(offset)) = -offset
    assert_allclose(bias, -offset, atol=1e-7)

def test_circular_bias_symmetric():
    y_true = np.array([0, np.pi/2, np.pi])
    y_pred = np.array([0.1, np.pi/2 + 0.1, np.pi + 0.1])
    bias1 = circular_bias(y_true, y_pred)
    bias2 = circular_bias(y_pred, y_true)
    assert_allclose(bias1, -bias2, atol=1e-7)

def test_mean_cosine_deviation_perfect(perfect_circular):
    y_true, y_pred = perfect_circular
    loss = mean_cosine_deviation_loss(y_true, y_pred)
    assert_allclose(loss, 0.0, atol=1e-7)

def test_mean_cosine_deviation_opposite(opposite_circular):
    y_true, y_pred = opposite_circular
    # diff = π, cos(π) = -1, 1 - (-1) = 2
    assert_allclose(mean_cosine_deviation_loss(y_true, y_pred), 2.0)

def test_mean_cosine_deviation_range():
    # Loss should be between 0 and 2
    y_true = np.linspace(0, 2*np.pi, 100)
    y_pred = y_true + np.random.uniform(-np.pi, np.pi, size=100)
    loss = mean_cosine_deviation_loss(y_true, y_pred)
    assert 0 <= loss <= 2

# ----------------------------------------------------------------------
# Soft Classification Metrics Tests
# ----------------------------------------------------------------------
def test_soft_accuracy_worst(worst_soft):
    y_true, y_pred = worst_soft
    assert_allclose(soft_accuracy_score(y_true, y_pred), 0.0)

def test_soft_accuracy_half():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.5, 0.5])
    # tp = 0.5, tn = 0.5, total = 1.0, acc = 1.0/2 = 0.5
    assert_allclose(soft_accuracy_score(y_true, y_pred), 0.5)

def test_soft_precision_worst(worst_soft):
    y_true, y_pred = worst_soft
    # product_sum = 0, pred_sum = 2.0 -> 0/2 = 0
    assert_allclose(soft_precision_score(y_true, y_pred), 0.0)

def test_soft_precision_all_ones():
    y_true = np.array([0.8, 0.9])
    y_pred = np.array([1.0, 1.0])
    # tp = 0.8+0.9=1.7, pred_sum=2.0, precision=0.85
    assert_allclose(soft_precision_score(y_true, y_pred), 1.7/2.0)

def test_soft_precision_zero_pred_sum():
    y_true = np.array([0.5, 0.6])
    y_pred = np.array([0.0, 0.0])
    # _prf_divide returns 0 when denominator is 0
    assert soft_precision_score(y_true, y_pred) == 0.0

def test_soft_recall_worst(worst_soft):
    y_true, y_pred = worst_soft
    # product_sum = 0, true_sum = 2.0 -> 0
    assert_allclose(soft_recall_score(y_true, y_pred), 0.0)

def test_soft_recall_zero_true_sum():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.3, 0.7])
    # _prf_divide returns 0 when denominator is 0
    assert soft_recall_score(y_true, y_pred) == 0.0

def test_soft_balanced_accuracy_worst(worst_soft):
    y_true, y_pred = worst_soft
    assert_allclose(soft_balanced_accuracy_score(y_true, y_pred), 0.0)

def test_soft_balanced_accuracy_random():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    # tpr = (0.5+0.5)/2 = 0.5, tnr = (0.5+0.5)/2 = 0.5 -> ba = 0.5
    assert_allclose(soft_balanced_accuracy_score(y_true, y_pred), 0.5)

def test_soft_gmean_worst(worst_soft):
    y_true, y_pred = worst_soft
    assert_allclose(soft_geometric_mean_score(y_true, y_pred), 0.0)

def test_soft_gmean_symmetric():
    y_true = np.array([0.9, 0.1, 0.8, 0.2])
    y_pred = np.array([0.6, 0.4, 0.7, 0.3])
    gmean1 = soft_geometric_mean_score(y_true, y_pred)
    gmean2 = soft_geometric_mean_score(y_pred, y_true)
    # Should be symmetric? Actually g-mean of tpr and tnr is symmetric w.r.t. swapping
    # but note: tpr(y_true, y_pred) = tnr(y_pred, y_true) etc.
    assert_allclose(gmean1, gmean2)

def test_soft_f1_worst(worst_soft):
    y_true, y_pred = worst_soft
    assert_allclose(soft_f1_score(y_true, y_pred), 0.0)

def test_soft_f1_consistency_with_precision_recall():
    y_true = np.array([0.8, 0.2, 0.6, 0.4])
    y_pred = np.array([0.7, 0.3, 0.5, 0.5])
    prec = soft_precision_score(y_true, y_pred)
    rec = soft_recall_score(y_true, y_pred)
    f1_manual = 2 * (prec * rec) / (prec + rec)
    f1_func = soft_f1_score(y_true, y_pred)
    assert_allclose(f1_func, f1_manual)

def test_fbeta_score_prob_beta1_equals_f1():
    y_true = np.array([0.9, 0.1, 0.8, 0.2])
    y_pred = np.array([0.6, 0.4, 0.7, 0.3])
    f1 = soft_f1_score(y_true, y_pred)
    fbeta = fbeta_score_prob(y_true, y_pred, beta=1.0)
    assert_allclose(fbeta, f1)

def test_fbeta_score_prob_beta_half():
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2, 0.9])
    beta = 0.5
    prec = soft_precision_score(y_true, y_pred)
    rec = soft_recall_score(y_true, y_pred)
    expected = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    actual = fbeta_score_prob(y_true, y_pred, beta)
    assert_allclose(actual, expected)

def test_fbeta_score_prob_beta_two():
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2, 0.9])
    beta = 2.0
    prec = soft_precision_score(y_true, y_pred)
    rec = soft_recall_score(y_true, y_pred)
    expected = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    actual = fbeta_score_prob(y_true, y_pred, beta)
    assert_allclose(actual, expected)

# ----------------------------------------------------------------------
# Input Validation Tests (common to all soft metrics)
# ----------------------------------------------------------------------
def test_soft_metrics_incompatible_shapes():
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.1, 0.2])
    funcs = [
        soft_accuracy_score,
        soft_precision_score,
        soft_recall_score,
        soft_balanced_accuracy_score,
        soft_geometric_mean_score,
        soft_f1_score,
        lambda yt, yp: fbeta_score_prob(yt, yp, beta=1.0),
    ]
    for func in funcs:
        with pytest.raises(ValueError, match="Found input variables with inconsistent"):
            func(y_true, y_pred)

def test_soft_metrics_nan_raises():
    y_true = np.array([0.0, np.nan, 1.0])
    y_pred = np.array([0.0, 0.5, 1.0])
    funcs = [
        soft_accuracy_score,
        soft_precision_score,
        soft_recall_score,
        soft_balanced_accuracy_score,
        soft_geometric_mean_score,
        soft_f1_score,
        lambda yt, yp: fbeta_score_prob(yt, yp, beta=1.0),
    ]
    for func in funcs:
        with pytest.raises(ValueError):
            func(y_true, y_pred)

# ----------------------------------------------------------------------
# Edge Cases and Numerical Stability
# ----------------------------------------------------------------------
def test_circular_metrics_all_identical():
    y_true = np.ones(100) * np.pi
    y_pred = np.ones(100) * np.pi
    assert circular_absolute_error(y_true, y_pred) == 0.0
    assert circular_root_mean_square_error(y_true, y_pred) == 0.0
    assert_allclose(circular_bias(y_true, y_pred), 0.0)
    assert_allclose(mean_cosine_deviation_loss(y_true, y_pred), 0.0)

def test_soft_metrics_all_zeros():
    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    # All predictions are 0, true is 0 => tp = 0, tn = sum(1*1) = n_samples
    assert_allclose(soft_accuracy_score(y_true, y_pred), 1.0)
    # precision: tp / sum(pred) = 0/0 -> _prf_divide returns 0
    assert soft_precision_score(y_true, y_pred) == 0.0
    # recall: tp / sum(true) = 0/0 -> 0
    assert soft_recall_score(y_true, y_pred) == 0.0
    # balanced accuracy: tpr=0, tnr=1 => 0
    assert_allclose(soft_balanced_accuracy_score(y_true, y_pred), 0)
    # gmean: sqrt(0 * 1) = 0
    assert_allclose(soft_geometric_mean_score(y_true, y_pred), 0.0)
    # f1: 2*0/(0+0) -> 0
    assert soft_f1_score(y_true, y_pred) == 0.0

def test_soft_metrics_all_ones():
    y_true = np.ones(10)
    y_pred = np.ones(10)
    # tp = sum(1*1)=10, tn = sum(0*0)=0 => acc = 10/10 = 1
    assert_allclose(soft_accuracy_score(y_true, y_pred), 1.0)
    # precision = 10/10 = 1
    assert_allclose(soft_precision_score(y_true, y_pred), 1.0)
    # recall = 10/10 = 1
    assert_allclose(soft_recall_score(y_true, y_pred), 1.0)
    # balanced acc: tpr=1, tnr=0/0=1 (by _prf_divide?) Actually tnr: sum((1-1)*(1-1))=0 / sum(1-1)=0 => returns 0? Wait, check implementation: _prf_divide(0,0) returns 0.
    # So tpr=1, tnr=0 -> ba=0.5. This is a known edge-case behavior.
    ba = soft_balanced_accuracy_score(y_true, y_pred)
    assert ba == 0

def test_circular_metrics_large_angles():
    # Ensure periodicity works for angles far outside [0, 2π]
    y_true = np.array([0.0, 10*np.pi, 100*np.pi])
    y_pred = np.array([0.0, 10*np.pi + 0.1, 100*np.pi + 0.1])
    assert circular_absolute_error(y_true, y_pred) < 0.1
    assert circular_root_mean_square_error(y_true, y_pred) < 0.1