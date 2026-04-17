import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.metrics._classification import _prf_divide

def circular_absolute_error(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    period = 2*np.pi
    pred_abs_diff = np.abs(y_true - y_pred)
    return np.mean(np.minimum(pred_abs_diff, period - pred_abs_diff))

def circular_root_mean_square_error(y_true, y_pred):
    period = 2*np.pi
    pred_abs_diff = np.abs(y_true - y_pred)
    angular_diff = np.minimum(pred_abs_diff, period - pred_abs_diff)

    return np.sqrt(np.mean(angular_diff**2))

def circular_bias(y_true, y_pred):
    period = 2*np.pi
    wrapped_angle_diff = (y_true - y_pred) % period
    mean_cos = np.cos(wrapped_angle_diff)
    mean_sin = np.sin(wrapped_angle_diff)

    return np.atan2(mean_sin, mean_cos)

def mean_cosine_deviation_loss(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    pred_diff = y_true - y_pred
    return np.mean(1 - np.cos(pred_diff))
