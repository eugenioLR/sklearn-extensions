import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.metrics._classification import _prf_divide

def circular_mean_absolute_error(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    period = 2 * np.pi
    pred_abs_diff = np.abs(y_true - y_pred)
    return np.mean(np.minimum(pred_abs_diff, period - pred_abs_diff))


def circular_root_mean_square_error(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    period = 2 * np.pi
    pred_abs_diff = np.abs(y_true - y_pred)
    angular_diff = np.minimum(pred_abs_diff, period - pred_abs_diff)

    return np.sqrt(np.mean(angular_diff**2))


def circular_bias(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    pred_diff = y_true - y_pred
    mean_cos = np.mean(np.cos(pred_diff))
    mean_sin = np.mean(np.sin(pred_diff))

    return np.atan2(mean_sin, mean_cos)


def mean_cosine_deviation(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    pred_diff = y_true - y_pred
    return np.mean(1 - np.cos(pred_diff))


def circular_r2_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    pred_diff = y_true - y_pred
    ss_res_circ = np.sum(1 - np.cos(pred_diff))

    mean_cos = np.mean(np.cos(y_true))
    mean_sin = np.mean(np.sin(y_true))
    mean_angle = np.atan2(mean_sin, mean_cos)

    ss_tot_circ = np.sum(1 - np.cos(y_true - mean_angle))
    
    return 1 - ss_res_circ/(ss_tot_circ + 1e-8)
