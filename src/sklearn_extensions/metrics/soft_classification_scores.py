import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.metrics._classification import _prf_divide

def soft_accuracy_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    n_samples = y_true.shape[0]

    tp_sum = np.sum(y_true * y_pred)
    tn_sum = np.sum((1 - y_true) * (1 - y_pred))

    return (tp_sum + tn_sum) / n_samples

def soft_precision_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    product_sum = np.sum(y_true * y_pred)
    pred_sum = np.sum(y_pred)
    return product_sum / (pred_sum + 1e-8)


def soft_recall_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    product_sum = np.sum(y_true * y_pred)
    true_sum = np.sum(y_true)
    return product_sum / (true_sum + 1e-8)


def soft_balanced_accuracy_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    tpr = np.sum(y_true * y_pred) / np.sum(y_true)
    tnr = np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true)

    if not np.isfinite(tpr) or not np.isfinite(tnr):
        return 0

    return (tpr + tnr) / 2


def soft_geometric_mean_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    tpr = np.sum(y_true * y_pred) / np.sum(y_true)
    tnr = np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true)

    if not np.isfinite(tpr) or not np.isfinite(tnr):
        return 0
    
    return np.sqrt(tpr * tnr)


def soft_f1_score(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    product_sum = np.sum(y_true * y_pred)
    true_pred_sum = np.sum(y_true + y_pred)

    return 2 * product_sum / (true_pred_sum + 1e-8)


def fbeta_score_prob(y_true, y_pred, beta):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    prec = soft_precision_score(y_true, y_pred)
    rec = soft_recall_score(y_true, y_pred)
    return (1 + beta**2) * (prec * rec) /((beta**2 * prec) + rec + 1e-8)

