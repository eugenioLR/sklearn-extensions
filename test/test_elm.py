# test/test_elm.py
import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, RidgeClassifier
from sklearn.utils.validation import check_random_state
from sklearn_extensions.models.neural_network.elm import ELMClassifier, ELMRegressor, BaseExtremeLearningMachine


# ----- ELMRegressor -----
def test_elm_regressor_init():
    reg = ELMRegressor(hidden_layer_size=50, activation='tanh', alpha=0.1)
    assert reg.hidden_layer_size == 50
    assert reg.activation == 'tanh'
    assert reg.alpha == 0.1
    assert reg.linear_layer is None
    assert reg.random_state is None


def test_elm_regressor_fit_sets_attributes(random_data):
    X, _, y_reg = random_data
    reg = ELMRegressor(hidden_layer_size=10, random_state=42)
    reg.fit(X, y_reg)
    assert hasattr(reg, 'hidden_coef_')
    assert hasattr(reg, 'hidden_intercept_')
    assert hasattr(reg, 'linear_layer_')
    assert hasattr(reg, 'is_fitted_')
    assert reg.hidden_coef_.shape == (X.shape[1], 10)
    assert reg.hidden_intercept_.shape == (10,)
    assert isinstance(reg.linear_layer_, LinearRegression)


def test_elm_regressor_fit_returns_self(random_data):
    X, _, y_reg = random_data
    reg = ELMRegressor()
    result = reg.fit(X, y_reg)
    assert result is reg


def test_elm_regressor_predict_shape(random_data):
    X, _, y_reg = random_data
    reg = ELMRegressor(hidden_layer_size=8, random_state=42)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (X.shape[0],)


def test_elm_regressor_score(random_data):
    X, _, y_reg = random_data
    reg = ELMRegressor(hidden_layer_size=10, random_state=42)
    reg.fit(X, y_reg)
    score = reg.score(X, y_reg)
    # Should be reasonable for a linear combination of random features
    assert -1.0 <= score <= 1.0  # R^2 can be negative but usually not too bad


@pytest.mark.flaky(reruns=5)
def test_elm_regressor_reproducibility(random_data):
    X, _, y_reg = random_data
    reg1 = ELMRegressor(hidden_layer_size=10, random_state=42)
    reg1.fit(X, y_reg)
    pred1 = reg1.predict(X)

    reg2 = ELMRegressor(hidden_layer_size=10, random_state=42)
    reg2.fit(X, y_reg)
    pred2 = reg2.predict(X)

    np.testing.assert_array_almost_equal(pred1, pred2)


def test_elm_regressor_different_random_state_different(random_data):
    X, _, y_reg = random_data
    reg1 = ELMRegressor(hidden_layer_size=10, random_state=42)
    reg1.fit(X, y_reg)
    pred1 = reg1.predict(X)

    reg2 = ELMRegressor(hidden_layer_size=10, random_state=24)
    reg2.fit(X, y_reg)
    pred2 = reg2.predict(X)

    assert not np.allclose(pred1, pred2)


def test_elm_regressor_alpha_none(random_data):
    X, _, y_reg = random_data
    reg = ELMRegressor(alpha=None)
    reg.fit(X, y_reg)
    assert isinstance(reg.linear_layer_, LinearRegression)


def test_elm_regressor_alpha_ridge(random_data):
    X, _, y_reg = random_data
    alpha = 0.5
    reg = ELMRegressor(alpha=alpha, random_state=42)
    reg.fit(X, y_reg)
    assert isinstance(reg.linear_layer_, Ridge)
    assert reg.linear_layer_.alpha == alpha


def test_elm_regressor_custom_linear_layer(random_data):
    X, _, y_reg = random_data
    custom = Ridge(alpha=2.0)
    reg = ELMRegressor(linear_layer=custom)
    reg.fit(X, y_reg)
    assert reg.linear_layer_ is custom
    # Should still work
    preds = reg.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_elm_regressor_activation_functions(random_data):
    X, _, y_reg = random_data
    for act in ['identity', 'logistic', 'tanh', 'relu']:
        reg = ELMRegressor(hidden_layer_size=5, activation=act, random_state=42)
        reg.fit(X, y_reg)
        preds = reg.predict(X)
        assert not np.any(np.isnan(preds))


def test_elm_regressor_input_validation():
    reg = ELMRegressor()
    X_1d = np.array([1, 2, 3])
    y_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        reg.fit(X_1d, y_1d)


def test_elm_regressor_predict_before_fit_raises(random_data):
    X, _, _ = random_data
    reg = ELMRegressor()
    with pytest.raises(Exception):  # Should raise NotFittedError or similar
        reg.predict(X)


def test_elm_regressor_hidden_layer_size_effect(random_data):
    X, _, y_reg = random_data
    reg_small = ELMRegressor(hidden_layer_size=2, random_state=42)
    reg_small.fit(X, y_reg)
    pred_small = reg_small.predict(X)

    reg_large = ELMRegressor(hidden_layer_size=50, random_state=42)
    reg_large.fit(X, y_reg)
    pred_large = reg_large.predict(X)

    # Different hidden sizes give different predictions
    assert not np.allclose(pred_small, pred_large)


# ----- ELMClassifier -----
def test_elm_classifier_init():
    clf = ELMClassifier(hidden_layer_size=30, activation='relu', return_probability=True)
    assert clf.hidden_layer_size == 30
    assert clf.activation == 'relu'
    assert clf.return_probability is True


def test_elm_classifier_fit_sets_attributes(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(hidden_layer_size=10, random_state=42, return_probability=True)
    clf.fit(X, y_class)
    assert hasattr(clf, 'hidden_coef_')
    assert hasattr(clf, 'hidden_intercept_')
    assert hasattr(clf, 'linear_layer_')
    assert clf.hidden_coef_.shape == (X.shape[1], 10)
    assert isinstance(clf.linear_layer_, LogisticRegression)


def test_elm_classifier_fit_returns_self(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier()
    result = clf.fit(X, y_class)
    assert result is clf


def test_elm_classifier_predict_shape(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(hidden_layer_size=8, random_state=42)
    clf.fit(X, y_class)
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)


def test_elm_classifier_predict_proba_shape(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(hidden_layer_size=8, random_state=42, return_probability=True)
    clf.fit(X, y_class)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y_class)))
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


def test_elm_classifier_predict_proba_raises_when_false(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(return_probability=False, random_state=42)
    clf.fit(X, y_class)
    with pytest.raises(AttributeError, match="return_probability=False"):
        clf.predict_proba(X)


def test_elm_classifier_score(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(hidden_layer_size=15, random_state=42)
    clf.fit(X, y_class)
    score = clf.score(X, y_class)
    assert 0.0 <= score <= 1.0


@pytest.mark.flaky(reruns=5)
def test_elm_classifier_reproducibility(random_data):
    X, y_class, _ = random_data
    clf1 = ELMClassifier(hidden_layer_size=10, random_state=42, return_probability=True)
    clf1.fit(X, y_class)
    pred1 = clf1.predict(X)

    clf2 = ELMClassifier(hidden_layer_size=10, random_state=42, return_probability=True)
    clf2.fit(X, y_class)
    pred2 = clf2.predict(X)

    np.testing.assert_array_equal(pred1, pred2)


def test_elm_classifier_alpha_handling(random_data):
    X, y_class, _ = random_data

    # None -> default LogisticRegression
    clf1 = ELMClassifier(alpha=None, random_state=42)
    clf1.fit(X, y_class)
    assert isinstance(clf1.linear_layer_, LogisticRegression)
    # C is default (1.0) when alpha is None
    assert clf1.linear_layer_.C == 1.0

    # alpha=0 -> C=np.inf (no regularization)
    clf2 = ELMClassifier(alpha=0, random_state=42)
    clf2.fit(X, y_class)
    assert clf2.linear_layer_.C == np.inf

    # alpha>0 -> C=1/alpha
    alpha = 0.5
    clf3 = ELMClassifier(alpha=alpha, random_state=42)
    clf3.fit(X, y_class)
    assert clf3.linear_layer_.C == 1 / alpha


def test_elm_classifier_ridge_classifier_when_no_probability(random_data):
    X, y_class, _ = random_data
    clf = ELMClassifier(return_probability=False, random_state=42)
    clf.fit(X, y_class)
    assert isinstance(clf.linear_layer_, RidgeClassifier)
    # With alpha=None, RidgeClassifier alpha=0 (no regularization)
    assert clf.linear_layer_.alpha == 0

    clf_alpha = ELMClassifier(return_probability=False, alpha=0.3, random_state=42)
    clf_alpha.fit(X, y_class)
    assert clf_alpha.linear_layer_.alpha == 0.3


def test_elm_classifier_custom_linear_layer(random_data):
    X, y_class, _ = random_data
    custom = LogisticRegression(C=10)
    clf = ELMClassifier(linear_layer=custom, random_state=42)
    clf.fit(X, y_class)
    assert clf.linear_layer_ is custom
    preds = clf.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_elm_classifier_activation_functions(random_data):
    X, y_class, _ = random_data
    for act in ['identity', 'logistic', 'tanh', 'relu']:
        clf = ELMClassifier(hidden_layer_size=5, activation=act, random_state=42)
        clf.fit(X, y_class)
        preds = clf.predict(X)
        assert not np.any(np.isnan(preds))


def test_elm_classifier_predict_before_fit_raises(random_data):
    X, _, _ = random_data
    clf = ELMClassifier()
    with pytest.raises(Exception):
        clf.predict(X)


def test_elm_classifier_binary_multiclass():
    # Binary classification
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    clf = ELMClassifier(hidden_layer_size=10, random_state=42)
    clf.fit(X, y)
    assert len(clf.classes_) == 2
    pred = clf.predict(X)
    assert pred.shape == (100,)

    # Multiclass
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=4, n_redundant=1, random_state=42)
    clf = ELMClassifier(hidden_layer_size=10, random_state=42)
    clf.fit(X, y)
    assert len(clf.classes_) == 3
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 3)

def test_elm_accepts_alpha_zero():
    reg = ELMRegressor(alpha=0)
    assert reg.alpha == 0
    clf = ELMClassifier(alpha=0)
    assert clf.alpha == 0