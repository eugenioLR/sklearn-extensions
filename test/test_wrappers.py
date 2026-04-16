import pytest
import numpy as np
from sklearn.linear_model import RidgeClassifier, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn_extensions.wrappers import MultiplexedRegressor

@pytest.fixture
def multi_data():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y_class = (X[:, 0] > 0).astype(int)  # binary
    y_reg = X[:, 0] * 2 + np.random.randn(100) * 0.1
    return X, y_class, y_reg

def test_multiplexed_fit_predict(multi_data):
    X, y_class, y_reg = multi_data
    regs = [LinearRegression(), LinearRegression()]
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=regs, classifier=clf)
    mr.fit(X, y_class, y_reg)
    regs_fitted = mr.regressors_
    preds = mr.predict(X)
    assert preds.shape == (X.shape[0],)
    # Check that each regressor was fitted on its class
    mask = mr.predict_class(X)
    # Predict on class0 points should match regressor0's prediction
    pred0 = regs_fitted[0].predict(X[mask == 0])
    np.testing.assert_allclose(preds[mask == 0], pred0, rtol=1e-6)
    pred1 = regs_fitted[1].predict(X[mask == 1])
    np.testing.assert_allclose(preds[mask == 1], pred1, rtol=1e-6)

def test_multiplexed_predict_class(multi_data):
    X, y_class, _ = multi_data
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=[LinearRegression()]*2, classifier=clf)
    mr.fit(X, y_class, np.zeros(X.shape[0]))  # dummy y_reg
    clf_fitted = mr.classifier_
    pred_class = mr.predict_class(X)
    assert pred_class.shape == (X.shape[0],)
    # Should match classifier's prediction
    clf_pred = clf_fitted.predict(X)
    np.testing.assert_array_equal(pred_class, clf_pred)

def test_multiplexed_score_methods(multi_data):
    X, y_class, y_reg = multi_data
    regs = [LinearRegression(), LinearRegression()]
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=regs, classifier=clf)
    mr.fit(X, y_class, y_reg)
    clf_fitted = mr.classifier_
    # score_class
    acc = mr.score_class(X, y_class)
    expected_acc = accuracy_score(y_class, clf_fitted.predict(X))
    assert acc == expected_acc
    # score (R2)
    r2 = mr.score(X, y_reg)
    preds = mr.predict(X)
    expected_r2 = r2_score(y_reg, preds)
    assert r2 == expected_r2
    # score_report
    report = mr.score_report(X, y_reg)
    assert set(report.keys()) == {"R2", "RMSE", "MAE"}

def test_multiplexed_fit_on_predictions(multi_data):
    X, y_class, y_reg = multi_data
    regs = [LinearRegression(), LinearRegression()]
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=regs, classifier=clf, fit_on_predictions=True)
    mr.fit(X, y_class, y_reg)
    regs_fitted = mr.regressors_
    # Check that regressors were fitted using classifier's predictions
    pred_class = clf.predict(X)
    mask0 = pred_class == 0
    mask1 = pred_class == 1
    # Manually fit regressors on those splits and compare coefs
    reg0_manual = LinearRegression().fit(X[mask0], y_reg[mask0])
    reg1_manual = LinearRegression().fit(X[mask1], y_reg[mask1])
    np.testing.assert_allclose(regs_fitted[0].coef_, reg0_manual.coef_)
    np.testing.assert_allclose(regs_fitted[1].coef_, reg1_manual.coef_)