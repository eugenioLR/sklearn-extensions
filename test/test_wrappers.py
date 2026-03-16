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
    preds = mr.predict(X)
    assert preds.shape == (X.shape[0],)
    # Check that each regressor was fitted on its class
    mask0 = y_class == 0
    mask1 = y_class == 1
    # Predict on class0 points should match regressor0's prediction
    pred0 = regs[0].predict(X[mask0])
    np.testing.assert_allclose(preds[mask0], pred0, rtol=1e-10)
    pred1 = regs[1].predict(X[mask1])
    np.testing.assert_allclose(preds[mask1], pred1, rtol=1e-10)

def test_multiplexed_predict_class(multi_data):
    X, y_class, _ = multi_data
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=[LinearRegression()]*2, classifier=clf)
    mr.fit(X, y_class, np.zeros(X.shape[0]))  # dummy y_reg
    pred_class = mr.predict_class(X)
    assert pred_class.shape == (X.shape[0],)
    # Should match classifier's prediction
    clf_pred = clf.predict(X)
    np.testing.assert_array_equal(pred_class, clf_pred)

def test_multiplexed_score_methods(multi_data):
    X, y_class, y_reg = multi_data
    regs = [LinearRegression(), LinearRegression()]
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=regs, classifier=clf)
    mr.fit(X, y_class, y_reg)
    # score_class
    acc = mr.score_class(X, y_class)
    expected_acc = accuracy_score(y_class, clf.predict(X))
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
    # Check that regressors were fitted using classifier's predictions
    pred_class = clf.predict(X)
    mask0 = pred_class == 0
    mask1 = pred_class == 1
    # Manually fit regressors on those splits and compare coefs
    reg0_manual = LinearRegression().fit(X[mask0], y_reg[mask0])
    reg1_manual = LinearRegression().fit(X[mask1], y_reg[mask1])
    np.testing.assert_allclose(regs[0].coef_, reg0_manual.coef_)
    np.testing.assert_allclose(regs[1].coef_, reg1_manual.coef_)

def test_multiplexed_empty_class(multi_data):
    X, y_class, y_reg = multi_data
    # Make class 1 empty
    y_class[:] = 0
    regs = [LinearRegression(), LinearRegression()]
    clf = RidgeClassifier()
    mr = MultiplexedRegressor(regressors=regs, classifier=clf)
    # Should still fit without error (regressor1 gets no data)
    mr.fit(X, y_class, y_reg)
    preds = mr.predict(X)
    # When predicting, class1 points won't appear, but if they do (unlikely because classifier predicts only 0),
    # then regressor1 predict may be called on empty? Our code handles mask.
    # Just check no crash
    assert preds.shape == (X.shape[0],)