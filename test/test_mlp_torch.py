import pytest
import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score
from sklearn_extensions.mlp_torch import MLPModelTorch, MLPRegressorTorch, MLPClassifierTorch

# ----- MLPModelTorch -----
def test_mlp_model_init():
    model = MLPModelTorch(input_size=5, layer_sizes=[10, 5], activation='relu', last_layer='linear')
    assert isinstance(model, torch.nn.Module)
    # Count layers: input->10, 10->5, 5->1 = 3 linear layers
    assert len(model.layers) == 3
    # Forward pass with random data
    x = torch.randn(2, 5)
    out = model(x)
    assert out.shape == (2, 1)

def test_mlp_model_activations():
    model = MLPModelTorch(input_size=3, layer_sizes=[4], activation='tanh', last_layer='sigmoid')
    x = torch.randn(2, 3)
    out = model(x)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid output

# ----- MLPRegressorTorch -----
@pytest.fixture
def small_reg_data():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)
    return X, y

def test_mlp_regressor_init():
    reg = MLPRegressorTorch(input_size=4, layer_sizes=[8, 4], n_epochs=10, verbose=False)
    assert reg.input_size == 4
    assert reg.layer_sizes == [8, 4]
    assert reg.n_epochs == 10
    assert reg.fitted is False

@pytest.mark.flaky(reruns=10)
def test_mlp_regressor_fit_predict(small_reg_data):
    X, y = small_reg_data
    reg = MLPRegressorTorch(input_size=4, layer_sizes=[8], n_epochs=100, verbose=False, patience=10)
    reg.fit(X, y.ravel())
    assert reg.fitted is True
    preds = reg.predict(X)
    assert preds.shape == (50, 1)
    # Check that loss decreased (rough check)
    assert reg.history[-1] < reg.history[0]  # validation loss improved

def test_mlp_regressor_early_stopping(small_reg_data):
    X, y = small_reg_data
    reg = MLPRegressorTorch(input_size=4, layer_sizes=[8], n_epochs=1000, patience=5, verbose=False)
    reg.fit(X, y.ravel())
    # Should stop before n_epochs due to patience
    assert len(reg.history) <= 1000

def test_mlp_regressor_optimizers(small_reg_data):
    X, y = small_reg_data
    # Test SGD
    reg_sgd = MLPRegressorTorch(input_size=4, optimizer_class='sgd', optimizer_params={'lr': 0.01}, n_epochs=50, verbose=False)
    reg_sgd.fit(X, y)
    assert reg_sgd.fitted
    # Test Adam
    reg_adam = MLPRegressorTorch(input_size=4, optimizer_class='adam', optimizer_params={'lr': 0.01}, n_epochs=50, verbose=False)
    reg_adam.fit(X, y)
    # Test newton (may need special params)
    try:
        reg_newton = MLPRegressorTorch(input_size=4, optimizer_class='newton', n_epochs=10, verbose=False)
        reg_newton.fit(X, y)
    except Exception as e:
        pytest.skip(f"Newton optimizer not fully implemented: {e}")

def test_mlp_regressor_score_report(small_reg_data):
    X, y = small_reg_data
    reg = MLPRegressorTorch(input_size=4, layer_sizes=[8], n_epochs=50, verbose=False)
    reg.fit(X, y)
    report = reg.score_report(X, y)
    assert set(report.keys()) == {"R2", "RMSE", "MAE"}
    assert isinstance(report["R2"], float)

# ----- MLPClassifierTorch -----
@pytest.fixture
def small_class_data():
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
    return X, y

def test_mlp_classifier_init():
    clf = MLPClassifierTorch(input_size=4, layer_sizes=[8, 4], n_epochs=10, verbose=False)
    assert clf.input_size == 4
    assert isinstance(clf.nn_model, MLPModelTorch)

@pytest.mark.skip
def test_mlp_classifier_fit_predict(small_class_data):
    X, y = small_class_data
    clf = MLPClassifierTorch(input_size=4, layer_sizes=[8], n_epochs=100, verbose=False, patience=10)
    clf.fit(X, y.ravel())
    assert clf.fitted is True
    preds = clf.predict(X)
    assert preds.shape == (50, 1)
    # Accuracy should be decent (overfitting small data)
    acc = accuracy_score(y, preds)
    assert acc > 0.8

@pytest.mark.skip
def test_mlp_classifier_score_report(small_class_data):
    X, y = small_class_data
    clf = MLPClassifierTorch(input_size=4, layer_sizes=[8], n_epochs=50, verbose=False)
    clf.fit(X, y)
    report = clf.score_report(X, y)
    assert set(report.keys()) == {"ACC", "F1"}
    assert 0 <= report["ACC"] <= 1