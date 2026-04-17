import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn_extensions.models.neural_network.rbfnn import RBFNNClassifier, RBFNNRegressor
from sklearn_extensions.models.pytorch.rbfnn import RBFLayer
import torch

# ----- RBFNNClassifier -----
def test_rbf_classifier_init():
    clf = RBFNNClassifier(n_units=5)
    assert clf.n_units == 5
    assert clf.classification is True
    assert clf.linear_layer is None

def test_rbf_classifier_fit_predict(random_data):
    X, y_class, _ = random_data
    clf = RBFNNClassifier(n_units=3, random_state=42)
    clf.fit(X, y_class)
    assert hasattr(clf, 'centers_')
    assert clf.centers_.shape == (3, X.shape[1])
    assert hasattr(clf, 'widths_')
    assert clf.widths_.shape == (3,)

    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)
    # Check that predict uses the linear layer
    assert hasattr(clf.linear_layer_, 'coef_')

def test_rbf_classifier_predict_proba(random_data):
    X, y_class, _ = random_data
    clf = RBFNNClassifier(n_units=3, return_probability=True, random_state=42)
    clf.fit(X, y_class)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)  # binary
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

def test_rbf_classifier_with_custom_linear(random_data):
    X, y_class, _ = random_data
    custom_linear = LogisticRegression(l1_ratio=0, C=1.0)
    clf = RBFNNClassifier(n_units=3, linear_layer=custom_linear)
    clf.fit(X, y_class)
    assert clf.linear_layer_ is custom_linear

def test_rbf_classifier_clustering_std(random_data):
    X, y_class, _ = random_data
    clf = RBFNNClassifier(n_units=3, std_from_clusters=True, random_state=42)
    clf.fit(X, y_class)
    # widths should be positive
    assert np.all(clf.widths_ > 0)

# ----- RBFNNRegressor -----
def test_rbf_regressor_init():
    reg = RBFNNRegressor(n_units=5)
    assert reg.n_units == 5
    assert reg.classification is False

def test_rbf_regressor_fit_predict(random_data):
    X, _, y_reg = random_data
    reg = RBFNNRegressor(n_units=3, random_state=42)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (X.shape[0],)
    assert hasattr(reg.linear_layer_, 'coef_')

def test_rbf_regressor_custom_cluster(random_data):
    X, _, y_reg = random_data
    custom_cluster = KMeans(n_clusters=4, random_state=0)
    reg = RBFNNRegressor(n_units=4, cluster_model=custom_cluster)
    reg.fit(X, y_reg)
    assert reg.cluster_model_ is custom_cluster
    assert reg.centers_.shape == (4, X.shape[1])

# ----- RBFLayer (PyTorch) -----
def test_rbf_layer_init():
    layer = RBFLayer(in_features=5, out_features=3)
    assert layer.centers.shape == (3, 5)
    assert layer.widths.shape == (3,)
    assert layer.centers.requires_grad is True
    assert layer.widths.requires_grad is True

def test_rbf_layer_forward():
    layer = RBFLayer(in_features=2, out_features=3)
    # Set fixed centers/widths for deterministic test
    with torch.no_grad():
        # layer.centers[:, :] = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        layer.centers[:, :] = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
        layer.widths[:] = torch.tensor([1., 1., 1.])
    x = torch.tensor([[1., 1.], [2., 2.]])
    out = layer.forward(x)
    # out shape: (2, 3)
    assert out.shape == (2, 3)
    # distances: row0: (0,0) -> 0, row1: (1,1) -> 2. So exp(-0)=1, exp(-2)=0.135
    expected = torch.tensor([[1., np.exp(-2), np.exp(-8)], [np.exp(-2), 1., np.exp(-2)]])
    assert torch.allclose(out.to(float), expected.to(float), rtol=1e-10)

def test_rbf_layer_initialize_clustering():
    X = np.random.randn(50, 3)
    layer = RBFLayer(in_features=3, out_features=4)
    layer.initialize_clustering(X, n_samples=30, width_init='random')
    assert layer.centers.shape == (4, 3)
    assert layer.widths.shape == (4,)
    # Check centers are from kmeans (approximate)
    # We can't guarantee exact but they should be within data range
    assert torch.all(layer.centers >= X.min()) and torch.all(layer.centers <= X.max())

    # Test width init methods
    layer.initialize_clustering(X, width_init='std')
    assert torch.all(layer.widths > 0)
    layer.initialize_clustering(X, width_init='maxdist')
    assert torch.all(layer.widths > 0)

    with pytest.raises(Exception):
        layer.initialize_clustering(X, width_init='unknown')