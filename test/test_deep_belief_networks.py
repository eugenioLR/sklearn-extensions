import pytest
import numpy as np
from sklearn_extensions.models.hopfield import BernoulliHopfieldNetwork, BernoulliBoltzmannMachine, BernoulliDBN


@pytest.fixture
def X():
    rng = np.random.RandomState(42)
    return rng.binomial(1, 0.5, size=(100, 20)).astype(np.float64)


@pytest.fixture
def X_small():
    rng = np.random.RandomState(42)
    return rng.binomial(1, 0.5, size=(10, 5)).astype(np.float64)


def test_dbn_init_default_params():
    dbn = BernoulliDBN()
    assert dbn.hidden_layer_sizes == (256, 256)
    assert dbn.learning_rate == 1e-3
    assert dbn.batch_size == 10
    assert dbn.n_iter == 10
    assert dbn.verbose is False
    assert dbn.random_state is None
    assert dbn.layers_ is None


def test_dbn_init_custom_params():
    dbn = BernoulliDBN(
        hidden_layer_sizes=(100, 50),
        learning_rate=0.01,
        batch_size=20,
        n_iter=5,
        verbose=True,
        random_state=123,
    )
    assert dbn.hidden_layer_sizes == (100, 50)
    assert dbn.learning_rate == 0.01
    assert dbn.batch_size == 20
    assert dbn.n_iter == 5
    assert dbn.verbose is True
    assert dbn.random_state == 123


def test_dbn_fit_sets_attributes(X_small):
    dbn = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn.fit(X_small)
    assert hasattr(dbn, "is_fitted_")
    assert dbn.is_fitted_ is True
    assert hasattr(dbn, "n_features_in_")
    assert dbn.n_features_in_ == X_small.shape[1]
    assert hasattr(dbn, "layers_")
    assert dbn.layers_ is not None
    assert len(dbn.layers_) == len(dbn.hidden_layer_sizes)


def test_dbn_fit_returns_self(X_small):
    dbn = BernoulliDBN()
    result = dbn.fit(X_small)
    assert result is dbn


def test_dbn_transform_before_fit_raises(X):
    dbn = BernoulliDBN()
    with pytest.raises(Exception):
        dbn.transform(X)


def test_dbn_transform_output_shape(X_small):
    dbn = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn.fit(X_small)
    X_trans = dbn.transform(X_small)
    assert X_trans.shape == (X_small.shape[0], 4)


def test_dbn_transform_output_shape_single_layer(X_small):
    dbn = BernoulliDBN(hidden_layer_sizes=(10,), random_state=42)
    dbn.fit(X_small)
    X_trans = dbn.transform(X_small)
    assert X_trans.shape == (X_small.shape[0], 10)


def test_dbn_transform_values_range(X_small):
    dbn = BernoulliDBN(hidden_layer_sizes=(5,), random_state=42)
    dbn.fit(X_small)
    X_trans = dbn.transform(X_small)
    assert np.all(X_trans >= 0) and np.all(X_trans <= 1)


def test_dbn_fit_transform_equivalence(X_small):
    dbn1 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn1.fit(X_small)
    trans1 = dbn1.transform(X_small)

    dbn2 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn2.fit(X_small)
    trans2 = dbn2.transform(X_small)

    np.testing.assert_array_almost_equal(trans1, trans2)


def test_dbn_random_state_reproducibility(X_small):
    dbn1 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn2 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)

    trans1 = dbn1.fit_transform(X_small)
    trans2 = dbn2.fit_transform(X_small)

    np.testing.assert_array_almost_equal(trans1, trans2)


def test_dbn_different_random_state_different_results(X_small):
    dbn1 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn2 = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=24)

    trans1 = dbn1.fit_transform(X_small)
    trans2 = dbn2.fit_transform(X_small)

    assert not np.allclose(trans1, trans2)


def test_dbn_transform_on_new_data(X):
    dbn = BernoulliDBN(hidden_layer_sizes=(10,), random_state=42)
    X_train = X[:80]
    X_test = X[80:]
    dbn.fit(X_train)
    X_trans = dbn.transform(X_test)
    assert X_trans.shape[0] == X_test.shape[0]


def test_dbn_input_validation_accepts_2d_array(X_small):
    dbn = BernoulliDBN()
    dbn.fit(X_small)
    dbn.transform(X_small)


def test_dbn_input_validation_rejects_1d():
    dbn = BernoulliDBN()
    X_1d = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        dbn.fit(X_1d)


def test_dbn_fit_with_empty_array():
    dbn = BernoulliDBN()
    X_empty = np.empty((0, 10))
    with pytest.raises(ValueError):
        dbn.fit(X_empty)


def test_dbn_fit_with_single_feature():
    dbn = BernoulliDBN(hidden_layer_sizes=(4,))
    X = np.random.binomial(1, 0.5, size=(20, 1))
    dbn.fit(X)
    assert dbn.n_features_in_ == 1
    assert dbn.transform(X).shape == (20, 4)


def test_dbn_multiple_layers_order(X_small):
    dbn = BernoulliDBN(hidden_layer_sizes=(8, 4), random_state=42)
    dbn.fit(X_small)

    X_manual = X_small.copy()
    for rbm in dbn.layers_:
        X_manual = rbm.transform(X_manual)

    X_dbn = dbn.transform(X_small)
    np.testing.assert_array_almost_equal(X_dbn, X_manual)