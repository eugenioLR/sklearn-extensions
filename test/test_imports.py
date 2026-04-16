def test_top_level_imports():
    import sklearn_extensions
    assert hasattr(sklearn_extensions, 'preprocessing')
    assert hasattr(sklearn_extensions, 'wrappers')
    assert hasattr(sklearn_extensions, 'model_zoo')
    assert hasattr(sklearn_extensions, 'models')

    from sklearn_extensions import models
    assert hasattr(models, 'elm')
    assert hasattr(models, 'hopfield')
    assert hasattr(models, 'mlp_torch')
    assert hasattr(models, 'rbfnn')


def test_model_zoo_imports():
    from sklearn_extensions import model_zoo
    # Just check that a few models are present
    assert hasattr(model_zoo, 'RandomForestClassifier')
    assert hasattr(model_zoo, 'LinearRegression')
    assert hasattr(model_zoo, 'KMeans')