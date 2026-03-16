def test_top_level_imports():
    import sklearn_extensions
    assert hasattr(sklearn_extensions, 'hopfield')
    assert hasattr(sklearn_extensions, 'preprocessing')
    assert hasattr(sklearn_extensions, 'rbf_networks')
    assert hasattr(sklearn_extensions, 'mlp_torch')
    assert hasattr(sklearn_extensions, 'wrappers')

def test_model_zoo_imports():
    from sklearn_extensions import model_zoo
    # Just check that a few models are present
    assert hasattr(model_zoo, 'RandomForestClassifier')
    assert hasattr(model_zoo, 'LinearRegression')
    assert hasattr(model_zoo, 'KMeans')