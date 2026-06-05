import numpy as np

from physionet_mi.baselines.riemannian import RiemannTangentLRClassifier, _trial_covariances


def test_covariance_shape():
    X = np.random.randn(8, 4, 50).astype(np.float32)
    covs = _trial_covariances(X)
    assert covs.shape == (8, 4, 4)


def test_tangent_lr_fit_predict():
    X = np.random.randn(12, 4, 40).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    clf = RiemannTangentLRClassifier()
    clf.fit(X, y)
    pred = clf.predict(X[:4])
    assert pred.shape == (4,)
