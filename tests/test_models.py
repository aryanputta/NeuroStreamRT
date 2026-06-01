"""Tests for sklearn EEG classifiers and (optionally) DL models."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from models.sklearn_models import MODEL_REGISTRY, get_model

N_CLASSES = 3
BATCH = 20
N_FEATURES = 19 * 5  # 19 channels * 5 bands


class TestSklearnModels:
    """Test all three sklearn classifiers end-to-end."""

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_fit_predict_shape(self, model_name: str):
        model = get_model(model_name)
        X = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = np.array([i % N_CLASSES for i in range(BATCH)])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (BATCH,)
        assert set(preds).issubset({0, 1, 2})

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_predict_proba_shape(self, model_name: str):
        model = get_model(model_name)
        X = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = np.array([i % N_CLASSES for i in range(BATCH)])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (BATCH, N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_deterministic(self, model_name: str):
        """Same data produces same predictions after fitting."""
        X = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = np.array([i % N_CLASSES for i in range(BATCH)])
        m1 = get_model(model_name)
        m1.fit(X, y)
        m2 = get_model(model_name)
        m2.fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_single_sample_predict(self, model_name: str):
        """Models must handle a single test sample (n=1) after fitting on multiple."""
        model = get_model(model_name)
        X = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = np.array([i % N_CLASSES for i in range(BATCH)])
        model.fit(X, y)
        single = X[:1]
        pred = model.predict(single)
        assert pred.shape == (1,)


# Optional: DL model tests (only if torch is installed)
try:
    import torch
    from models.eegnet import EEGNet
    from models.shallow_convnet import ShallowConvNet

    N_CH = 19
    N_SAMPLES = 512

    class TestEEGNet:
        def setup_method(self):
            self.model = EEGNet(n_channels=N_CH, n_samples=N_SAMPLES, n_classes=N_CLASSES)
            self.model.eval()

        def test_output_shape(self):
            x = torch.randn(8, N_CH, N_SAMPLES)
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (8, N_CLASSES)

        def test_param_count_small(self):
            n = sum(p.numel() for p in self.model.parameters())
            assert n < 10_000, f"EEGNet has {n} params"

    class TestShallowConvNet:
        def setup_method(self):
            self.model = ShallowConvNet(n_channels=N_CH, n_samples=N_SAMPLES, n_classes=N_CLASSES)
            self.model.eval()

        def test_output_shape(self):
            x = torch.randn(8, N_CH, N_SAMPLES)
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (8, N_CLASSES)

except ImportError:
    pass  # torch not installed, skip DL tests
