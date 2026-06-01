"""
Sklearn-based classifiers for EEG band-power features.

Three classifiers:
  svm_rbf     — RBF SVM, strong baseline for EEG classification
  random_forest — 200-tree RF, robust to feature correlations
  mlp_sklearn  — 3-layer MLP via sklearn, matches MLPBaseline architecture

These are exported to ONNX via skl2onnx for deployment benchmarking.

All models are trained with LOSO (Leave-One-Subject-Out) CV.
Feature input: band-power vectors (n_channels * n_bands = 19*5 = 95 features)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_CLASSES = 3


def build_svm_rbf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_mlp_sklearn() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            batch_size=64,
            learning_rate_init=1e-3,
        )),
    ])


MODEL_REGISTRY = {
    "svm_rbf": build_svm_rbf,
    "random_forest": build_random_forest,
    "mlp_sklearn": build_mlp_sklearn,
}


def get_model(name: str) -> Pipeline:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
