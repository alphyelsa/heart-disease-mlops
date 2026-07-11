import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pytest


@pytest.mark.parametrize("model", [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=10, random_state=42)
])
def test_model_training_and_prediction(model):
    X, y = make_classification(
        n_samples=50,
        n_features=13,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert not np.isnan(predictions).any()
    assert set(predictions).issubset({0, 1})