import numpy as np
from pyod.models.iforest import IForest


def test_isolation_forest_detects_anomalies():
    rng = np.random.default_rng(42)

    normal_data = rng.normal(
        loc=0,
        scale=1,
        size=(200, 2),
    )

    anomalies = rng.uniform(
        low=6,
        high=9,
        size=(10, 2),
    )

    X = np.vstack([normal_data, anomalies])

    model = IForest(
        contamination=10 / 210,
        n_estimators=100,
        random_state=42,
    )

    model.fit(X)
    predictions = model.predict(X)

    assert len(predictions) == 210
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.sum(predictions == 1) > 0
    assert np.sum(predictions[200:] == 1) >= 5
