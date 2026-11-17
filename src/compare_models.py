import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

DATA_PATH = "data/raw/ai4i2020.csv"
RANDOM_STATE = 42

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
X = df[SENSOR_COLUMNS]
y = df["Machine failure"]

# Train / validation / test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_temp
)

# Models
models = {
    "Isolation Forest": IForest(
        contamination=float(y_train.mean()),
        n_estimators=200,
        random_state=RANDOM_STATE,
    ),
    "KNN": KNN(
        contamination=float(y_train.mean()),
        n_neighbors=10,
    ),
    "LOF": LOF(
        contamination=float(y_train.mean()),
        n_neighbors=20,
    ),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train without failure labels
    model.fit(X_train)

    # Get anomaly scores
    val_scores = model.decision_function(X_val)
    test_scores = model.decision_function(X_test)

    # Select threshold using validation data only
    candidates = pd.Series(val_scores).quantile(
        [i / 1000 for i in range(500, 1000)]
    ).unique()

    best_threshold = float(model.threshold_)
    best_f1 = -1

    for threshold in candidates:
        val_predictions = (val_scores > threshold).astype(int)
        score = f1_score(
            y_val, val_predictions, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    # Evaluate on untouched test data
    predictions = (test_scores > best_threshold).astype(int)

    results[name] = {
        "threshold": best_threshold,
        "validation_f1": float(best_f1),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_scores)
        ),
        "average_precision": float(
            average_precision_score(y_test, test_scores)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1-score: {results[name]['f1_score']:.4f}")
    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
    print(
        "Average Precision: "
        f"{results[name]['average_precision']:.4f}"
    )
    print("Confusion Matrix:")
    print(results[name]["confusion_matrix"])

with open("results/model_comparison.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nResults saved to results/model_comparison.json")
