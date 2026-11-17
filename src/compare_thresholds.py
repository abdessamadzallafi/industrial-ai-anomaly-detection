import json
import numpy as np
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

DATA_PATH = "data/raw/ai4i2020.csv"
RANDOM_STATE = 42

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Load data
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
X = df[SENSOR_COLUMNS]
y = df["Machine failure"]

# 60% training, 20% validation, 20% final test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_temp
)

# Fit model using training sensor data only
model = IForest(
    contamination=float(y_train.mean()),
    n_estimators=200,
    random_state=RANDOM_STATE,
)
model.fit(X_train)

# Higher scores mean more anomalous
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

# Original PyOD threshold
default_threshold = float(model.threshold_)

# Select threshold using validation labels only
candidate_thresholds = np.unique(
    np.quantile(val_scores, np.linspace(0.50, 0.999, 500))
)

best_threshold = default_threshold
best_f1 = -1

for threshold in candidate_thresholds:
    val_predictions = (val_scores > threshold).astype(int)
    score = f1_score(y_val, val_predictions, zero_division=0)

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

def evaluate(name, threshold):
    predictions = (test_scores > threshold).astype(int)

    return {
        "method": name,
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

results = {
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "roc_auc": float(roc_auc_score(y_test, test_scores)),
    "average_precision": float(
        average_precision_score(y_test, test_scores)
    ),
    "validation_best_f1": float(best_f1),
    "default_threshold": evaluate(
        "PyOD default", default_threshold
    ),
    "calibrated_threshold": evaluate(
        "Validation calibrated", best_threshold
    ),
}

print("\n========== THRESHOLD COMPARISON ==========")
print(json.dumps(results, indent=4))

with open("results/threshold_comparison.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nSaved to results/threshold_comparison.json")
