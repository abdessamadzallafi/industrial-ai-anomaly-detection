import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from pyod.models.iforest import IForest

DATA_PATH = "data/raw/ai4i2020.csv"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

RANDOM_STATE = 42

# 1. Load dataset
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

X = df[SENSOR_COLUMNS]
y = df["Machine failure"]

# 2. Split data before model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 3. Configure and train Isolation Forest
# The contamination value is estimated from the training labels.
# Labels are NOT passed to fit().
contamination = float(y_train.mean())

model = IForest(
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_estimators=200,
)

model.fit(X_train)

# 4. Predict anomalies on unseen test data
predictions = model.predict(X_test)

# PyOD: 0 = normal, 1 = anomaly
# Ground truth: 0 = normal, 1 = machine failure
scores = model.decision_function(X_test)

# 5. Evaluate
metrics = {
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "contamination": contamination,
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, scores)),
    "average_precision": float(average_precision_score(y_test, scores)),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}

print("\n========== MODEL EVALUATION ==========")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Failure"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nMetrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# 6. Save metrics
with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to results/metrics.json")
