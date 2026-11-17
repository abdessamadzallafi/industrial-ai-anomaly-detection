import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "model_comparison.json") as f:
    original = json.load(f)

with open(RESULTS_DIR / "model_comparison_scaled.json") as f:
    scaled = json.load(f)

models = [
    ("Isolation Forest", original["Isolation Forest"]),
    ("KNN", original["KNN"]),
    ("LOF", original["LOF"]),
    ("Isolation Forest (Scaled)", scaled["Isolation Forest"]),
    ("KNN (Scaled)", scaled["KNN"]),
    ("LOF (Scaled)", scaled["LOF"]),
]

labels = [item[0] for item in models]
metrics = ["precision", "recall", "f1_score"]
metric_labels = ["Precision", "Recall", "F1-score"]

# Plot 1: Precision, Recall and F1-score
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))

for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    values = [item[1][metric] for item in models]
    ax.bar(x + (i - 1) * width, values, width, label=metric_label)

ax.set_ylabel("Score")
ax.set_title("Model Comparison: Precision, Recall and F1-score")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "model_metrics_comparison.png", dpi=300)
plt.close(fig)

# Plot 2: ROC-AUC and Average Precision
metrics = ["roc_auc", "average_precision"]
metric_labels = ["ROC-AUC", "Average Precision"]

fig, ax = plt.subplots(figsize=(13, 6))

for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    values = [item[1][metric] for item in models]
    ax.bar(x + (i - 0.5) * 0.35, values, 0.35, label=metric_label)

ax.set_ylabel("Score")
ax.set_title("Model Comparison: ROC-AUC and Average Precision")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ranking_metrics_comparison.png", dpi=300)
plt.close(fig)

# Plot 3: Confusion matrix for scaled KNN
cm = np.array(scaled["KNN"]["confusion_matrix"])

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"],
    xlabel="Predicted label",
    ylabel="True label",
    title="Confusion Matrix - KNN with StandardScaler",
)

threshold = cm.max() / 2

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
        )

fig.tight_layout()
fig.savefig(PLOTS_DIR / "knn_scaled_confusion_matrix.png", dpi=300)
plt.close(fig)

print("Plots generated successfully:")
for path in sorted(PLOTS_DIR.glob("*.png")):
    print(f"- {path}")
