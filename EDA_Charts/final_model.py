import csv
import os
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

INPUT = "Student_Performance_ML_Ready.csv"

MODEL_OUTPUT = "final_gradient_boosting_model.joblib"
RESULTS_OUTPUT = "final_model_results.txt"


TARGET = "final_grade_encoded"


# ============================================================
# Load data
# ============================================================

print("Loading ML-ready dataset...")

with open(INPUT, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames


if not rows:
    raise ValueError("Dataset is empty.")


print(f"Rows loaded: {len(rows)}")


# ============================================================
# Prepare X and y
# ============================================================

feature_columns = [
    h for h in headers
    if h not in [TARGET, "final_grade"]
]


X = []

for row in rows:
    X.append([
        float(row[col])
        for col in feature_columns
    ])


y = [
    int(row[TARGET])
    for row in rows
]


print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")


# ============================================================
# Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# Final Gradient Boosting model
# Tuned parameters from hyperparameter tuning
# ============================================================

model = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)


print("\nTraining final Gradient Boosting model...")

model.fit(X_train, y_train)


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n==============================")
print("FINAL MODEL RESULTS")
print("==============================")

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# Save model
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "target": TARGET
    },
    MODEL_OUTPUT
)


print(f"\nSaved model: {MODEL_OUTPUT}")


# ============================================================
# Save results
# ============================================================

with open(RESULTS_OUTPUT, "w", encoding="utf-8") as f:

    f.write("FINAL GRADIENT BOOSTING MODEL\n")
    f.write("=============================\n\n")

    f.write("Model: Gradient Boosting\n\n")

    f.write("Hyperparameters:\n")
    f.write("learning_rate = 0.05\n")
    f.write("max_depth = 3\n")
    f.write("n_estimators = 200\n")
    f.write("subsample = 0.8\n")
    f.write("random_state = 42\n\n")

    f.write("Features:\n")
    for feature in feature_columns:
        f.write(f"- {feature}\n")

    f.write("\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"Macro F1  : {macro_f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))


print(f"Saved results: {RESULTS_OUTPUT}")

print("\nFinal model training complete.")