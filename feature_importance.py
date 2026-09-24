"""
feature_importance.py

Feature importance analysis for Experiment B.

Input:
    Student_Performance_ML_Ready.csv

Target:
    final_grade_encoded

Experiment B:
    18 input features
    Tuned Gradient Boosting
    learning_rate=0.05
    max_depth=3
    n_estimators=200
    subsample=0.8
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "Student_Performance_ML_Ready.csv"
OUTPUT_DIR = "ML_Results"

TARGET = "final_grade_encoded"

# These are NOT input features.
# overall_score is excluded because it is directly related to
# the final grade and was not part of the 18-feature Experiment B set.
EXCLUDED_COLUMNS = {
    "overall_score",
    "final_grade_encoded",
    "final_grade",
}


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print(f"Input file : {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE} in the current folder."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows       : {len(df)}")
print(f"Columns    : {len(df.columns)}")


# ============================================================
# 3. CHECK TARGET
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# 4. SELECT THE 18 EXPERIMENT B FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

print()
print(f"Number of features: {len(feature_columns)}")

print()
print("Features used:")
for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:2}. {feature}")


if len(feature_columns) != 18:
    raise ValueError(
        f"Expected 18 Experiment B features, "
        f"but found {len(feature_columns)}."
    )


X = df[feature_columns]
y = df[TARGET]


# ============================================================
# 5. GRADIENT BOOSTING
# ============================================================

print()
print("=" * 60)
print("GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 60)

gb_model = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

print("Training Gradient Boosting model...")

gb_model.fit(X, y)

gb_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": gb_model.feature_importances_
})

gb_importance = gb_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

gb_importance["rank"] = range(1, len(gb_importance) + 1)

gb_importance = gb_importance[
    ["rank", "feature", "importance"]
]


print()
print("Gradient Boosting feature importance:")
print(gb_importance.to_string(index=False))


# ============================================================
# 6. RANDOM FOREST
# ============================================================

print()
print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training Random Forest model...")

rf_model.fit(X, y)

rf_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

rf_importance["rank"] = range(1, len(rf_importance) + 1)

rf_importance = rf_importance[
    ["rank", "feature", "importance"]
]


print()
print("Random Forest feature importance:")
print(rf_importance.to_string(index=False))


# ============================================================
# 7. SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance_results.csv"
)

combined = pd.DataFrame({
    "feature": feature_columns,
    "gradient_boosting_importance": [
        gb_importance.set_index("feature").loc[f, "importance"]
        for f in feature_columns
    ],
    "random_forest_importance": [
        rf_importance.set_index("feature").loc[f, "importance"]
        for f in feature_columns
    ]
})

combined["gb_rank"] = (
    combined["gradient_boosting_importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)

combined["rf_rank"] = (
    combined["random_forest_importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)

combined = combined.sort_values(
    "gradient_boosting_importance",
    ascending=False
)

combined.to_csv(csv_path, index=False)


# ============================================================
# 8. GRADIENT BOOSTING CHART
# ============================================================

gb_chart = os.path.join(
    OUTPUT_DIR,
    "gradient_boosting_feature_importance.png"
)

plot_data = gb_importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_data["feature"],
    plot_data["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Experiment B - Gradient Boosting Feature Importance"
)

plt.tight_layout()
plt.savefig(gb_chart, dpi=300)
plt.close()


# ============================================================
# 9. RANDOM FOREST CHART
# ============================================================

rf_chart = os.path.join(
    OUTPUT_DIR,
    "random_forest_feature_importance.png"
)

plot_data = rf_importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_data["feature"],
    plot_data["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()
plt.savefig(rf_chart, dpi=300)
plt.close()


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 60)

print(f"Saved: {csv_path}")
print(f"Saved: {gb_chart}")
print(f"Saved: {rf_chart}")

print()
print("Top 5 Gradient Boosting features:")

for _, row in gb_importance.head(5).iterrows():
    print(
        f"{int(row['rank'])}. "
        f"{row['feature']} "
        f"({row['importance']:.4f})"
    )

print()
print("Analysis finished successfully.")