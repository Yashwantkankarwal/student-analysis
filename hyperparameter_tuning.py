"""
hyperparameter_tuning.py
========================
GridSearchCV hyperparameter tuning for Experiment B feature set.

Input  : Student_Performance_ML_Ready.csv  (never modified)
Target : final_grade_encoded  (F=0 E=1 D=2 C=3 B=4 A=5)
CV     : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Scoring: macro-F1 (primary), accuracy (secondary)
Imbalance: class_weight='balanced' on all classifiers that support it;
           for GradientBoosting a FunctionTransformer wrapper passes
           per-sample weights into fit() via Pipeline.

Outputs (never overwrites existing files):
  ML_Results/hyperparameter_tuning_results.csv
  ML_Results/hyperparameter_tuning_summary.txt
"""

import csv, os, time
from collections import Counter

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import make_scorer, f1_score, accuracy_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT      = 'Student_Performance_ML_Ready.csv'
OUTPUT_DIR = 'ML_Results'
CSV_OUT    = os.path.join(OUTPUT_DIR, 'hyperparameter_tuning_results.csv')
TXT_OUT    = os.path.join(OUTPUT_DIR, 'hyperparameter_tuning_summary.txt')

# Guard: refuse to overwrite existing output files
for p in (CSV_OUT, TXT_OUT):
    if os.path.exists(p):
        raise FileExistsError(
            f'{p} already exists. Remove it manually before re-running.')

RANDOM_STATE = 42
N_FOLDS      = 5

# Experiment B features (identical to ml_training.py and kfold_cv.py)
FEATURES = [
    'age', 'study_hours', 'attendance_percentage',
    'internet_access', 'extra_activities',
    'travel_time', 'parent_education',
    'gender_female', 'gender_male',
    'school_type_private',
    'study_method_coaching', 'study_method_group_study',
    'study_method_mixed', 'study_method_notes',
    'study_method_online_videos',
    'math_score', 'science_score', 'english_score',
]
TARGET = 'final_grade_encoded'

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data ...')
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows   = list(reader)

X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
y = np.array([int(r[TARGET]) for r in rows])
print(f'  {len(rows)} rows | {len(FEATURES)} features')
print(f'  Class distribution: { {k: int(v) for k,v in sorted(Counter(y).items())} }')

# ── CV splitter ────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

# ── Parameter grids ────────────────────────────────────────────────────────────
# Each entry: (name, estimator_or_pipeline, param_grid)
# Logistic Regression uses a StandardScaler Pipeline; param keys prefixed 'lr__'
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        class_weight='balanced', random_state=RANDOM_STATE,
        solver='lbfgs', max_iter=2000)),
])
lr_grid = {
    'lr__C'           : [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__max_iter'    : [500, 1000, 2000],
}

dt_est  = DecisionTreeClassifier(
    class_weight='balanced', random_state=RANDOM_STATE)
dt_grid = {
    'max_depth'        : [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'criterion'        : ['gini', 'entropy'],
}

rf_est  = RandomForestClassifier(
    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf' : [1, 2],
    'max_features'     : ['sqrt', 'log2'],
}

# GradientBoosting does not natively support class_weight.
# We use sample_weight computed once for the whole dataset.
# GridSearchCV passes fit_params to each inner fold's fit() call.
# Compute inverse-frequency weights over the full y array.
class_counts = Counter(y)
total        = len(y)
n_cls        = len(class_counts)
sample_weight_all = np.array(
    [total / (n_cls * class_counts[yi]) for yi in y]
)

gb_est  = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb_grid = {
    'n_estimators'  : [100, 200],
    'learning_rate' : [0.05, 0.1, 0.2],
    'max_depth'     : [3, 5, 7],
    'subsample'     : [0.8, 1.0],
}

SEARCH_CONFIGS = [
    ('Logistic Regression', lr_pipe,  lr_grid, {}),
    ('Decision Tree',       dt_est,   dt_grid, {}),
    ('Random Forest',       rf_est,   rf_grid, {}),
    ('Gradient Boosting',   gb_est,   gb_grid,
        {'sample_weight': sample_weight_all}),
]

# ── Run grid searches ──────────────────────────────────────────────────────────
all_results = []

for name, estimator, param_grid, fit_params in SEARCH_CONFIGS:
    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    print(f'\n{"="*60}')
    print(f'Tuning: {name}  ({n_combinations} param combinations x {N_FOLDS} folds)')
    print(f'{"="*60}')

    t0 = time.time()
    gs = GridSearchCV(
        estimator  = estimator,
        param_grid = param_grid,
        scoring    = {'macro_f1': macro_f1_scorer,
                      'accuracy': 'accuracy'},
        refit      = 'macro_f1',
        cv         = skf,
        n_jobs     = -1,
        verbose    = 1,
        return_train_score = False,
    )
    gs.fit(X, y, **fit_params)
    elapsed = time.time() - t0

    best_idx      = gs.best_index_
    cv_results    = gs.cv_results_
    mean_acc      = cv_results['mean_test_accuracy'][best_idx]
    std_acc       = cv_results['std_test_accuracy'][best_idx]
    mean_f1       = cv_results['mean_test_macro_f1'][best_idx]
    std_f1        = cv_results['std_test_macro_f1'][best_idx]
    best_params   = gs.best_params_

    print(f'  Best params : {best_params}')
    print(f'  CV Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}')
    print(f'  CV Macro-F1 : {mean_f1:.4f} +/- {std_f1:.4f}')
    print(f'  Time        : {elapsed:.1f}s')

    all_results.append({
        'model'        : name,
        'best_params'  : str(best_params),
        'cv_acc_mean'  : round(mean_acc, 4),
        'cv_acc_std'   : round(std_acc,  4),
        'cv_f1_mean'   : round(mean_f1,  4),
        'cv_f1_std'    : round(std_f1,   4),
        'n_folds'      : N_FOLDS,
        'n_combinations': n_combinations,
        'time_seconds' : round(elapsed, 1),
    })

# ── Save CSV ───────────────────────────────────────────────────────────────────
fields = ['model','best_params','cv_acc_mean','cv_acc_std',
          'cv_f1_mean','cv_f1_std','n_folds','n_combinations','time_seconds']
with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_results)
print(f'\nSaved: {CSV_OUT}')

# ── Save TXT summary ───────────────────────────────────────────────────────────
lines = []
lines.append('EXPERIMENT B -- HYPERPARAMETER TUNING SUMMARY')
lines.append('=' * 68)
lines.append(f'Input file   : {INPUT}')
lines.append(f'Feature set  : Experiment B  ({len(FEATURES)} features)')
lines.append(f'Target       : {TARGET}  (F=0 E=1 D=2 C=3 B=4 A=5)')
lines.append(f'CV strategy  : StratifiedKFold  n_splits={N_FOLDS}  '
             f'shuffle=True  random_state={RANDOM_STATE}')
lines.append(f'Primary metric for selection: macro-F1')
lines.append(f'Imbalance    : class_weight="balanced" / sample_weight (GB)')
lines.append('')

for r in all_results:
    lines.append(f'{"─"*68}')
    lines.append(f'Model        : {r["model"]}')
    lines.append(f'Grid size    : {r["n_combinations"]} combinations  '
                 f'({r["n_combinations"] * N_FOLDS} fits)')
    lines.append(f'Best params  : {r["best_params"]}')
    lines.append(f'CV Accuracy  : {r["cv_acc_mean"]:.4f}  +/- {r["cv_acc_std"]:.4f}')
    lines.append(f'CV Macro-F1  : {r["cv_f1_mean"]:.4f}  +/- {r["cv_f1_std"]:.4f}')
    lines.append(f'Folds        : {r["n_folds"]}')
    lines.append(f'Time         : {r["time_seconds"]}s')
    lines.append('')

lines.append('=' * 68)
lines.append('RANKED BY CV MACRO-F1 (best first)')
lines.append('-' * 68)
lines.append(f'{"Rank":<5} {"Model":<22} {"Acc Mean":>9} {"Acc Std":>8} '
             f'{"F1 Mean":>8} {"F1 Std":>8}')
lines.append('-' * 68)
for rank, r in enumerate(
        sorted(all_results, key=lambda x: -x['cv_f1_mean']), 1):
    lines.append(f'{rank:<5} {r["model"]:<22} {r["cv_acc_mean"]:>9.4f} '
                 f'{r["cv_acc_std"]:>8.4f} {r["cv_f1_mean"]:>8.4f} '
                 f'{r["cv_f1_std"]:>8.4f}')

with open(TXT_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Saved: {TXT_OUT}')
print('\nHyperparameter tuning complete.')
