"""
ml_training.py
==============
Trains and compares two feature-strategy experiments to predict final_grade.

Experiment A : features include overall_score,  exclude subject scores
Experiment B : features include math/sci/eng,   exclude overall_score

Both experiments:
  - Same stratified 80/20 train-test split (random_state=42)
  - class_weight='balanced' on each classifier to handle grade imbalance
  - Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
  - Metrics: accuracy, precision, recall, macro-F1, weighted-F1, confusion matrix

Outputs (written to ML_Results/):
  results_summary.txt      -- human-readable table of all metrics
  experiment_A_detail.txt  -- per-model detail + confusion matrices (Exp A)
  experiment_B_detail.txt  -- per-model detail + confusion matrices (Exp B)
  metrics_comparison.csv   -- machine-readable metrics for all runs
"""

import csv, os, math, json
from collections import Counter

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)
from sklearn.preprocessing   import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT      = 'Student_Performance_ML_Ready.csv'
OUTPUT_DIR = 'ML_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET       = 'final_grade_encoded'
GRADE_LABELS = ['F','E','D','C','B','A']   # 0-5

# Feature sets
BASE_FEATURES = [
    'age', 'study_hours', 'attendance_percentage',
    'internet_access', 'extra_activities',
    'travel_time', 'parent_education',
    'gender_female', 'gender_male',
    'school_type_private',
    'study_method_coaching', 'study_method_group_study',
    'study_method_mixed', 'study_method_notes',
    'study_method_online_videos',
]

EXPERIMENT_A_FEATURES = BASE_FEATURES + ['overall_score']
EXPERIMENT_B_FEATURES = BASE_FEATURES + ['math_score', 'science_score', 'english_score']

EXPERIMENTS = {
    'A': {'label': 'Experiment A (overall_score)',
          'features': EXPERIMENT_A_FEATURES},
    'B': {'label': 'Experiment B (subject scores)',
          'features': EXPERIMENT_B_FEATURES},
}

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data ...')
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows   = list(reader)
    all_cols = list(reader.fieldnames)

print(f'  {len(rows)} rows, {len(all_cols)} columns')

def to_array(rows, col_names):
    return np.array([[float(r[c]) for c in col_names] for r in rows])

y_all = np.array([int(r[TARGET]) for r in rows])

# ── Shared stratified split ────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(np.zeros(len(y_all)), y_all))

print(f'  Train: {len(train_idx)}   Test: {len(test_idx)}')
print(f'  Test class dist: {sorted(Counter(y_all[test_idx]).items())}')

# ── Model definitions ──────────────────────────────────────────────────────────
def make_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=RANDOM_STATE, solver='lbfgs'),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE,
            max_depth=15),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE),
    }

# ── Helpers ────────────────────────────────────────────────────────────────────
def metric_row(y_true, y_pred, model_name, exp_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    mf1  = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    wf1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {'experiment': exp_name, 'model': model_name,
            'accuracy': round(acc,4), 'precision_w': round(prec,4),
            'recall_w': round(rec,4), 'macro_f1': round(mf1,4),
            'weighted_f1': round(wf1,4)}

def format_cm(cm, labels):
    w = max(len(l) for l in labels) + 2
    header = ' '*(w+2) + '  '.join(f'{l:>{w}}' for l in labels) + '  <- Predicted'
    lines  = [header]
    for i, row in enumerate(cm):
        line = f'{labels[i]:>{w}} |' + '  '.join(f'{v:>{w}}' for v in row)
        lines.append(line)
    return '\n'.join(lines)

# ── Run experiments ────────────────────────────────────────────────────────────
all_metrics = []
detail_lines = {'A': [], 'B': []}

for exp_id, exp_cfg in EXPERIMENTS.items():
    feat_names = exp_cfg['features']
    label      = exp_cfg['label']
    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'  Features ({len(feat_names)}): {feat_names}')
    print(f'{"="*60}')

    X_all = to_array(rows, feat_names)
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Scale for Logistic Regression (others are tree-based, don't need it)
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    detail = detail_lines[exp_id]
    detail.append(f'{"="*60}')
    detail.append(f'{label}')
    detail.append(f'Features ({len(feat_names)}): {", ".join(feat_names)}')
    detail.append(f'Train samples: {len(y_train)}  |  Test samples: {len(y_test)}')
    detail.append(f'{"="*60}\n')

    models = make_models()
    for model_name, model in models.items():
        print(f'  Training {model_name} ...', end=' ', flush=True)

        # Logistic Regression uses scaled data
        if model_name == 'Logistic Regression':
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            # GradientBoosting doesn't support class_weight; compensate via
            # sample_weight at fit time
            if model_name == 'Gradient Boosting':
                class_counts = Counter(y_train)
                total = len(y_train)
                n_cls = len(class_counts)
                sample_weights = np.array([
                    total / (n_cls * class_counts[yi]) for yi in y_train
                ])
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        metrics = metric_row(y_test, y_pred, model_name, exp_id)
        all_metrics.append(metrics)
        print(f'acc={metrics["accuracy"]:.4f}  macro-F1={metrics["macro_f1"]:.4f}')

        # Detail block
        detail.append(f'--- {model_name} ---')
        detail.append(f'Accuracy        : {metrics["accuracy"]:.4f}')
        detail.append(f'Precision (w)   : {metrics["precision_w"]:.4f}')
        detail.append(f'Recall    (w)   : {metrics["recall_w"]:.4f}')
        detail.append(f'Macro F1        : {metrics["macro_f1"]:.4f}')
        detail.append(f'Weighted F1     : {metrics["weighted_f1"]:.4f}')
        detail.append('')
        detail.append('Classification Report:')
        detail.append(classification_report(
            y_test, y_pred,
            target_names=GRADE_LABELS, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        detail.append('Confusion Matrix (rows=Actual, cols=Predicted):')
        detail.append(format_cm(cm, GRADE_LABELS))
        detail.append('\n')

        # Feature importances (where available)
        if hasattr(model, 'feature_importances_'):
            fi = sorted(zip(feat_names, model.feature_importances_),
                        key=lambda x: -x[1])
            detail.append('Top-10 Feature Importances:')
            for fname, imp in fi[:10]:
                bar = '#' * int(imp * 60)
                detail.append(f'  {fname:<35s} {imp:.4f}  {bar}')
            detail.append('')
        elif hasattr(model, 'coef_'):
            # LR: mean absolute coefficient per feature
            abs_coef = np.abs(model.coef_).mean(axis=0)
            fi = sorted(zip(feat_names, abs_coef), key=lambda x: -x[1])
            detail.append('Top-10 Feature Importances (mean |coef|):')
            for fname, imp in fi[:10]:
                bar = '#' * int(imp * 20)
                detail.append(f'  {fname:<35s} {imp:.4f}  {bar}')
            detail.append('')

# ── Write detail files ─────────────────────────────────────────────────────────
for exp_id in ('A', 'B'):
    path = os.path.join(OUTPUT_DIR, f'experiment_{exp_id}_detail.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(detail_lines[exp_id]))
    print(f'\nSaved: {path}')

# ── Write metrics CSV ──────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.csv')
csv_fields = ['experiment','model','accuracy','precision_w',
              'recall_w','macro_f1','weighted_f1']
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    w.writerows(all_metrics)
print(f'Saved: {csv_path}')

# ── Write summary report ───────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, 'results_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('STUDENT PERFORMANCE — ML EXPERIMENT RESULTS SUMMARY\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'Input file  : {INPUT}\n')
    f.write(f'Split       : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} '
            f'stratified (random_state={RANDOM_STATE})\n')
    f.write(f'Imbalance   : handled via class_weight="balanced" '
            f'(sample_weight for GB)\n')
    f.write(f'Classes     : F=0  E=1  D=2  C=3  B=4  A=5\n\n')

    for exp_id in ('A','B'):
        exp_label = EXPERIMENTS[exp_id]['label']
        f.write(f'\n{exp_label}\n')
        f.write('-' * 55 + '\n')
        f.write(f'{"Model":<22} {"Acc":>6} {"Prec(w)":>8} '
                f'{"Rec(w)":>7} {"MacroF1":>8} {"WtdF1":>7}\n')
        f.write('-' * 55 + '\n')
        for m in all_metrics:
            if m['experiment'] == exp_id:
                f.write(f'{m["model"]:<22} {m["accuracy"]:>6.4f} '
                        f'{m["precision_w"]:>8.4f} {m["recall_w"]:>7.4f} '
                        f'{m["macro_f1"]:>8.4f} {m["weighted_f1"]:>7.4f}\n')

    f.write('\n\nHEAD-TO-HEAD COMPARISON (Macro F1)\n')
    f.write('=' * 55 + '\n')
    f.write(f'{"Model":<22} {"Exp A":>8} {"Exp B":>8} {"Winner":>8}\n')
    f.write('-' * 55 + '\n')
    a_metrics = {m['model']: m for m in all_metrics if m['experiment']=='A'}
    b_metrics = {m['model']: m for m in all_metrics if m['experiment']=='B'}
    for mname in a_metrics:
        fa = a_metrics[mname]['macro_f1']
        fb = b_metrics[mname]['macro_f1']
        winner = 'A' if fa > fb else ('B' if fb > fa else 'TIE')
        f.write(f'{mname:<22} {fa:>8.4f} {fb:>8.4f} {winner:>8}\n')

print(f'Saved: {summary_path}')
print('\nAll done.')
