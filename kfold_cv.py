"""
kfold_cv.py
===========
5-fold stratified cross-validation for Experiment B feature set.
Evaluates: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
Metrics:   accuracy, macro-F1 (mean ± std across folds)
Output:    ML_Results/kfold_cv_results.txt
           ML_Results/kfold_cv_results.csv

Does NOT modify any existing results files.
"""

import csv, os
from collections import Counter

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import accuracy_score, f1_score
from sklearn.preprocessing   import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT      = 'Student_Performance_ML_Ready.csv'
OUTPUT_DIR = 'ML_Results'
TXT_OUT    = os.path.join(OUTPUT_DIR, 'kfold_cv_results.txt')
CSV_OUT    = os.path.join(OUTPUT_DIR, 'kfold_cv_results.csv')
N_SPLITS   = 5
RANDOM_STATE = 42
TARGET     = 'final_grade_encoded'

# Experiment B feature set (same as ml_training.py)
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

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data ...')
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows   = list(reader)

X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
y = np.array([int(r[TARGET]) for r in rows])
print(f'  {len(rows)} rows | {len(FEATURES)} features | classes: {sorted(Counter(y).keys())}')

# ── CV setup ───────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def make_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=RANDOM_STATE, solver='lbfgs'),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE, max_depth=15),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE),
    }

# ── Run CV ─────────────────────────────────────────────────────────────────────
results = {}   # model_name -> {acc: [], f1: []}

for model_name, model in make_models().items():
    acc_scores, f1_scores = [], []
    print(f'\n  {model_name}')
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_name == 'Logistic Regression':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            model.fit(X_train, y_train)
        elif model_name == 'Gradient Boosting':
            class_counts = Counter(y_train)
            total  = len(y_train)
            n_cls  = len(class_counts)
            sw = np.array([total / (n_cls * class_counts[yi]) for yi in y_train])
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mf1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        acc_scores.append(acc)
        f1_scores.append(mf1)
        print(f'    Fold {fold}: acc={acc:.4f}  macro-F1={mf1:.4f}')

    results[model_name] = {'acc': acc_scores, 'f1': f1_scores}
    print(f'    --> mean acc={np.mean(acc_scores):.4f} (+/-{np.std(acc_scores):.4f})  '
          f'mean macro-F1={np.mean(f1_scores):.4f} (+/-{np.std(f1_scores):.4f})')

# ── Write TXT report ───────────────────────────────────────────────────────────
lines = []
lines.append('EXPERIMENT B -- 5-FOLD STRATIFIED CROSS-VALIDATION RESULTS')
lines.append('=' * 65)
lines.append(f'Input file   : {INPUT}')
lines.append(f'Feature set  : Experiment B (math/science/english + 15 others)')
lines.append(f'Features ({len(FEATURES)}): {", ".join(FEATURES)}')
lines.append(f'Target       : {TARGET}  (F=0 E=1 D=2 C=3 B=4 A=5)')
lines.append(f'Folds        : {N_SPLITS} (StratifiedKFold, shuffle=True, random_state={RANDOM_STATE})')
lines.append(f'Imbalance    : class_weight="balanced" (sample_weight for Gradient Boosting)')
lines.append('')
lines.append(f'{"Model":<22} {"Acc Mean":>9} {"Acc Std":>8} {"MacroF1 Mean":>13} {"MacroF1 Std":>11}')
lines.append('-' * 65)

for model_name, r in results.items():
    acc_arr = r['acc']
    f1_arr  = r['f1']
    lines.append(
        f'{model_name:<22} {np.mean(acc_arr):>9.4f} {np.std(acc_arr):>8.4f} '
        f'{np.mean(f1_arr):>13.4f} {np.std(f1_arr):>11.4f}'
    )

lines.append('')
lines.append('Per-fold detail:')
lines.append('-' * 65)
for model_name, r in results.items():
    lines.append(f'\n{model_name}:')
    for i, (a, f) in enumerate(zip(r['acc'], r['f1']), 1):
        lines.append(f'  Fold {i}: accuracy={a:.4f}  macro-F1={f:.4f}')
    lines.append(f'  Mean : accuracy={np.mean(r["acc"]):.4f}  macro-F1={np.mean(r["f1"]):.4f}')
    lines.append(f'  Std  : accuracy={np.std(r["acc"]):.4f}  macro-F1={np.std(r["f1"]):.4f}')

with open(TXT_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'\nSaved: {TXT_OUT}')

# ── Write CSV ──────────────────────────────────────────────────────────────────
csv_rows = []
for model_name, r in results.items():
    for i, (a, f) in enumerate(zip(r['acc'], r['f1']), 1):
        csv_rows.append({'model': model_name, 'fold': i,
                         'accuracy': round(a, 4), 'macro_f1': round(f, 4)})
    csv_rows.append({'model': model_name, 'fold': 'mean',
                     'accuracy': round(float(np.mean(r['acc'])), 4),
                     'macro_f1': round(float(np.mean(r['f1'])), 4)})
    csv_rows.append({'model': model_name, 'fold': 'std',
                     'accuracy': round(float(np.std(r['acc'])), 4),
                     'macro_f1': round(float(np.std(r['f1'])), 4)})

with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['model','fold','accuracy','macro_f1'])
    w.writeheader()
    w.writerows(csv_rows)
print(f'Saved: {CSV_OUT}')

print('\nDone.')
