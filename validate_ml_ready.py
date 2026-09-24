import csv
from collections import Counter

PATH = 'Student_Performance_ML_Ready.csv'

with open(PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = list(reader.fieldnames)

TARGET_ENCODED = 'final_grade_encoded'
TARGET_LABEL   = 'final_grade'
NON_FEATURES   = {TARGET_ENCODED, TARGET_LABEL}

features = [h for h in headers if h not in NON_FEATURES]

print('=' * 62)
print('ML-READY DATASET VALIDATION REPORT')
print('=' * 62)

# 1. Shape
print(f'\n1. SHAPE')
print(f'   Rows    : {len(rows)}')
print(f'   Columns : {len(headers)}  ({len(features)} features + {len(NON_FEATURES)} target columns)')

# 2. Feature names
print(f'\n2. FEATURE NAMES ({len(features)} total)')
for i, h in enumerate(features, 1):
    print(f'   {i:2d}. {h}')

# 3. Target column
print(f'\n3. TARGET COLUMN')
print(f'   Encoded : {TARGET_ENCODED}')
print(f'   Label   : {TARGET_LABEL}  (kept for readability, not a feature)')
grade_map_display = {'0':'F','1':'E','2':'D','3':'C','4':'B','5':'A'}
print(f'   Classes : 0=F  1=E  2=D  3=C  4=B  5=A')

# 4. Data types (infer per column)
print(f'\n4. DATA TYPES (inferred)')
for h in headers:
    sample = [r[h] for r in rows[:200] if r[h].strip()]
    try:
        [int(v) for v in sample]
        dtype = 'int'
    except ValueError:
        try:
            [float(v) for v in sample]
            dtype = 'float'
        except ValueError:
            dtype = 'str'
    uniq = sorted(set(r[h] for r in rows))
    if len(uniq) <= 10:
        note = f'values: {uniq}'
    else:
        try:
            fvals = sorted(float(r[h]) for r in rows)
            note = f'range [{fvals[0]:.1f}, {fvals[-1]:.1f}]'
        except:
            note = f'{len(uniq)} unique'
    marker = ' <- TARGET' if h in NON_FEATURES else ''
    print(f'   {h:<35s} {dtype:<6s}  {note}{marker}')

# 5. Missing values
print(f'\n5. MISSING VALUES')
total_missing = 0
for h in headers:
    n = sum(1 for r in rows if r[h].strip() == '')
    total_missing += n
    if n > 0:
        print(f'   {h}: {n}')
if total_missing == 0:
    print(f'   None. All {len(headers)} columns are complete.')

# 6. Duplicate rows
print(f'\n6. DUPLICATE ROWS')
tuples = [tuple(r[h] for h in headers) for r in rows]
dupes  = len(tuples) - len(set(tuples))
print(f'   {dupes}' + (' -- none found.' if dupes == 0 else ' duplicates detected!'))

# 7. Class distribution
print(f'\n7. CLASS DISTRIBUTION  (final_grade / final_grade_encoded)')
grade_counter = Counter(r[TARGET_LABEL] for r in rows)
enc_counter   = Counter(r[TARGET_ENCODED] for r in rows)
total = len(rows)
print(f'   {"Grade":<8} {"Encoded":<10} {"Count":<8} {"Percent"}')
print(f'   {"-"*38}')
for enc in ['0','1','2','3','4','5']:
    lbl  = grade_map_display[enc]
    cnt  = enc_counter.get(enc, 0)
    pct  = 100 * cnt / total
    print(f'   {lbl:<8} {enc:<10} {cnt:<8} {pct:.1f}%')

# 8. student_id check
print(f'\n8. student_id EXCLUSION CHECK')
if 'student_id' in headers:
    print('   WARNING: student_id is present in the file!')
else:
    print('   PASS. student_id is NOT present in the dataset.')

# 9. Target leakage check
print(f'\n9. TARGET LEAKAGE CHECK')
for col in [TARGET_ENCODED, TARGET_LABEL]:
    if col in features:
        print(f'   FAIL: {col} appears in the feature list!')
    else:
        print(f'   PASS: {col} is NOT in the feature list.')

# 10. overall_score vs subject scores
print(f'\n10. overall_score / SUBJECT SCORES COLLINEARITY CHECK')
has_overall  = 'overall_score'   in features
has_math     = 'math_score'      in features
has_science  = 'science_score'   in features
has_english  = 'english_score'   in features

print(f'    overall_score present   : {has_overall}')
print(f'    math_score present      : {has_math}')
print(f'    science_score present   : {has_science}')
print(f'    english_score present   : {has_english}')

if has_overall and (has_math or has_science or has_english):
    print()
    print('    STATUS : Both overall_score AND individual subject scores')
    print('             are present in the feature set.')
    print()
    print('    DECISION (documented in feature_engineering.py):')
    print('      All four columns are RETAINED in the CSV so that the')
    print('      model-builder can choose one of two strategies:')
    print()
    print('      Strategy A -- "overall_score only"')
    print('        Features: overall_score (drop math/science/english)')
    print('        Pro: no multicollinearity, single strong predictor (r=0.906)')
    print()
    print('      Strategy B -- "subject scores only"')
    print('        Features: math_score, science_score, english_score')
    print('                  (drop overall_score)')
    print('        Pro: preserves subject-level signal, useful for')
    print('             interpreting which subject matters most')
    print()
    print('      ACTION REQUIRED: pick one strategy before model training.')
    print('      Do NOT feed all four into the same model simultaneously.')
else:
    print('    No conflict detected.')

print()
print('=' * 62)
print('VALIDATION COMPLETE')
print('=' * 62)
