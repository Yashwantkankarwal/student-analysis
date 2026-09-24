"""
feature_engineering.py
=======================
Reads  : Student_Performance_Cleaned.csv   (never modified)
Writes : Student_Performance_ML_Ready.csv

Encoding decisions
------------------
DROPPED
  student_id          -- identifier, carries no predictive signal

PASS-THROUGH (numeric, already on a meaningful scale)
  age
  study_hours
  attendance_percentage
  math_score
  science_score
  english_score
  overall_score       -- kept as its own feature; the user-instruction asks us
                         NOT to use overall_score *together* with the three
                         subject scores in the same model, so we keep all four
                         columns and let the model-builder choose which set
                         to use (a column-comment in the header documents this).

BINARY ENCODING  (yes->1 / no->0)
  internet_access
  extra_activities

ORDINAL ENCODING  (natural order preserved as integers)
  travel_time         : <15 min=1, 15-30 min=2, 30-60 min=3, >60 min=4
  parent_education    : no formal=1, high school=2, diploma=3,
                        graduate=4, post graduate=5, phd=6

LABEL ENCODING of the TARGET
  final_grade         : f=0, e=1, d=2, c=3, b=4, a=5
                        (ordinal: higher label = better grade)

ONE-HOT ENCODING  (nominal, no natural order)
  gender              : gender_female, gender_male, gender_other
  school_type         : school_type_private, school_type_public
  study_method        : study_method_coaching, study_method_group_study,
                        study_method_mixed, study_method_notes,
                        study_method_online_videos, study_method_textbook
  NOTE: one category is kept as the reference (drop_first=True equivalent):
    gender_other dropped  (reference = other)
    school_type_public dropped  (reference = public)
    study_method_textbook dropped  (reference = textbook)
"""

import csv
import os

INPUT  = 'Student_Performance_Cleaned.csv'
OUTPUT = 'Student_Performance_ML_Ready.csv'

# ── Ordinal maps ───────────────────────────────────────────────────────────────
TRAVEL_MAP = {
    '<15 min' : 1,
    '15-30 min': 2,
    '30-60 min': 3,
    '>60 min'  : 4,
}

PARENT_EDU_MAP = {
    'no formal'    : 1,
    'high school'  : 2,
    'diploma'      : 3,
    'graduate'     : 4,
    'post graduate': 5,
    'phd'          : 6,
}

GRADE_MAP = {
    'f': 0,
    'e': 1,
    'd': 2,
    'c': 3,
    'b': 4,
    'a': 5,
}

BINARY_MAP = {'yes': 1, 'no': 0}

# One-hot categories (all levels collected from data)
GENDER_LEVELS      = ['female', 'male']          # 'other' = reference (dropped)
SCHOOL_LEVELS      = ['private']                  # 'public' = reference (dropped)
STUDY_METHOD_LEVELS = [
    'coaching', 'group study', 'mixed', 'notes', 'online videos'
]                                                 # 'textbook' = reference (dropped)

def slugify(s):
    """Convert a category label to a safe column name."""
    return s.strip().lower().replace(' ', '_').replace('-', '_')

# ── Build output column order ──────────────────────────────────────────────────
NUMERIC_COLS = [
    'age', 'study_hours', 'attendance_percentage',
    'math_score', 'science_score', 'english_score', 'overall_score',
]

BINARY_COLS = ['internet_access', 'extra_activities']

ORDINAL_COLS = {
    'travel_time'      : TRAVEL_MAP,
    'parent_education' : PARENT_EDU_MAP,
}

ONE_HOT_SPEC = [
    ('gender',       GENDER_LEVELS,       'gender'),
    ('school_type',  SCHOOL_LEVELS,       'school_type'),
    ('study_method', STUDY_METHOD_LEVELS, 'study_method'),
]

TARGET_COL = 'final_grade'   # encoded as final_grade_encoded

def encode_row(row):
    out = {}

    # Numeric pass-through
    for c in NUMERIC_COLS:
        out[c] = row[c]

    # Binary
    for c in BINARY_COLS:
        out[c] = BINARY_MAP[row[c].strip().lower()]

    # Ordinal
    for c, mapping in ORDINAL_COLS.items():
        out[c] = mapping[row[c].strip().lower()]

    # One-hot
    for src_col, levels, prefix in ONE_HOT_SPEC:
        val = row[src_col].strip().lower()
        for lvl in levels:
            col_name = f'{prefix}_{slugify(lvl)}'
            out[col_name] = 1 if val == lvl else 0

    # Target
    out['final_grade_encoded'] = GRADE_MAP[row[TARGET_COL].strip().lower()]
    # Keep original label for readability
    out['final_grade'] = row[TARGET_COL].strip().lower()

    return out

# ── Process ────────────────────────────────────────────────────────────────────
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Input rows : {len(rows)}')

encoded_rows = [encode_row(r) for r in rows]

# Determine column order deterministically
out_headers = (
    NUMERIC_COLS
    + BINARY_COLS
    + list(ORDINAL_COLS.keys())
    + [f'{prefix}_{slugify(lvl)}'
       for src_col, levels, prefix in ONE_HOT_SPEC
       for lvl in levels]
    + ['final_grade_encoded', 'final_grade']
)

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=out_headers)
    writer.writeheader()
    writer.writerows(encoded_rows)

print(f'Output rows: {len(encoded_rows)}')
print(f'Output cols: {len(out_headers)}')
print(f'Saved to   : {OUTPUT}')

# ── Verification report ────────────────────────────────────────────────────────
print('\n=== Column listing ===')
for i, h in enumerate(out_headers, 1):
    if h in NUMERIC_COLS:
        kind = 'numeric (pass-through)'
    elif h in BINARY_COLS:
        kind = 'binary  (yes=1 / no=0)'
    elif h in ORDINAL_COLS:
        kind = f'ordinal ({"/".join(str(v) for v in sorted(ORDINAL_COLS[h].values()))})'
    elif h == 'final_grade_encoded':
        kind = 'target  (f=0 e=1 d=2 c=3 b=4 a=5)'
    elif h == 'final_grade':
        kind = 'target  (original label, not a feature)'
    else:
        kind = 'one-hot (0/1)'
    print(f'  {i:2d}. {h:<35s} {kind}')

# Spot-check: verify no unexpected values
print('\n=== Spot-check: unique values per encoded column ===')
from collections import Counter
for h in out_headers:
    unique_vals = sorted(set(str(r[h]) for r in encoded_rows))
    if len(unique_vals) <= 10:
        print(f'  {h:<35s} {unique_vals}')
    else:
        vals = sorted(float(r[h]) for r in encoded_rows)
        print(f'  {h:<35s} range [{vals[0]:.1f}, {vals[-1]:.1f}], '
              f'{len(unique_vals)} unique values')

# Confirm original file is untouched
orig_size = os.path.getsize(INPUT)
print(f'\nOriginal file size ({INPUT}): {orig_size:,} bytes — untouched.')
