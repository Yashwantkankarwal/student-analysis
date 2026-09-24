import csv
import math
from collections import Counter

path = 'Student_Performance.csv'

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames

print(f'Total rows (data): {len(rows)}')
print(f'Total columns: {len(headers)}')
print(f'Columns: {headers}')
print()

# Infer types and collect values
numeric_cols = []
categorical_cols = []
col_values = {h: [] for h in headers}

for row in rows:
    for h in headers:
        col_values[h].append(row[h])

# Determine column types
for h in headers:
    vals = col_values[h]
    non_empty = [v for v in vals if v.strip() != '']
    try:
        for v in non_empty[:50]:
            float(v)
        numeric_cols.append(h)
    except ValueError:
        categorical_cols.append(h)

print(f'Numeric columns: {numeric_cols}')
print(f'Categorical columns: {categorical_cols}')
print()

# Missing values
print('=== Missing Values ===')
for h in headers:
    missing = sum(1 for v in col_values[h] if v.strip() == '')
    print(f'  {h}: {missing}')
print()

# Duplicates (full row)
row_tuples = [tuple(row[h] for h in headers) for row in rows]
dup_count = len(row_tuples) - len(set(row_tuples))
print(f'Duplicate rows: {dup_count}')
print()

# Numeric stats
print('=== Numeric Stats ===')
for h in numeric_cols:
    vals = []
    for v in col_values[h]:
        try:
            vals.append(float(v))
        except:
            pass
    if vals:
        vals_sorted = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(variance)
        mn = min(vals)
        mx = max(vals)
        q1 = vals_sorted[int(n * 0.25)]
        median = vals_sorted[int(n * 0.50)]
        q3 = vals_sorted[int(n * 0.75)]
        print(f'  {h}: count={n}, mean={mean:.2f}, std={std:.2f}, min={mn}, Q1={q1}, median={median}, Q3={q3}, max={mx}')
print()

# Categorical unique values
print('=== Categorical Unique Values ===')
for h in categorical_cols:
    vals = [v.strip() for v in col_values[h] if v.strip() != '']
    unique = sorted(set(vals))
    counts = Counter(vals)
    print(f'  {h} ({len(unique)} unique): {unique}')
    for u in unique:
        print(f'    "{u}": {counts[u]}')
print()

# Data quality checks
print('=== Data Quality Issues ===')

ages = [int(row['age']) for row in rows if row['age'].strip()]
print(f'  Age range: {min(ages)} - {max(ages)}')
out_age = [a for a in ages if a < 5 or a > 25]
print(f'  Ages outside 5-25: {len(out_age)}')

sh = [float(row['study_hours']) for row in rows if row['study_hours'].strip()]
neg_sh = [v for v in sh if v < 0]
high_sh = [v for v in sh if v > 24]
print(f'  Negative study_hours: {len(neg_sh)}')
print(f'  study_hours > 24: {len(high_sh)}')

for sc in ['math_score', 'science_score', 'english_score', 'overall_score']:
    sv = [float(row[sc]) for row in rows if row[sc].strip()]
    out = [v for v in sv if v < 0 or v > 100]
    print(f'  {sc} out of [0,100]: {len(out)}')

att = [float(row['attendance_percentage']) for row in rows if row['attendance_percentage'].strip()]
out_att = [v for v in att if v < 0 or v > 100]
print(f'  attendance_percentage out of [0,100]: {len(out_att)}')

print()
print('=== overall_score consistency check (vs avg of 3 subject scores) ===')
mismatches = 0
for row in rows:
    try:
        m = float(row['math_score'])
        s = float(row['science_score'])
        e = float(row['english_score'])
        o = float(row['overall_score'])
        computed = (m + s + e) / 3
        if abs(computed - o) > 2.0:
            mismatches += 1
    except:
        pass
print(f'  Rows where overall_score differs from simple avg by >2 pts: {mismatches}')
