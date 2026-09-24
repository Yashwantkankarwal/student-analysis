import csv
import math
from collections import Counter, defaultdict

path = 'Student_Performance.csv'

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames

print("=" * 60)
print("SECTION 1: DUPLICATE INVESTIGATION")
print("=" * 60)

total_rows = len(rows)
print(f"\nTotal rows: {total_rows}")

# Full-row duplicates
row_tuples = [tuple(row[h] for h in headers) for row in rows]
unique_rows = set(row_tuples)
num_unique_rows = len(unique_rows)
num_duplicates = total_rows - num_unique_rows
print(f"Unique rows (all columns): {num_unique_rows}")
print(f"Exact duplicate rows: {num_duplicates}")

# Unique student_id values
student_ids = [row['student_id'] for row in rows]
unique_ids = set(student_ids)
print(f"Unique student_id values: {len(unique_ids)}")
print(f"student_id range: {min(int(x) for x in unique_ids)} - {max(int(x) for x in unique_ids)}")

# How many IDs appear more than once?
id_counts = Counter(student_ids)
ids_appearing_more_than_once = {k: v for k, v in id_counts.items() if v > 1}
print(f"\nStudent IDs appearing more than once: {len(ids_appearing_more_than_once)}")
count_dist = Counter(id_counts.values())
print("Frequency distribution of ID repetitions:")
for times, count in sorted(count_dist.items()):
    print(f"  appears {times}x: {count} student_ids")

# Show example duplicate rows
print("\n--- Example duplicate rows (first 3 groups) ---")
seen = {}
shown = 0
for i, t in enumerate(row_tuples):
    if t in seen:
        if shown < 3:
            orig_idx = seen[t]
            print(f"\n  Original (row {orig_idx + 2}): {dict(zip(headers, t))}")
            print(f"  Duplicate (row {i + 2}): {dict(zip(headers, t))}")
            shown += 1
    else:
        seen[t] = i

# Are duplicate rows always FULLY identical (every column)?
print("\n--- Verifying: are all duplicates 100% identical across every column? ---")
dup_groups = defaultdict(list)
for i, t in enumerate(row_tuples):
    dup_groups[t].append(i)
partial_dups = 0
full_dups = 0
for t, indices in dup_groups.items():
    if len(indices) > 1:
        full_dups += len(indices) - 1
print(f"  All {full_dups} duplicate rows are 100% identical (every column matches).")

print("\n\n" + "=" * 60)
print("SECTION 2: OVERALL_SCORE INVESTIGATION")
print("=" * 60)

# Compute simple average vs overall_score
diffs = []
for row in rows:
    try:
        m = float(row['math_score'])
        s = float(row['science_score'])
        e = float(row['english_score'])
        o = float(row['overall_score'])
        avg = (m + s + e) / 3
        diffs.append((m, s, e, o, avg, round(o - avg, 4)))
    except:
        pass

exact_match = sum(1 for d in diffs if abs(d[5]) < 0.01)
within_1 = sum(1 for d in diffs if abs(d[5]) <= 1.0)
within_2 = sum(1 for d in diffs if abs(d[5]) <= 2.0)
beyond_2 = sum(1 for d in diffs if abs(d[5]) > 2.0)

print(f"\nTotal rows analysed: {len(diffs)}")
print(f"overall_score == simple avg (within 0.01): {exact_match}")
print(f"overall_score within 1.0 of simple avg:    {within_1}")
print(f"overall_score within 2.0 of simple avg:    {within_2}")
print(f"overall_score differs by >2.0:              {beyond_2}")

# Distribution of differences
diff_vals = [abs(d[5]) for d in diffs]
print(f"\nDiff stats: mean={sum(diff_vals)/len(diff_vals):.2f}, "
      f"max={max(diff_vals):.2f}, min={min(diff_vals):.2f}")

# Show examples across the full range
print("\n--- Sample rows: exact matches (diff ~0) ---")
shown = 0
for d in diffs:
    if abs(d[5]) < 0.01 and shown < 3:
        print(f"  math={d[0]}, sci={d[1]}, eng={d[2]} | avg={d[4]:.2f} | overall={d[3]} | diff={d[5]}")
        shown += 1

print("\n--- Sample rows: moderate difference (diff 5-10) ---")
shown = 0
for d in diffs:
    if 5 <= abs(d[5]) <= 10 and shown < 5:
        print(f"  math={d[0]}, sci={d[1]}, eng={d[2]} | avg={d[4]:.2f} | overall={d[3]} | diff={d[5]}")
        shown += 1

print("\n--- Sample rows: large difference (diff > 15) ---")
shown = 0
for d in diffs:
    if abs(d[5]) > 15 and shown < 5:
        print(f"  math={d[0]}, sci={d[1]}, eng={d[2]} | avg={d[4]:.2f} | overall={d[3]} | diff={d[5]}")
        shown += 1

# Try weighted formulas
print("\n--- Testing weighted formula hypotheses ---")

# Try weights that might sum to 1
weight_candidates = [
    ("0.4*math + 0.3*sci + 0.3*eng", 0.4, 0.3, 0.3),
    ("0.3*math + 0.4*sci + 0.3*eng", 0.3, 0.4, 0.3),
    ("0.3*math + 0.3*sci + 0.4*eng", 0.3, 0.3, 0.4),
    ("0.5*math + 0.25*sci + 0.25*eng", 0.5, 0.25, 0.25),
    ("0.25*math + 0.5*sci + 0.25*eng", 0.25, 0.5, 0.25),
    ("0.25*math + 0.25*sci + 0.5*eng", 0.25, 0.25, 0.5),
    ("0.4*math + 0.4*sci + 0.2*eng", 0.4, 0.4, 0.2),
    ("0.2*math + 0.4*sci + 0.4*eng", 0.2, 0.4, 0.4),
    ("0.4*math + 0.2*sci + 0.4*eng", 0.4, 0.2, 0.4),
]
for label, wm, ws, we in weight_candidates:
    mismatches = 0
    for row in rows:
        try:
            m = float(row['math_score'])
            s = float(row['science_score'])
            e = float(row['english_score'])
            o = float(row['overall_score'])
            computed = wm * m + ws * s + we * e
            if abs(computed - o) > 1.0:
                mismatches += 1
        except:
            pass
    print(f"  {label}: mismatches (>1pt) = {mismatches}")

# Try including attendance or study_hours as a factor
print("\n--- Testing attendance as a factor ---")
for att_weight in [0.05, 0.1, 0.15, 0.2]:
    s_weight = (1 - att_weight) / 3
    mismatches = 0
    for row in rows:
        try:
            m = float(row['math_score'])
            sc = float(row['science_score'])
            e = float(row['english_score'])
            o = float(row['overall_score'])
            a = float(row['attendance_percentage'])
            computed = s_weight * m + s_weight * sc + s_weight * e + att_weight * a
            if abs(computed - o) > 1.0:
                mismatches += 1
        except:
            pass
    print(f"  equal_subj*{s_weight:.3f} + att*{att_weight}: mismatches = {mismatches}")

# Check: is overall_score simply always between min and max of the three?
print("\n--- Is overall_score always within [min(scores), max(scores)]? ---")
outside_range = 0
for row in rows:
    try:
        m = float(row['math_score'])
        s = float(row['science_score'])
        e = float(row['english_score'])
        o = float(row['overall_score'])
        lo = min(m, s, e)
        hi = max(m, s, e)
        if o < lo - 0.5 or o > hi + 0.5:
            outside_range += 1
    except:
        pass
print(f"  Rows where overall_score is outside [min, max] of subject scores: {outside_range}")

# Correlation: which subject score correlates most with overall_score?
print("\n--- Pearson correlation of each subject score with overall_score ---")
def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy > 0 else 0

math_scores, sci_scores, eng_scores, overall_scores = [], [], [], []
for row in rows:
    try:
        math_scores.append(float(row['math_score']))
        sci_scores.append(float(row['science_score']))
        eng_scores.append(float(row['english_score']))
        overall_scores.append(float(row['overall_score']))
    except:
        pass

print(f"  math_score    vs overall_score: r = {pearson(math_scores, overall_scores):.4f}")
print(f"  science_score vs overall_score: r = {pearson(sci_scores, overall_scores):.4f}")
print(f"  english_score vs overall_score: r = {pearson(eng_scores, overall_scores):.4f}")

avg_scores = [(m + s + e) / 3 for m, s, e in zip(math_scores, sci_scores, eng_scores)]
print(f"  simple_avg    vs overall_score: r = {pearson(avg_scores, overall_scores):.4f}")
