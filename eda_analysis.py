import csv
import math
import os
from collections import Counter, defaultdict

# ── Minimal matplotlib setup (no display, save only) ──────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH = 'Student_Performance_Cleaned.csv'
CHART_DIR  = 'EDA_Charts'
os.makedirs(CHART_DIR, exist_ok=True)

PALETTE  = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']
GRADE_ORDER = ['a','b','c','d','e','f']

# ── Load data ──────────────────────────────────────────────────────────────────
with open(INPUT_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    ROWS    = list(reader)
    HEADERS = list(reader.fieldnames)

NUM_COLS = ['age','study_hours','attendance_percentage',
            'math_score','science_score','english_score','overall_score']
CAT_COLS = ['gender','school_type','parent_education','internet_access',
            'travel_time','extra_activities','study_method','final_grade']

def fv(row, col): return float(row[col])
def col_floats(col): return [fv(r, col) for r in ROWS]

def save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')

def pearson(xs, ys):
    n  = len(xs)
    mx = sum(xs)/n;  my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx  = math.sqrt(sum((x-mx)**2 for x in xs))
    dy  = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else 0

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 1. Dataset Shape ===')
print(f'  Rows: {len(ROWS)}   Columns: {len(HEADERS)}')

print('\n=== 2. Column names & inferred types ===')
for h in HEADERS:
    kind = 'numeric' if h in NUM_COLS else 'categorical'
    print(f'  {h:30s} {kind}')

print('\n=== 3. Missing values ===')
missing = {h: sum(1 for r in ROWS if r[h].strip()=='') for h in HEADERS}
any_missing = any(v>0 for v in missing.values())
print('  None.' if not any_missing else
      '\n'.join(f'  {k}: {v}' for k,v in missing.items() if v>0))

print('\n=== 4. Duplicate rows ===')
dupes = len(ROWS) - len(set(tuple(r[h] for h in HEADERS) for r in ROWS))
print(f'  {dupes}')

print('\n=== 5. Descriptive statistics (numeric columns) ===')
for col in NUM_COLS:
    vals = sorted(col_floats(col))
    n    = len(vals)
    mean = sum(vals)/n
    std  = math.sqrt(sum((x-mean)**2 for x in vals)/n)
    q1   = vals[int(n*0.25)]; med = vals[int(n*0.50)]; q3 = vals[int(n*0.75)]
    print(f'  {col:28s} mean={mean:6.2f}  std={std:6.2f}  '
          f'min={vals[0]:5.1f}  Q1={q1:5.1f}  med={med:5.1f}  '
          f'Q3={q3:5.1f}  max={vals[-1]:5.1f}')

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: histogram
def histogram(vals, bins, title, xlabel, colour, fname, extra_lines=None):
    fig, ax = plt.subplots(figsize=(8,4))
    counts, edges = np.histogram(vals, bins=bins)
    ax.bar([0.5*(edges[i]+edges[i+1]) for i in range(len(counts))],
           counts, width=(edges[1]-edges[0])*0.85,
           color=colour, edgecolor='white', linewidth=0.6)
    mean_v = sum(vals)/len(vals)
    ax.axvline(mean_v, color='#C44E52', linewidth=1.5, linestyle='--',
               label=f'Mean = {mean_v:.1f}')
    if extra_lines:
        for lv, lc, ll in extra_lines:
            ax.axvline(lv, color=lc, linewidth=1.5, linestyle=':', label=ll)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel('Count')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    save(fig, fname)

# Helper: bar chart from Counter
def bar_chart(counter, title, xlabel, fname, colour=None, order=None):
    if order:
        keys = [k for k in order if k in counter]
    else:
        keys = sorted(counter, key=lambda k: -counter[k])
    vals = [counter[k] for k in keys]
    colours = colour if colour else PALETTE[:len(keys)]
    if isinstance(colours, str):
        colours = [colours]*len(keys)
    fig, ax = plt.subplots(figsize=(max(6, len(keys)*1.1), 4))
    bars = ax.bar(keys, vals, color=colours, edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                str(v), ha='center', va='bottom', fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    save(fig, fname)

# Helper: box plot groups
def box_plot(groups, labels, title, ylabel, fname, colours=None):
    data = [groups[l] for l in labels]
    colours = colours or PALETTE[:len(labels)]
    fig, ax = plt.subplots(figsize=(max(7, len(labels)*1.2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colours):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
    for i, lbl in enumerate(labels, 1):
        m = sum(groups[lbl])/len(groups[lbl])
        ax.text(i, ax.get_ylim()[0]-2, f'μ={m:.1f}', ha='center',
                fontsize=8, color='#444')
    save(fig, fname)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 6-10. Distributions (numeric) ===')

# 6. Age
histogram(col_floats('age'), bins=range(13,21), title='Distribution of Age',
          xlabel='Age', colour=PALETTE[0], fname='06_age_distribution.png')

# 7. Study hours
histogram(col_floats('study_hours'), bins=20,
          title='Distribution of Study Hours',
          xlabel='Study Hours per Day', colour=PALETTE[1],
          fname='07_study_hours_distribution.png')

# 8. Attendance
histogram(col_floats('attendance_percentage'), bins=20,
          title='Distribution of Attendance Percentage',
          xlabel='Attendance (%)', colour=PALETTE[2],
          fname='08_attendance_distribution.png')

# 9. Subject scores – overlapping histograms
print('  Saved: 09_subject_scores_distribution.png')
fig, ax = plt.subplots(figsize=(9,4))
for col, c, lbl in [('math_score', PALETTE[0], 'Math'),
                     ('science_score', PALETTE[1], 'Science'),
                     ('english_score', PALETTE[2], 'English')]:
    vals = col_floats(col)
    counts, edges = np.histogram(vals, bins=25, range=(0,100))
    centres = [0.5*(edges[i]+edges[i+1]) for i in range(len(counts))]
    ax.plot(centres, counts, linewidth=2, color=c, label=f'{lbl} (μ={sum(vals)/len(vals):.1f})')
ax.set_title('Distribution of Subject Scores', fontsize=13, fontweight='bold')
ax.set_xlabel('Score'); ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)
save(fig, '09_subject_scores_distribution.png')

# 10. Overall score
histogram(col_floats('overall_score'), bins=25,
          title='Distribution of Overall Score',
          xlabel='Overall Score', colour=PALETTE[3],
          fname='10_overall_score_distribution.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 11-15. Categorical distributions ===')

# 11. Final grade
grade_counts = Counter(r['final_grade'] for r in ROWS)
grade_colours = ['#2ca02c','#98df8a','#ffbb78','#ff7f0e','#f7b6d2','#d62728']
bar_chart(grade_counts, 'Distribution of Final Grade', 'Grade',
          '11_final_grade_distribution.png',
          colour=grade_colours, order=GRADE_ORDER)

# 12. Gender
bar_chart(Counter(r['gender'] for r in ROWS),
          'Student Count by Gender', 'Gender',
          '12_gender_distribution.png', colour=PALETTE)

# 13. School type
bar_chart(Counter(r['school_type'] for r in ROWS),
          'Student Count by School Type', 'School Type',
          '13_school_type_distribution.png', colour=PALETTE)

# 14. Parent education
edu_order = ['no formal','high school','diploma','graduate','post graduate','phd']
bar_chart(Counter(r['parent_education'] for r in ROWS),
          'Student Count by Parent Education', 'Education Level',
          '14_parent_education_distribution.png',
          colour=PALETTE, order=edu_order)

# 15. Study method
bar_chart(Counter(r['study_method'] for r in ROWS),
          'Student Count by Study Method', 'Study Method',
          '15_study_method_distribution.png', colour=PALETTE)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 16-18. overall_score across categories ===')

# 16. overall_score by school_type
school_groups = defaultdict(list)
for r in ROWS: school_groups[r['school_type']].append(fv(r,'overall_score'))
box_plot(school_groups, ['private','public'],
         'Overall Score by School Type', 'Overall Score',
         '16_overall_by_school_type.png')

# 17. overall_score by gender
gender_groups = defaultdict(list)
for r in ROWS: gender_groups[r['gender']].append(fv(r,'overall_score'))
box_plot(gender_groups, ['female','male','other'],
         'Overall Score by Gender', 'Overall Score',
         '17_overall_by_gender.png', colours=PALETTE)

# 18. overall_score by parent_education
edu_groups = defaultdict(list)
for r in ROWS: edu_groups[r['parent_education']].append(fv(r,'overall_score'))
box_plot(edu_groups, edu_order,
         'Overall Score by Parent Education Level', 'Overall Score',
         '18_overall_by_parent_education.png', colours=PALETTE)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 19. study_hours vs overall_score ===')
sh_vals  = col_floats('study_hours')
ov_vals  = col_floats('overall_score')
r_val    = pearson(sh_vals, ov_vals)
print(f'  Pearson r = {r_val:.4f}')

fig, ax = plt.subplots(figsize=(8,5))
# Bin study_hours into 0.5-hr buckets, show mean+std
buckets = defaultdict(list)
for sh, ov in zip(sh_vals, ov_vals):
    b = round(sh * 2) / 2
    buckets[b].append(ov)
bkeys = sorted(buckets)
bmeans = [sum(buckets[b])/len(buckets[b]) for b in bkeys]
bstds  = [math.sqrt(sum((x-sum(buckets[b])/len(buckets[b]))**2
           for x in buckets[b])/len(buckets[b])) for b in bkeys]
ax.errorbar(bkeys, bmeans, yerr=bstds, fmt='o-', color=PALETTE[0],
            ecolor='#aac4e8', capsize=4, linewidth=2, markersize=5,
            label=f'Mean overall score (r={r_val:.3f})')
ax.set_title('Study Hours vs Overall Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Study Hours per Day'); ax.set_ylabel('Overall Score (mean ± std)')
ax.legend(); ax.grid(alpha=0.3)
save(fig, '19_study_hours_vs_overall_score.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 20. attendance_percentage vs overall_score ===')
att_vals = col_floats('attendance_percentage')
r_att    = pearson(att_vals, ov_vals)
print(f'  Pearson r = {r_att:.4f}')

fig, ax = plt.subplots(figsize=(8,5))
att_buckets = defaultdict(list)
for a, ov in zip(att_vals, ov_vals):
    b = round(a / 5) * 5
    att_buckets[b].append(ov)
akeys  = sorted(att_buckets)
ameans = [sum(att_buckets[b])/len(att_buckets[b]) for b in akeys]
astds  = [math.sqrt(sum((x-sum(att_buckets[b])/len(att_buckets[b]))**2
           for x in att_buckets[b])/len(att_buckets[b])) for b in akeys]
ax.errorbar(akeys, ameans, yerr=astds, fmt='s-', color=PALETTE[2],
            ecolor='#aadec0', capsize=4, linewidth=2, markersize=5,
            label=f'Mean overall score (r={r_att:.3f})')
ax.set_title('Attendance Percentage vs Overall Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Attendance (%)'); ax.set_ylabel('Overall Score (mean ± std)')
ax.legend(); ax.grid(alpha=0.3)
save(fig, '20_attendance_vs_overall_score.png')

# ═══════════════════════════════════════════════════════════════════════════════
print('\n=== 21. Correlation matrix ===')
corr_cols = ['age','study_hours','attendance_percentage',
             'math_score','science_score','english_score','overall_score']
col_data = {c: col_floats(c) for c in corr_cols}

n_c = len(corr_cols)
corr_matrix = [[0.0]*n_c for _ in range(n_c)]
for i, ci in enumerate(corr_cols):
    for j, cj in enumerate(corr_cols):
        corr_matrix[i][j] = pearson(col_data[ci], col_data[cj])

print('\n  Correlation matrix:')
header_line = '  ' + ''.join(f'{c[:8]:>10}' for c in corr_cols)
print(header_line)
for i, ci in enumerate(corr_cols):
    row_line = f'  {ci[:14]:16}' + ''.join(f'{corr_matrix[i][j]:10.3f}' for j in range(n_c))
    print(row_line)

fig, ax = plt.subplots(figsize=(9,7))
mat = np.array(corr_matrix)
im  = ax.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
short = [c.replace('_percentage','_%').replace('_score','') for c in corr_cols]
ax.set_xticks(range(n_c)); ax.set_xticklabels(short, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(n_c)); ax.set_yticklabels(short, fontsize=9)
for i in range(n_c):
    for j in range(n_c):
        v = corr_matrix[i][j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=8, color='black' if abs(v)<0.6 else 'white',
                fontweight='bold' if abs(v)>0.7 else 'normal')
ax.set_title('Correlation Matrix — Numeric Features', fontsize=13, fontweight='bold')
save(fig, '21_correlation_matrix.png')

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Grade distribution by study_method
print('\n=== BONUS: Grade distribution by study method ===')
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
methods = ['coaching','group study','mixed','notes','online videos','textbook']
for ax, method in zip(axes.flat, methods):
    gc = Counter(r['final_grade'] for r in ROWS if r['study_method']==method)
    vals = [gc.get(g, 0) for g in GRADE_ORDER]
    ax.bar(GRADE_ORDER, vals, color=grade_colours, edgecolor='white')
    ax.set_title(method.title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('Grade'); ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Grade Distribution by Study Method', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, '22_grade_by_study_method.png')

# BONUS: overall_score by study_method
sm_groups = defaultdict(list)
for r in ROWS: sm_groups[r['study_method']].append(fv(r,'overall_score'))
box_plot(sm_groups, methods,
         'Overall Score by Study Method', 'Overall Score',
         '23_overall_by_study_method.png', colours=PALETTE)

# ═══════════════════════════════════════════════════════════════════════════════
# Print key numeric findings for the summary
print('\n=== KEY FINDINGS ===')

# Grade counts
print('\nFinal grade counts:')
for g in GRADE_ORDER:
    print(f'  {g.upper()}: {grade_counts[g]}  ({100*grade_counts[g]/len(ROWS):.1f}%)')

# Mean overall score by category
print('\nMean overall score by school_type:')
for k,v in school_groups.items():
    print(f'  {k}: {sum(v)/len(v):.2f}')

print('\nMean overall score by gender:')
for k,v in gender_groups.items():
    print(f'  {k}: {sum(v)/len(v):.2f}')

print('\nMean overall score by parent_education:')
for k in edu_order:
    v = edu_groups[k]
    print(f'  {k}: {sum(v)/len(v):.2f}')

print('\nMean overall score by study_method:')
for k,v in sm_groups.items():
    print(f'  {k}: {sum(v)/len(v):.2f}')

print('\nPearson correlations with overall_score:')
for col in ['age','study_hours','attendance_percentage',
            'math_score','science_score','english_score']:
    r = pearson(col_data[col], col_data['overall_score'])
    print(f'  {col}: r = {r:.4f}')

print(f'\nAll charts saved to: {CHART_DIR}/')
print('EDA complete.')
