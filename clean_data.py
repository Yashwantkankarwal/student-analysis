import csv

input_path = 'Student_Performance.csv'
output_path = 'Student_Performance_Cleaned.csv'

with open(input_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

original_count = len(rows)

# Deduplicate: keep first occurrence of each unique row
seen = set()
cleaned_rows = []
for row in rows:
    key = tuple(row[h] for h in headers)
    if key not in seen:
        seen.add(key)
        cleaned_rows.append(row)

removed_count = original_count - len(cleaned_rows)
final_count = len(cleaned_rows)

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(cleaned_rows)

# Verification
with open(output_path, newline='', encoding='utf-8') as f:
    reader2 = csv.DictReader(f)
    verify_rows = list(reader2)
    verify_headers = reader2.fieldnames

# Missing values
missing_per_col = {h: sum(1 for r in verify_rows if r[h].strip() == '') for h in verify_headers}
total_missing = sum(missing_per_col.values())

# Remaining duplicates
verify_tuples = [tuple(r[h] for h in verify_headers) for r in verify_rows]
remaining_dups = len(verify_tuples) - len(set(verify_tuples))

print("=== DATA CLEANING REPORT ===")
print(f"Original row count:      {original_count}")
print(f"Duplicate rows removed:  {removed_count}")
print(f"Final row count:         {len(verify_rows)}")
print(f"Final column count:      {len(verify_headers)}")
print(f"Columns: {list(verify_headers)}")
print()
print(f"Missing values remaining: {total_missing}")
if total_missing == 0:
    print("  -> No missing values in any column.")
else:
    for h, v in missing_per_col.items():
        if v > 0:
            print(f"  {h}: {v}")
print()
print(f"Duplicate rows remaining: {remaining_dups}")
if remaining_dups == 0:
    print("  -> No duplicates remain.")
print()
print(f"Output file: {output_path}")
