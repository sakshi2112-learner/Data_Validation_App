import pandas as pd
from comparison_engine import smart_read_file, validate_dates, month_to_abbr

# Test REVERSED file order: datafeed as File 1, flowchart as File 2
df1 = smart_read_file("test_data/datafeed.csv")       # File 1 = datafeed
df2 = smart_read_file("test_data/Flowchart.xlsx")      # File 2 = flowchart

print("File 1 (datafeed) columns:", list(df1.columns))
print("File 2 (flowchart) columns:", list(df2.columns))
print()

# The date_config stays the same as what the app generated:
date_config = {
    "file1_start_col": "min of atvy_dt",
    "file1_end_col": "max of atvy_dt2",
    "file1_date_format": "date",
    "file2_range_col": "flight",
    "file2_date_format": "Mon-Mon",
}

# key mapping: datafeed cols -> flowchart cols
key_mapping = {"channel": "channel", "ptnr_nm": "partner", "pubhr_prd_nm": "tactic"}

print("=== Datafeed as File 1, Flowchart as File 2, 3 keys ===")
results = validate_dates(df1, df2, key_mapping, date_config)
print(f"Mismatches found: {len(results)}")
for r in results:
    ptnr = r.get('ptnr_nm', '?')
    tactic = r.get('pubhr_prd_nm', '?')
    print(f"  {ptnr:15s} {tactic:15s} -> {r['comment']}")
