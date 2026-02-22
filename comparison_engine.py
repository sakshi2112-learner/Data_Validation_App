"""
Flexible CSV Comparison Engine
No hardcoded column names, date formats, or file structures.
Everything is parameterized and driven by the agent/GUI.
"""

import pandas as pd
import os
import re
from datetime import datetime
from typing import Optional


# -----------------------------------------------------------
# Month mapping for flexible date parsing
# -----------------------------------------------------------
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}
MONTH_NAMES = {v: k.capitalize() for k, v in MONTH_MAP.items()}


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV and normalize column names (strip + lowercase)."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_columns(filepath: str) -> list[str]:
    """Return the column names of a CSV file."""
    df = pd.read_csv(filepath, nrows=0)
    return [c.strip().lower() for c in df.columns.tolist()]


def detect_date_format(sample_values: list[str]) -> str:
    """
    Auto-detect the date format from sample values.
    Returns one of: 'DD-MM-YYYY', 'MM-DD-YYYY', 'YYYY-MM-DD', 'Mon-Mon', 'unknown'
    """
    if not sample_values:
        return "unknown"

    sample = str(sample_values[0]).strip()

    # Check for month-range format like "Jan-Dec"
    if re.match(r'^[A-Za-z]{3}-[A-Za-z]{3}$', sample):
        return "Mon-Mon"

    # Check for date formats
    for fmt, label in [
        ("%d-%m-%Y", "DD-MM-YYYY"),
        ("%m-%d-%Y", "MM-DD-YYYY"),
        ("%Y-%m-%d", "YYYY-MM-DD"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
        ("%m/%d/%Y", "MM/DD/YYYY"),
    ]:
        try:
            datetime.strptime(sample, fmt)
            return label
        except ValueError:
            continue

    return "unknown"


def parse_date_to_month(value: str, fmt: str) -> Optional[int]:
    """Parse a date string to its month number using the given format."""
    fmt_map = {
        "DD-MM-YYYY": "%d-%m-%Y",
        "MM-DD-YYYY": "%m-%d-%Y",
        "YYYY-MM-DD": "%Y-%m-%d",
        "DD/MM/YYYY": "%d/%m/%Y",
        "MM/DD/YYYY": "%m/%d/%Y",
    }
    py_fmt = fmt_map.get(fmt)
    if py_fmt:
        try:
            return datetime.strptime(str(value).strip(), py_fmt).month
        except (ValueError, TypeError):
            return None
    return None


def parse_month_range(flight_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse a 'Jan-Dec' style string into (start_month, end_month)."""
    parts = str(flight_str).strip().split("-")
    if len(parts) == 2:
        start = MONTH_MAP.get(parts[0].strip().lower())
        end = MONTH_MAP.get(parts[1].strip().lower())
        return start, end
    return None, None


def month_to_abbr(month_num: Optional[int]) -> str:
    """Convert month number to 3-letter abbreviation."""
    if month_num and month_num in MONTH_NAMES:
        return MONTH_NAMES[month_num]
    return "?"


def find_missing_records(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_mapping: dict[str, str],
) -> list[dict]:
    """
    Find records in df1 that are missing from df2.

    key_mapping: {df1_col: df2_col} — maps column names between files.
    Returns list of dicts with missing record info.
    """
    results = []
    df1_keys = list(key_mapping.keys())
    df2_keys = list(key_mapping.values())

    for _, row in df1.iterrows():
        conditions = True
        for k1, k2 in key_mapping.items():
            conditions = conditions & (df2[k2] == row[k1])
        match = df2[conditions]

        if match.empty:
            record = {col: row.get(col, "") for col in df1.columns}
            results.append(record)

    return results


def validate_dates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_mapping: dict[str, str],
    date_config: dict,
) -> list[dict]:
    """
    Compare date ranges between matching records.

    date_config example:
    {
        "file1_start_col": "min date",
        "file1_end_col": "max date",
        "file1_date_format": "DD-MM-YYYY",
        "file2_range_col": "flight",
        "file2_date_format": "Mon-Mon",
    }
    """
    results = []
    df1_keys = list(key_mapping.keys())

    f1_start = date_config.get("file1_start_col", "")
    f1_end = date_config.get("file1_end_col", "")
    f1_fmt = date_config.get("file1_date_format", "")
    f2_range = date_config.get("file2_range_col", "")
    f2_fmt = date_config.get("file2_date_format", "")

    for _, row in df1.iterrows():
        conditions = True
        for k1, k2 in key_mapping.items():
            conditions = conditions & (df2[k2] == row[k1])
        match = df2[conditions]

        if match.empty:
            continue  # Missing records handled separately

        matched_row = match.iloc[0]

        # Parse dates from file1
        if f1_fmt in ("DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"):
            actual_start = parse_date_to_month(row.get(f1_start, ""), f1_fmt)
            actual_end = parse_date_to_month(row.get(f1_end, ""), f1_fmt)
        elif f1_fmt == "Mon-Mon":
            actual_start, actual_end = parse_month_range(row.get(f1_start, ""))
        else:
            continue

        # Parse expected from file2
        if f2_fmt == "Mon-Mon":
            expected_start, expected_end = parse_month_range(matched_row.get(f2_range, ""))
        elif f2_fmt in ("DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"):
            expected_start = parse_date_to_month(matched_row.get(f2_range, ""), f2_fmt)
            expected_end = expected_start  # single date
        else:
            continue

        # Compare
        if actual_start != expected_start or actual_end != expected_end:
            actual_str = f"{month_to_abbr(actual_start)}-{month_to_abbr(actual_end)}"
            expected_str = f"{month_to_abbr(expected_start)}-{month_to_abbr(expected_end)}"

            record = {col: row.get(col, "") for col in df1.columns}
            record["comment"] = (
                f"Date mismatch: Expected {expected_str}, but found {actual_str}"
            )
            results.append(record)

    return results


def run_comparison(
    file1: str,
    file2: str,
    key_mapping: dict[str, str],
    display_columns: list[str],
    date_config: Optional[dict] = None,
    output_path: str = "comparison_output.csv",
) -> str:
    """
    Full comparison pipeline.

    Args:
        file1: Path to first CSV.
        file2: Path to second CSV.
        key_mapping: {file1_col: file2_col} for matching rows.
        display_columns: Columns to include in output.
        date_config: Optional date validation config.
        output_path: Where to save the output CSV.

    Returns:
        Status message.
    """
    if not os.path.exists(file1):
        return f"Error: {file1} not found."
    if not os.path.exists(file2):
        return f"Error: {file2} not found."

    df1 = load_csv(file1)
    df2 = load_csv(file2)

    all_results = []

    # 1. Missing from file2 — records have df1 column names
    missing_from_2 = find_missing_records(df1, df2, key_mapping)
    for r in missing_from_2:
        r["comment"] = f"Missing from {os.path.basename(file2)}"
    all_results.extend(missing_from_2)

    # 2. Missing from file1 — records have df2 column names,
    #    rename them to df1 column names so output is consistent
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    col_rename = {v: k for k, v in key_mapping.items()}  # df2_col -> df1_col
    missing_from_1_raw = find_missing_records(df2, df1, reverse_mapping)
    for r in missing_from_1_raw:
        renamed = {}
        for col, val in r.items():
            renamed[col_rename.get(col, col)] = val
        renamed["comment"] = f"Missing from {os.path.basename(file1)}"
        all_results.append(renamed)

    # 3. Date validation (if configured)
    if date_config:
        date_mismatches = validate_dates(df1, df2, key_mapping, date_config)
        all_results.extend(date_mismatches)

    # Build output DataFrame
    output_df = pd.DataFrame(all_results)

    # Clean NaN: fill actual NaN and replace string "nan"
    output_df = output_df.fillna("")
    for col in output_df.columns:
        output_df[col] = output_df[col].apply(
            lambda x: "" if str(x).strip().lower() == "nan" else x
        )

    # Collect date columns from config
    exclude_cols = set()
    f2_range = ""
    f1_start = ""
    f1_end = ""
    if date_config:
        f2_range = date_config.get("file2_range_col", "")
        f1_start = date_config.get("file1_start_col", "")
        f1_end = date_config.get("file1_end_col", "")
        for col_name in [f2_range, f1_start, f1_end]:
            if col_name:
                exclude_cols.add(col_name)

    # Auto-detect additional date-like columns (no hardcoding)
    for col in output_df.columns:
        if col in exclude_cols or col == "comment":
            continue
        samples = output_df[col].astype(str).tolist()
        samples = [s for s in samples if s and s != ""][:5]
        if samples and detect_date_format(samples) != "unknown":
            exclude_cols.add(col)

    # Build unified "date" column: flight → min/max → other detected date cols
    other_date_cols = [c for c in exclude_cols if c not in {f2_range, f1_start, f1_end}]

    def _build_date_value(row):
        # Priority 1: file2 range column (e.g. flight)
        if f2_range and f2_range in row.index:
            val = str(row[f2_range]).strip()
            if val and val != "":
                return val
        # Priority 2: file1 start + end columns (e.g. min date – max date)
        start_val = ""
        end_val = ""
        if f1_start and f1_start in row.index:
            sv = str(row[f1_start]).strip()
            if sv and sv != "":
                start_val = sv
        if f1_end and f1_end in row.index:
            ev = str(row[f1_end]).strip()
            if ev and ev != "":
                end_val = ev
        if start_val and end_val:
            return f"{start_val} - {end_val}"
        elif start_val:
            return start_val
        elif end_val:
            return end_val
        # Priority 3: any other auto-detected date column
        for dc in other_date_cols:
            if dc in row.index:
                val = str(row[dc]).strip()
                if val and val != "":
                    return val
        return ""

    if exclude_cols:
        output_df["date"] = output_df.apply(_build_date_value, axis=1)

    # Build column order: display cols → extra non-date cols → date → comment
    all_cols = list(output_df.columns)
    output_cols = [c for c in display_columns if c in all_cols and c not in exclude_cols]
    extra_cols = [
        c for c in all_cols
        if c not in output_cols and c not in {"comment", "date"} and c not in exclude_cols
    ]
    output_cols = output_cols + extra_cols
    if "date" in output_df.columns:
        output_cols.append("date")
    output_cols.append("comment")

    # Ensure all expected columns exist
    for col in output_cols:
        if col not in output_df.columns:
            output_df[col] = ""

    output_df = output_df[output_cols]
    output_df.to_csv(output_path, index=False)

    abs_path = os.path.abspath(output_path)
    return f"Comparison complete. {len(all_results)} issues found. Output saved at: {abs_path}"
