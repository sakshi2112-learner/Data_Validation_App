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


def _detect_header_row(filepath: str, skip_rows: int = 0) -> int:
    """
    Auto-detect which row contains the actual column headers.
    Scans the first 20 rows and finds the row where most cells
    are non-empty text strings (column names), not numbers or blanks.
    Returns the 0-based row index to use as header.
    """
    try:
        if filepath.lower().endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(filepath, header=None, nrows=20)
        else:
            raw = pd.read_csv(filepath, header=None, nrows=20)
    except Exception:
        return skip_rows

    best_row = 0
    best_score = 0

    for i in range(len(raw)):
        row = raw.iloc[i]
        total = len(row)
        if total == 0:
            continue

        # Count cells that look like column headers:
        # non-empty, text (not pure numbers), reasonable length
        text_count = 0
        for val in row:
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s and not s.replace('.', '').replace(',', '').isdigit() and len(s) < 60:
                text_count += 1

        # Score = ratio of text cells to total cells
        score = text_count / total
        if text_count >= 3 and score > best_score:
            best_score = score
            best_row = i

    return best_row


def smart_read_file(filepath: str, skip_rows: int = None) -> pd.DataFrame:
    """
    Read a CSV or Excel file with smart header detection.
    If skip_rows is provided, uses that. Otherwise auto-detects the header row.
    Handles merged cells in Excel by forward-filling NaN values.
    """
    if skip_rows is None:
        header_row = _detect_header_row(filepath)
    else:
        header_row = skip_rows

    is_excel = filepath.lower().endswith(('.xlsx', '.xls'))

    if is_excel:
        df = pd.read_excel(filepath, header=header_row)
    else:
        df = pd.read_csv(filepath, header=header_row)

    # Handle merged/blank cells in any file type:
    # When pandas reads merged cells (or blank repeated values in CSV),
    # only the first row gets the value, the rest become NaN.
    # Forward-fill fixes this by copying the value down.
    df = df.ffill()

    # Drop fully empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Normalize column names
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def load_csv(filepath: str, skip_rows: int = None) -> pd.DataFrame:
    """Load a CSV or Excel file and normalize column names."""
    return smart_read_file(filepath, skip_rows)


def get_columns(filepath: str, skip_rows: int = None) -> list[str]:
    """Return the column names of a CSV or Excel file."""
    df = smart_read_file(filepath, skip_rows)
    return [c for c in df.columns.tolist() if c and c != 'nan']


def detect_date_format(sample_values: list[str]) -> str:
    """
    Auto-detect the date format from sample values.
    Returns: 'Mon-Mon' for month ranges like 'Jan - Jun', 'date' for any date, 'unknown'.
    """
    if not sample_values:
        return "unknown"

    sample = str(sample_values[0]).strip()

    # Check for month-range format like "Jan-Dec" or "Jan - Dec"
    cleaned = sample.replace(" ", "")
    if re.match(r'^[A-Za-z]{3}-[A-Za-z]{3}$', cleaned):
        return "Mon-Mon"

    # Try to parse as a date using pandas (handles ANY date format)
    try:
        pd.to_datetime(sample)
        return "date"
    except Exception:
        pass

    return "unknown"


def parse_date_to_month(value, fmt: str = "date") -> Optional[int]:
    """Parse any date value to its month number. Uses pandas — works with any format."""
    try:
        return pd.to_datetime(str(value).strip()).month
    except Exception:
        return None


def parse_month_range(flight_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse 'Jan-Dec' or 'Jan - Dec' style string into (start_month, end_month)."""
    text = str(flight_str).strip()
    parts = [p.strip() for p in text.split("-")]
    if len(parts) == 2:
        start = MONTH_MAP.get(parts[0].lower())
        end = MONTH_MAP.get(parts[1].lower())
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
    source_file_name: str = "File 1",
    target_file_name: str = "File 2",
) -> list[dict]:
    """
    Find records in df1 that are missing from df2.

    key_mapping: {df1_col: df2_col} — maps column names between files.
    source_file_name: Name of the file where the record exists.
    target_file_name: Name of the file where the record is missing.
    Returns list of dicts with missing record info + descriptive comment.
    """
    results = []

    for _, row in df1.iterrows():
        conditions = True
        for k1, k2 in key_mapping.items():
            conditions = conditions & (df2[k2] == row[k1])
        match = df2[conditions]

        if match.empty:
            record = {col: row.get(col, "") for col in df1.columns}
            record["comment"] = f"Missing from {target_file_name}"
            results.append(record)

    return results


def validate_dates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_mapping: dict[str, str],
    date_config: dict,
    file1_name: str = "File 1",
    file2_name: str = "File 2",
) -> list[dict]:
    """
    Compare date ranges between matching records.
    Columns can come from either file — we look them up flexibly.

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

    f1_start = date_config.get("file1_start_col", "")
    f1_end = date_config.get("file1_end_col", "")
    f1_fmt = date_config.get("file1_date_format", "")
    f2_range = date_config.get("file2_range_col", "")
    f2_fmt = date_config.get("file2_date_format", "")

    # Helper: get a value from whichever row has the column
    def _get_val(col, row1, row2):
        if col in row1.index:
            return row1.get(col, "")
        if col in row2.index:
            return row2.get(col, "")
        return ""

    for _, row in df1.iterrows():
        conditions = True
        for k1, k2 in key_mapping.items():
            conditions = conditions & (df2[k2] == row[k1])
        match = df2[conditions]

        if match.empty:
            continue  # Missing records handled separately

        matched_row = match.iloc[0]

        # Parse actual dates (start/end columns — could be in either file)
        start_val = _get_val(f1_start, row, matched_row)
        end_val = _get_val(f1_end, row, matched_row)

        if f1_fmt == "date":
            actual_start = parse_date_to_month(start_val, f1_fmt)
            actual_end = parse_date_to_month(end_val, f1_fmt)
        elif f1_fmt == "Mon-Mon":
            actual_start, actual_end = parse_month_range(start_val)
        else:
            continue

        # Parse expected (range column — could be in either file)
        range_val = _get_val(f2_range, row, matched_row)

        if f2_fmt == "Mon-Mon":
            expected_start, expected_end = parse_month_range(range_val)
        elif f2_fmt == "date":
            expected_start = parse_date_to_month(range_val, f2_fmt)
            expected_end = expected_start
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
    output_columns: list[str],
    date_config: Optional[dict] = None,
    output_path: str = "comparison_output.csv",
) -> str:
    """
    Full comparison pipeline.

    Args:
        file1: Path to first CSV.
        file2: Path to second CSV.
        key_mapping: {file1_col: file2_col} for matching rows.
        output_columns: Columns to include in output (only checked/mapped ones).
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

    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)

    all_results = []

    # 1. Missing from file2 — records exist in file1 but not in file2
    missing_from_2 = find_missing_records(
        df1, df2, key_mapping,
        source_file_name=file1_name,
        target_file_name=file2_name,
    )
    all_results.extend(missing_from_2)

    # 2. Missing from file1 — records exist in file2 but not in file1
    #    Rename df2 columns to df1 column names so output is consistent
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    col_rename = {v: k for k, v in key_mapping.items()}  # df2_col -> df1_col
    missing_from_1_raw = find_missing_records(
        df2, df1, reverse_mapping,
        source_file_name=file2_name,
        target_file_name=file1_name,
    )
    for r in missing_from_1_raw:
        renamed = {}
        for col, val in r.items():
            renamed[col_rename.get(col, col)] = val
        all_results.append(renamed)

    # 3. Date validation (if configured)
    if date_config:
        date_mismatches = validate_dates(
            df1, df2, key_mapping, date_config,
            file1_name=file1_name,
            file2_name=file2_name,
        )
        all_results.extend(date_mismatches)

    # Build output DataFrame
    output_df = pd.DataFrame(all_results)

    if output_df.empty:
        # Create empty DataFrame with expected columns
        output_df = pd.DataFrame(columns=output_columns + ["comment"])
        output_df.to_csv(output_path, index=False)
        abs_path = os.path.abspath(output_path)
        return f"Comparison complete. 0 issues found. Files match perfectly. Output saved at: {abs_path}"

    # Clean NaN: fill actual NaN and replace string "nan"
    output_df = output_df.fillna("")
    for col in output_df.columns:
        output_df[col] = output_df[col].apply(
            lambda x: "" if str(x).strip().lower() == "nan" else x
        )

    # Build final column order: only user-selected columns + comment
    # output_columns = the columns the user checked in the mapping UI
    final_cols = [c for c in output_columns if c in output_df.columns]

    # Ensure comment column always present
    if "comment" not in final_cols:
        final_cols.append("comment")

    # Ensure all expected columns exist in the DataFrame
    for col in final_cols:
        if col not in output_df.columns:
            output_df[col] = ""

    output_df = output_df[final_cols]
    output_df.to_csv(output_path, index=False)

    abs_path = os.path.abspath(output_path)
    return f"Comparison complete. {len(all_results)} issues found. Output saved at: {abs_path}"
