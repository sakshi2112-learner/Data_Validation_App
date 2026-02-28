"""
Local LLM Agent for CSV comparison.
Uses Ollama with Phi-3-mini for fully offline operation.
Provides tools for column discovery, mapping, date detection, and Q&A.
"""

import json
import pandas as pd
import ollama
from comparison_engine import (
    get_columns, detect_date_format, load_csv, run_comparison
)


MODEL_NAME = "phi3:mini"


class LocalAgent:
    """
    Local LLM agent that handles CSV comparison tasks.
    Uses Ollama for fully offline inference.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.available = False
        try:
            # Check if the model is available
            ollama.show(self.model_name)
            self.available = True
        except Exception:
            print(f"Warning: Model '{self.model_name}' not available. Agent will use rule-based fallback.")

    def _ask_llm(self, prompt: str, system_prompt: str = "", timeout: int = 120, max_tokens: int = 512) -> str:
        """Send a prompt to the local LLM and return the response."""
        if not self.available:
            return ""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.1, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return ""

    # -------------------------------------------------------
    # Tool: Suggest column mapping
    # -------------------------------------------------------
    def suggest_column_mapping(
        self, cols1: list[str], cols2: list[str],
        df1: pd.DataFrame = None, df2: pd.DataFrame = None,
    ) -> dict[str, str]:
        """
        Use the LLM to suggest which columns in file1 map to columns in file2.
        Falls back to exact name matching if LLM is unavailable.
        Handles 10+ columns by giving LLM sample data for context.
        Excludes date columns from regular mapping (dates are handled separately).
        """
        # Rule-based fallback (always runs first as baseline)
        mapping = {}

        # Expanded synonym dictionary for common media/advertising columns
        synonyms = {
            "publisher name": ["tactic name", "publisher"],
            "tactic name": ["publisher name", "tactic"],
            "channel": ["channel name", "channel_code"],
            "channel name": ["channel", "channel_code"],
            "channel_code": ["channel", "channel name"],
            "vendor": ["partner name", "partner", "supplier", "media partner"],
            "partner name": ["vendor", "partner", "supplier", "media partner"],
            "partner": ["partner name", "vendor"],
            "supplier": ["partner name", "vendor"],
            "media partner": ["partner name", "vendor"],
            "campaign": ["campaign_name", "campaign name"],
            "campaign_name": ["campaign", "campaign name"],
            "campaign name": ["campaign", "campaign_name"],
            "brand": ["brand_name", "brand name"],
            "brand_name": ["brand", "brand name"],
            "brand name": ["brand", "brand_name"],
            "placement": ["placement_name", "placement name"],
            "placement_name": ["placement", "placement name"],
            "placement name": ["placement", "placement_name"],
            "device": ["device_type_name", "device type", "device_type"],
            "device_type_name": ["device", "device type"],
            "publisher_product_name": ["program description", "product name"],
            "program description": ["publisher_product_name", "product name"],
            "activity_code": ["tactic.1", "activity code"],
            "metric_type": ["kpi", "metric"],
            "kpi": ["metric_type", "metric"],
        }

        for c1 in cols1:
            if c1 in cols2:
                mapping[c1] = c1
            else:
                # Check synonyms
                if c1 in synonyms:
                    for syn in synonyms[c1]:
                        if syn in cols2:
                            mapping[c1] = syn
                            break

        # LLM enhancement for better mapping with large column sets
        if self.available:
            # Build sample data context for better LLM understanding
            sample_context = ""
            if df1 is not None and df2 is not None:
                sample1 = df1.head(2).to_string(index=False, max_cols=20)
                sample2 = df2.head(2).to_string(index=False, max_cols=20)
                sample_context = f"""\n\nSample data from File 1:\n{sample1}\n\nSample data from File 2:\n{sample2}"""

            prompt = f"""Given two CSV files with these columns:
File 1 columns ({len(cols1)} total): {json.dumps(cols1)}
File 2 columns ({len(cols2)} total): {json.dumps(cols2)}
{sample_context}

Match columns from File 1 to File 2 that represent the SAME data concept.
They might have different names but represent the same data.
IMPORTANT: Do NOT include date columns (like min date, max date, flight, go live date, etc.) in the mapping. Date columns are handled separately.
Return ONLY a JSON object like: {{"file1_col": "file2_col", ...}}
Only include confident matches."""

            try:
                response = self._ask_llm(prompt, "You are a data mapping assistant. Return only JSON. Do not include date-related columns in the mapping.")
                if "{" in response and "}" in response:
                    json_str = response[response.index("{"):response.rindex("}") + 1]
                    llm_mapping = json.loads(json_str)
                    for k, v in llm_mapping.items():
                        k_lower = k.lower().strip()
                        v_lower = v.lower().strip()
                        if k_lower in cols1 and v_lower in cols2:
                            mapping[k_lower] = v_lower
            except (json.JSONDecodeError, Exception):
                pass  # Use rule-based fallback

        return mapping

    # -------------------------------------------------------
    # Tool: Detect date columns
    # -------------------------------------------------------
    def detect_date_columns(self, filepath: str) -> list[str]:
        """Identify columns that likely contain dates."""
        df = load_csv(filepath)
        date_cols = []

        for col in df.columns:
            sample = df[col].dropna().head(5).astype(str).tolist()
            fmt = detect_date_format(sample)
            if fmt != "unknown":
                date_cols.append(col)

        return date_cols

    # -------------------------------------------------------
    # Tool: Classify date role for a file
    # -------------------------------------------------------
    def classify_date_role(self, filepath: str) -> dict:
        """
        Determine whether a file has individual date columns (min/max)
        or a range column (Mon-Mon flight style).

        Returns:
            {
                "role": "min_max" | "range" | "unknown",
                "start_col": str or None,   # for min_max
                "end_col": str or None,      # for min_max
                "range_col": str or None,    # for range
                "all_date_cols": list[str],
            }
        """
        df = load_csv(filepath)
        date_info = []

        for col in df.columns:
            sample = df[col].dropna().head(5).astype(str).tolist()
            fmt = detect_date_format(sample)
            if fmt != "unknown":
                date_info.append({"col": col, "format": fmt})

        result = {
            "role": "unknown",
            "start_col": None,
            "end_col": None,
            "range_col": None,
            "all_date_cols": [d["col"] for d in date_info],
        }

        # Check for Mon-Mon range column
        range_cols = [d for d in date_info if d["format"] == "Mon-Mon"]
        if range_cols:
            result["role"] = "range"
            result["range_col"] = range_cols[0]["col"]
            return result

        # Check for individual date columns (look for min/max patterns)
        date_only = [d for d in date_info if d["format"] != "Mon-Mon"]
        if len(date_only) >= 2:
            result["role"] = "min_max"
            # Try to identify start/end by name
            start_keywords = ["min", "start", "begin", "from"]
            end_keywords = ["max", "end", "finish", "to"]

            for d in date_only:
                col_lower = d["col"].lower()
                if any(kw in col_lower for kw in start_keywords):
                    result["start_col"] = d["col"]
                elif any(kw in col_lower for kw in end_keywords):
                    result["end_col"] = d["col"]

            # Fallback: first = start, second = end
            if not result["start_col"] and date_only:
                result["start_col"] = date_only[0]["col"]
            if not result["end_col"] and len(date_only) > 1:
                result["end_col"] = date_only[1]["col"]

            return result

        # Single date column — treat as start only
        if len(date_only) == 1:
            result["role"] = "min_max"
            result["start_col"] = date_only[0]["col"]
            result["end_col"] = date_only[0]["col"]

        return result

    # -------------------------------------------------------
    # Tool: Build date config
    # -------------------------------------------------------
    def build_date_config(
        self,
        file1: str,
        file2: str,
        f1_start_col: str,
        f1_end_col: str,
        f2_range_col: str,
    ) -> dict:
        """
        Auto-detect date formats and build the date_config dict
        needed by the comparison engine.
        Columns can come from either file — we find them automatically.
        """
        df1 = load_csv(file1)
        df2 = load_csv(file2)

        # Find start/end date samples from whichever file has the column
        if f1_start_col in df1.columns:
            f1_samples = df1[f1_start_col].dropna().head(5).astype(str).tolist()
        elif f1_start_col in df2.columns:
            f1_samples = df2[f1_start_col].dropna().head(5).astype(str).tolist()
        else:
            f1_samples = []

        # Find range/flight samples from whichever file has the column
        if f2_range_col in df2.columns:
            f2_samples = df2[f2_range_col].dropna().head(5).astype(str).tolist()
        elif f2_range_col in df1.columns:
            f2_samples = df1[f2_range_col].dropna().head(5).astype(str).tolist()
        else:
            f2_samples = []

        return {
            "file1_start_col": f1_start_col,
            "file1_end_col": f1_end_col,
            "file1_date_format": detect_date_format(f1_samples),
            "file2_range_col": f2_range_col,
            "file2_date_format": detect_date_format(f2_samples),
        }

    # -------------------------------------------------------
    # Tool: Enhance comparison comments with AI
    # -------------------------------------------------------
    def enhance_comments(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """
        Use the LLM to rewrite comparison comments to be smarter and more descriptive.
        Sends all results in ONE batch call for speed.
        Returns the dataframe with improved comments.
        """
        if not self.available or output_df is None or output_df.empty:
            return output_df

        # Build a summary of all records with their comments
        records_text = []
        for i, row in output_df.iterrows():
            # Get key column values (all columns except 'comment')
            key_vals = {col: str(row[col]) for col in output_df.columns if col != "comment"}
            key_str = ", ".join(f"{k}={v}" for k, v in key_vals.items() if v and v != "nan")
            records_text.append(f"Row {i}: [{key_str}] → Comment: {row.get('comment', '')}")

        all_records = "\n".join(records_text)

        prompt = f"""Below are comparison results between two CSV files. Each row has key column values and a basic comment.

Rewrite EACH comment to be smarter and more descriptive. Rules:
- For missing records: mention the key values (like partner name, tactic) and which file it's missing from
- For date mismatches: mention the key values and explain the expected vs actual date range clearly
- Keep each comment to 1 sentence, concise but informative
- Return ONLY the rewritten comments, one per line, in the same order as input
- Each line should be JUST the comment text, nothing else

Records:
{all_records}

Rewritten comments (one per line):"""

        response = self._ask_llm(
            prompt,
            "You rewrite data comparison comments to be clear, specific, and professional. Output only the rewritten comments, one per line.",
            max_tokens=512,
        )

        if not response:
            return output_df

        # Parse response — one comment per line
        new_comments = [line.strip() for line in response.strip().split("\n") if line.strip()]

        # Only apply if we got the right number of comments
        if len(new_comments) == len(output_df):
            output_df = output_df.copy()
            output_df["comment"] = new_comments
        else:
            # Try to match as many as possible
            output_df = output_df.copy()
            for i in range(min(len(new_comments), len(output_df))):
                if new_comments[i]:
                    output_df.iloc[i, output_df.columns.get_loc("comment")] = new_comments[i]

        return output_df

    # -------------------------------------------------------
    # Tool: Q&A over loaded data
    # -------------------------------------------------------
    def query_data(
        self,
        question: str,
        dataframes: dict[str, pd.DataFrame],
    ) -> str:
        """
        Answer a question about the loaded CSV data.
        Tries rule-based first (instant), falls back to LLM for complex questions.
        """
        # Try rule-based first — instant answers for common questions
        rule_answer = self._rule_based_qa(question, dataframes)
        if not rule_answer.startswith("I can answer questions about"):
            return rule_answer

        # Fall back to LLM for complex questions
        if not self.available:
            return rule_answer

        # Build minimal context for speed
        context_parts = []
        for name, df in dataframes.items():
            if df is not None and not df.empty:
                context_parts.append(
                    f"--- {name} ---\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Total rows: {len(df)}\n"
                    f"Sample (first 5 rows):\n{df.head(5).to_string(index=False)}\n"
                )

        context = "\n".join(context_parts)

        prompt = f"""You are an assistant that ONLY answers questions about the CSV data provided below.
Do NOT answer general knowledge questions, coding questions, or anything unrelated to this data.
If the question is not about the data, say "I can only answer questions about your loaded CSV files and comparison results."

Data:
{context}

Question: {question}

Answer concisely based ONLY on the data above:"""

        response = self._ask_llm(
            prompt,
            "You are a CSV data analysis assistant. ONLY answer questions about the provided CSV data and comparison results. Refuse any unrelated questions. Be concise.",
            max_tokens=128,
        )
        if not response:
            return "I could not generate an answer. Please try rephrasing your question."
        return response

    def _rule_based_qa(
        self, question: str, dataframes: dict[str, pd.DataFrame]
    ) -> str:
        """Comprehensive rule-based Q&A fallback when LLM is not available."""
        q = question.lower()
        output_df = dataframes.get("Output")
        df1 = dataframes.get("File 1")
        df2 = dataframes.get("File 2")

        # Questions about comparison results
        if output_df is not None and not output_df.empty:
            if "missing" in q:
                missing = output_df[output_df["comment"].str.contains("Missing from", case=False, na=False)]
                if not missing.empty:
                    return f"There are {len(missing)} missing records:\n{missing.to_string(index=False)}"
                return "No missing records found."

            if "mismatch" in q or "date" in q:
                mismatches = output_df[output_df["comment"].str.contains("mismatch", case=False, na=False)]
                if not mismatches.empty:
                    return f"There are {len(mismatches)} date mismatches:\n{mismatches.to_string(index=False)}"
                return "No date mismatches found."

            if "how many" in q or "count" in q or "total" in q:
                total = len(output_df)
                missing_c = len(output_df[output_df["comment"].str.contains("Missing from", case=False, na=False)])
                mismatch_c = len(output_df[output_df["comment"].str.contains("mismatch", case=False, na=False)])
                return f"Total issues: {total}\n- Missing records: {missing_c}\n- Date mismatches: {mismatch_c}"

            if "summary" in q or "result" in q or "issue" in q:
                return f"Comparison found {len(output_df)} issues:\n{output_df.to_string(index=False, max_rows=20)}"

        # Questions about individual files
        if ("column" in q or "field" in q) and df1 is not None:
            cols1 = list(df1.columns) if df1 is not None else []
            cols2 = list(df2.columns) if df2 is not None else []
            return f"File 1 columns ({len(cols1)}): {', '.join(cols1)}\nFile 2 columns ({len(cols2)}): {', '.join(cols2)}"

        if ("row" in q or "record" in q) and ("how many" in q or "count" in q):
            parts = []
            if df1 is not None:
                parts.append(f"File 1: {len(df1)} rows")
            if df2 is not None:
                parts.append(f"File 2: {len(df2)} rows")
            return "\n".join(parts) if parts else "No files loaded."

        if "file" in q and ("info" in q or "about" in q or "detail" in q):
            parts = []
            if df1 is not None:
                parts.append(f"File 1: {len(df1)} rows, {len(df1.columns)} columns")
            if df2 is not None:
                parts.append(f"File 2: {len(df2)} rows, {len(df2.columns)} columns")
            return "\n".join(parts) if parts else "No files loaded."

        return "I can answer questions about: missing records, date mismatches, column names, row counts, and comparison results. Try asking about one of these!"
