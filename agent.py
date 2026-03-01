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
    # Tool: Intent classification
    # -------------------------------------------------------
    def _classify_intent(
        self, question: str, dataframes: dict[str, pd.DataFrame]
    ) -> str:
        """
        Classify what the user is asking about.
        Returns: 'file1', 'file2', 'results', or 'general'.
        """
        q = question.lower()

        # Check for actual file names in the dataframe keys
        file1_names = []
        file2_names = []
        for key in dataframes:
            key_lower = key.lower()
            if "file 1" in key_lower or "file1" in key_lower:
                # Extract filename if present, e.g. "File 1 (Flowchart.xlsx)"
                file1_names.append(key_lower)
                if "(" in key and ")" in key:
                    fname = key[key.index("(") + 1:key.index(")")].lower()
                    file1_names.extend([fname, fname.replace(".", " "), fname.split(".")[0]])
            elif "file 2" in key_lower or "file2" in key_lower:
                file2_names.append(key_lower)
                if "(" in key and ")" in key:
                    fname = key[key.index("(") + 1:key.index(")")].lower()
                    file2_names.extend([fname, fname.replace(".", " "), fname.split(".")[0]])

        # Keyword-based intent classification
        file1_keywords = [
            "file 1", "file1", "first file", "flowchart", "flow chart",
        ] + file1_names
        file2_keywords = [
            "file 2", "file2", "second file", "aggregate", "datafeed",
            "data feed", "data file",
        ] + file2_names
        results_keywords = [
            "result", "comparison", "issue", "mismatch", "missing",
            "output", "discrepanc", "error", "problem", "difference",
        ]

        # Count keyword matches for each intent
        f1_score = sum(1 for kw in file1_keywords if kw in q)
        f2_score = sum(1 for kw in file2_keywords if kw in q)
        res_score = sum(1 for kw in results_keywords if kw in q)

        if res_score > f1_score and res_score > f2_score:
            return "results"
        if f1_score > f2_score:
            return "file1"
        if f2_score > f1_score:
            return "file2"

        return "general"

    # -------------------------------------------------------
    # Tool: Build rich context for LLM
    # -------------------------------------------------------
    def _build_rich_context(self, df: pd.DataFrame, df_name: str) -> str:
        """Build rich context with column stats instead of just sample rows."""
        if df is None or df.empty:
            return f"{df_name}: No data loaded."

        parts = [f"--- {df_name} ---"]
        parts.append(f"Shape: {len(df)} rows × {len(df.columns)} columns")
        parts.append(f"Columns: {list(df.columns)}")

        # Column details: type, unique count, top values
        for col in df.columns:
            unique_count = df[col].nunique()
            non_null = df[col].notna().sum()
            parts.append(f"\n  Column '{col}': {unique_count} unique values, {non_null} non-null")
            if unique_count <= 20:
                # Show all unique values if few enough
                vals = df[col].dropna().unique().tolist()
                vals_str = [str(v) for v in vals[:20]]
                parts.append(f"    Values: {', '.join(vals_str)}")
            else:
                # Show top 10 most common
                top = df[col].value_counts().head(10)
                top_str = [f"{v} ({c})" for v, c in zip(top.index, top.values)]
                parts.append(f"    Top 10: {', '.join(top_str)}")

        return "\n".join(parts)

    # -------------------------------------------------------
    # Tool: Pandas code generation (the smart layer)
    # -------------------------------------------------------
    def _pandas_query(
        self, question: str, df: pd.DataFrame, df_name: str
    ) -> str | None:
        """
        Ask the LLM to write pandas code to answer the question.
        Execute it safely and return the result.
        Returns None if code generation or execution fails.
        """
        if df is None or df.empty:
            return None

        # Build column info with ACTUAL VALUES so the LLM sees exact casing
        col_info = []
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 25:
                # Show ALL unique values — critical for exact matching
                vals = df[col].dropna().unique().tolist()
                vals_str = [str(v) for v in vals]
                col_info.append(f"  '{col}': {unique_count} unique values: {vals_str}")
            else:
                # Show top values for high-cardinality columns
                top = df[col].value_counts().head(8)
                top_str = [str(v) for v in top.index.tolist()]
                col_info.append(f"  '{col}': {unique_count} unique values, top values: {top_str}")

        col_context = "\n".join(col_info)

        # Detect if the user wants a list/names vs a count
        q_lower = question.lower()
        wants_list = any(kw in q_lower for kw in [
            "what are", "which", "show", "list", "tell me", "name",
            "give me", "display", "all the", "all unique",
        ])
        wants_count = any(kw in q_lower for kw in [
            "how many", "count", "total", "number of",
        ])

        if wants_list and not wants_count:
            output_hint = "Return a LIST of values (use .unique().tolist() or .tolist()), NOT a count."
        elif wants_count:
            output_hint = "Return a NUMBER (use .nunique() for unique count or len() for total count)."
        else:
            output_hint = "Return the most appropriate result for the question."

        prompt = f"""You have a pandas DataFrame `df` with {len(df)} rows and these columns:
{col_context}

Question: "{question}"

Write ONE pandas expression to answer this. Rules:
- Use `df` directly (it's already defined)
- IMPORTANT: Use the EXACT column names and values shown above (they are case-sensitive!)
- For text filtering, use .str.contains('value', case=False, na=False) for safety
- {output_hint}
- Output ONLY the code expression, nothing else
- No imports, no print, no comments

Code:"""

        code = self._generate_and_clean_code(prompt)
        if not code:
            return None

        print(f"[Pandas Query] Generated code: {code}")

        # Execute the code
        result = self._safe_exec(code, df)

        # Safety net: if result is 0 or empty but the question mentions data values,
        # retry with a more explicit prompt
        if result is not None and self._is_empty_result(result):
            # Try once more with a simpler approach
            retry_prompt = f"""DataFrame `df` columns and values:
{col_context}

Question: "{question}"

The previous code returned 0/empty. Write a DIFFERENT pandas expression.
Use .str.contains('keyword', case=False, na=False) for ALL text comparisons.
Output ONLY the code, nothing else:"""

            retry_code = self._generate_and_clean_code(retry_prompt)
            if retry_code and retry_code != code:
                print(f"[Pandas Query] Retry code: {retry_code}")
                retry_result = self._safe_exec(retry_code, df)
                if retry_result is not None and not self._is_empty_result(retry_result):
                    return self._format_result(retry_result)

        if result is not None:
            return self._format_result(result)

        return None

    def _generate_and_clean_code(self, prompt: str) -> str | None:
        """Ask LLM for pandas code and clean the response."""
        response = self._ask_llm(
            prompt,
            "You are a pandas code generator. Output ONLY one line of executable pandas code. No explanations, no markdown, no comments, no extra text.",
            max_tokens=150,
        )
        if not response:
            return None

        code = response.strip()

        # Remove markdown code blocks if present
        if "```" in code:
            lines = code.split("\n")
            code_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    code_lines.append(line)
            code = "\n".join(code_lines).strip() if code_lines else code

        # Take only the first meaningful line
        for line in code.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("import"):
                code = line
                break

        # Remove any print() wrapping
        if code.startswith("print(") and code.endswith(")"):
            code = code[6:-1]

        # Safety check
        dangerous = ["import ", "exec(", "eval(", "open(", "os.", "sys.",
                      "subprocess", "shutil", "__", "globals", "locals",
                      "delattr", "setattr", "getattr", "compile", "input("]
        if any(d in code for d in dangerous):
            return None

        return code

    def _safe_exec(self, code: str, df: pd.DataFrame):
        """Execute pandas code safely. Returns result or None."""
        try:
            namespace = {"df": df, "pd": pd}
            result = eval(code, {"__builtins__": {}}, namespace)
            return result
        except Exception as e:
            print(f"[Pandas Query] Code failed: {code} | Error: {e}")
            return None

    def _is_empty_result(self, result) -> bool:
        """Check if a result is effectively empty/zero."""
        if result is None:
            return True
        if isinstance(result, (int, float)) and result == 0:
            return True
        if isinstance(result, pd.DataFrame) and result.empty:
            return True
        if isinstance(result, pd.Series) and result.empty:
            return True
        if isinstance(result, (list, set)) and len(result) == 0:
            return True
        return False

    def _format_result(self, result) -> str | None:
        """Format a pandas query result into a readable string."""
        if result is None:
            return None

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "No matching records found."
            return f"Found {len(result)} records:\n{result.to_string(index=False, max_rows=25)}"
        elif isinstance(result, pd.Series):
            if result.empty:
                return "No matching data found."
            return result.to_string()
        elif isinstance(result, (list, set)):
            items = list(result)
            if not items:
                return "No results found."
            return f"Found {len(items)} items: {', '.join(str(x) for x in items[:50])}"
        elif isinstance(result, (int, float)):
            return str(result)
        else:
            return str(result)

    # -------------------------------------------------------
    # Tool: Results-specific Q&A (fast path)
    # -------------------------------------------------------
    def _results_qa(
        self, question: str, output_df: pd.DataFrame
    ) -> str | None:
        """Fast rule-based answers for comparison results questions."""
        if output_df is None or output_df.empty:
            return None

        q = question.lower()

        if "missing" in q:
            missing = output_df[output_df["comment"].str.contains("Missing from", case=False, na=False)]
            if not missing.empty:
                return f"There are {len(missing)} missing records:\n{missing.to_string(index=False)}"
            return "No missing records found."

        if "mismatch" in q or ("date" in q and ("issue" in q or "problem" in q or "wrong" in q or "error" in q)):
            mismatches = output_df[output_df["comment"].str.contains("mismatch", case=False, na=False)]
            if not mismatches.empty:
                return f"There are {len(mismatches)} date mismatches:\n{mismatches.to_string(index=False)}"
            return "No date mismatches found."

        # Generic count questions about results
        if any(kw in q for kw in ["how many", "count", "total", "number of"]):
            total = len(output_df)
            missing_c = len(output_df[output_df["comment"].str.contains("Missing from", case=False, na=False)])
            mismatch_c = len(output_df[output_df["comment"].str.contains("mismatch", case=False, na=False)])
            return f"Total issues: {total}\n- Missing records: {missing_c}\n- Date mismatches: {mismatch_c}"

        if any(kw in q for kw in ["summary", "overview", "all issue", "all result", "show result"]):
            total = len(output_df)
            missing_c = len(output_df[output_df["comment"].str.contains("Missing from", case=False, na=False)])
            mismatch_c = len(output_df[output_df["comment"].str.contains("mismatch", case=False, na=False)])
            return (
                f"Comparison found {total} issues:\n"
                f"- Missing records: {missing_c}\n"
                f"- Date mismatches: {mismatch_c}\n\n"
                f"{output_df.to_string(index=False, max_rows=25)}"
            )

        # Not a clearly results-specific question
        return None

    # -------------------------------------------------------
    # Tool: Q&A over loaded data (SMART orchestrator)
    # -------------------------------------------------------
    def query_data(
        self,
        question: str,
        dataframes: dict[str, pd.DataFrame],
    ) -> str:
        """
        Smart Q&A orchestrator.
        1. Classify intent (what is the user asking about?)
        2. Route to the right data source
        3. Try pandas code generation for data questions
        4. Fall back to LLM with rich context
        """
        q = question.lower().strip()

        # Find dataframes by role
        output_df = None
        df1 = None
        df2 = None
        df1_name = "File 1"
        df2_name = "File 2"

        for key, df in dataframes.items():
            key_lower = key.lower()
            if "output" in key_lower or "result" in key_lower:
                output_df = df
            elif "file 1" in key_lower or "file1" in key_lower:
                df1 = df
                df1_name = key
            elif "file 2" in key_lower or "file2" in key_lower:
                df2 = df
                df2_name = key

        # Step 1: Classify intent
        intent = self._classify_intent(question, dataframes)
        print(f"[Agent] Intent: {intent} | Question: {question}")

        # Step 2: Route based on intent
        if intent == "results":
            # Try rule-based results first (instant)
            rule_answer = self._results_qa(question, output_df)
            if rule_answer:
                return rule_answer
            # Try pandas code-gen on results
            if output_df is not None and not output_df.empty and self.available:
                pandas_answer = self._pandas_query(question, output_df, "Comparison Results")
                if pandas_answer:
                    return pandas_answer

        elif intent == "file1":
            if df1 is not None and not df1.empty:
                # Try pandas code-gen on File 1
                if self.available:
                    pandas_answer = self._pandas_query(question, df1, df1_name)
                    if pandas_answer:
                        return pandas_answer
                # Fallback: basic file info
                return self._basic_file_info(df1, df1_name)
            return "File 1 is not loaded. Please load files first."

        elif intent == "file2":
            if df2 is not None and not df2.empty:
                # Try pandas code-gen on File 2
                if self.available:
                    pandas_answer = self._pandas_query(question, df2, df2_name)
                    if pandas_answer:
                        return pandas_answer
                # Fallback: basic file info
                return self._basic_file_info(df2, df2_name)
            return "File 2 is not loaded. Please load files first."

        else:  # general
            # Try all sources
            # First try files with pandas code-gen
            for df, name in [(df1, df1_name), (df2, df2_name), (output_df, "Comparison Results")]:
                if df is not None and not df.empty and self.available:
                    pandas_answer = self._pandas_query(question, df, name)
                    if pandas_answer:
                        return f"From {name}:\n{pandas_answer}"

            # Try rule-based results
            if output_df is not None:
                rule_answer = self._results_qa(question, output_df)
                if rule_answer:
                    return rule_answer

        # Step 3: LLM fallback with rich context
        if self.available:
            return self._llm_fallback(question, intent, dataframes)

        # Step 4: No LLM available — provide basic info
        return self._no_llm_fallback(question, dataframes)

    def _basic_file_info(self, df: pd.DataFrame, name: str) -> str:
        """Basic file information when we can't do code-gen."""
        if df is None or df.empty:
            return f"{name}: No data."
        parts = [f"{name}: {len(df)} rows, {len(df.columns)} columns"]
        parts.append(f"Columns: {', '.join(df.columns)}")
        # Show unique value counts for each column
        for col in df.columns:
            unique = df[col].nunique()
            parts.append(f"  {col}: {unique} unique values")
        return "\n".join(parts)

    def _llm_fallback(
        self, question: str, intent: str, dataframes: dict[str, pd.DataFrame]
    ) -> str:
        """LLM fallback with rich context (column stats, value counts)."""
        context_parts = []

        # Build rich context for relevant dataframes
        for key, df in dataframes.items():
            if df is not None and not df.empty:
                key_lower = key.lower()
                # Only include relevant context based on intent
                if intent == "file1" and "file 1" not in key_lower:
                    continue
                if intent == "file2" and "file 2" not in key_lower:
                    continue
                if intent == "results" and "output" not in key_lower and "result" not in key_lower:
                    continue

                context_parts.append(self._build_rich_context(df, key))

        # If intent filtering gave us nothing, include everything
        if not context_parts:
            for key, df in dataframes.items():
                if df is not None and not df.empty:
                    context_parts.append(self._build_rich_context(df, key))

        context = "\n\n".join(context_parts)

        prompt = f"""You are a data analysis assistant. Answer questions about the CSV data below.
IMPORTANT: Base your answer ONLY on the actual data provided. Be precise with numbers.
If the question is not about this data, say "I can only answer questions about your loaded CSV files and comparison results."

Data Context:
{context}

Question: {question}

Answer clearly and concisely:"""

        response = self._ask_llm(
            prompt,
            "You are a precise CSV data analysis assistant. Answer based ONLY on the provided data. Give exact numbers and specific details. Be concise but complete.",
            max_tokens=300,
        )
        if not response:
            return "I could not generate an answer. Please try rephrasing your question."
        return response

    def _no_llm_fallback(
        self, question: str, dataframes: dict[str, pd.DataFrame]
    ) -> str:
        """Fallback when LLM is not available — basic info from data."""
        q = question.lower()

        # Try to give some useful info even without LLM
        parts = []
        for key, df in dataframes.items():
            if df is not None and not df.empty:
                parts.append(f"{key}: {len(df)} rows, {len(df.columns)} columns")
                parts.append(f"  Columns: {', '.join(df.columns)}")

        if parts:
            return "Here's what I know about your data:\n" + "\n".join(parts) + \
                   "\n\nNote: LLM is offline. For detailed answers, please ensure Ollama is running with phi3:mini."

        return "No files loaded. Please load files first in the FILES tab."
