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

    def _ask_llm(self, prompt: str, system_prompt: str = "") -> str:
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
                options={"temperature": 0.1, "num_predict": 512},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return ""

    # -------------------------------------------------------
    # Tool: Suggest column mapping
    # -------------------------------------------------------
    def suggest_column_mapping(
        self, cols1: list[str], cols2: list[str]
    ) -> dict[str, str]:
        """
        Use the LLM to suggest which columns in file1 map to columns in file2.
        Falls back to exact name matching if LLM is unavailable.
        """
        # Rule-based fallback (always runs first as baseline)
        mapping = {}
        for c1 in cols1:
            if c1 in cols2:
                mapping[c1] = c1
            else:
                # Common synonyms
                synonyms = {
                    "publisher name": "tactic name",
                    "tactic name": "publisher name",
                    "channel": "channel name",
                    "channel name": "channel",
                    "vendor": "partner name",
                    "partner name": "vendor",
                    "supplier": "partner name",
                    "media partner": "partner name",
                }
                if c1 in synonyms and synonyms[c1] in cols2:
                    mapping[c1] = synonyms[c1]

        # LLM enhancement
        if self.available:
            prompt = f"""Given two CSV files with these columns:
File 1 columns: {json.dumps(cols1)}
File 2 columns: {json.dumps(cols2)}

Which columns from File 1 correspond to columns in File 2?
They might have different names but represent the same data.
Return ONLY a JSON object like: {{"file1_col": "file2_col", ...}}
Only include confident matches."""

            try:
                response = self._ask_llm(prompt, "You are a data mapping assistant. Return only JSON.")
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
        """
        df1 = load_csv(file1)
        df2 = load_csv(file2)

        f1_samples = df1[f1_start_col].dropna().head(5).astype(str).tolist()
        f2_samples = df2[f2_range_col].dropna().head(5).astype(str).tolist()

        return {
            "file1_start_col": f1_start_col,
            "file1_end_col": f1_end_col,
            "file1_date_format": detect_date_format(f1_samples),
            "file2_range_col": f2_range_col,
            "file2_date_format": detect_date_format(f2_samples),
        }

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
        dataframes: {"File 1": df1, "File 2": df2, "Output": output_df}
        """
        # Build context from dataframes
        context_parts = []
        for name, df in dataframes.items():
            if df is not None and not df.empty:
                context_parts.append(
                    f"--- {name} ---\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Total rows: {len(df)}\n"
                    f"Data:\n{df.to_string(index=False, max_rows=50)}\n"
                )

        context = "\n".join(context_parts)

        if not self.available:
            return self._rule_based_qa(question, dataframes)

        prompt = f"""Answer the following question based ONLY on the data provided.
If you cannot find the answer in the data, say "I was unable to generate an answer based on the available data."

Data:
{context}

Question: {question}

Answer concisely:"""

        response = self._ask_llm(
            prompt,
            "You are a data analysis assistant. Answer questions about CSV data concisely."
        )
        if not response:
            return "I was unable to generate an answer based on the available data."
        return response

    def _rule_based_qa(
        self, question: str, dataframes: dict[str, pd.DataFrame]
    ) -> str:
        """Simple rule-based Q&A fallback when LLM is not available."""
        q = question.lower()
        output_df = dataframes.get("Output")

        if output_df is not None and not output_df.empty:
            if "missing" in q:
                missing = output_df[output_df["comment"].str.contains("Missing", na=False)]
                if not missing.empty:
                    return f"There are {len(missing)} missing records:\n{missing.to_string(index=False)}"
                return "No missing records found."

            if "mismatch" in q or "date" in q:
                mismatches = output_df[output_df["comment"].str.contains("mismatch", na=False)]
                if not mismatches.empty:
                    return f"There are {len(mismatches)} date mismatches:\n{mismatches.to_string(index=False)}"
                return "No date mismatches found."

            if "how many" in q or "count" in q or "total" in q:
                return f"Total issues found: {len(output_df)}"

        return "I was unable to generate an answer based on the available data."
