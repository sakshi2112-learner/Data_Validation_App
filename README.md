# ⚡ CSV Compare Agent

A desktop application that compares two CSV files (e.g., a **Flowchart** and an **Aggregate/Data Feed**) to find missing vendor records and date mismatches. Built with Python, it uses a **local AI agent** (Ollama + Phi-3 Mini) for smart column mapping and Q&A — fully offline, no data leaves your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/AI-Ollama%20%2B%20Phi--3-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-brightgreen)

---

## 🎯 What It Does

- **Finds missing records** — identifies vendors/tactics present in one file but missing from the other, with clear comments mentioning the filename
- **Detects date mismatches** — extracts months from start/end date columns and compares against flight/range columns (e.g., `Jan - Jun`), supporting any date format
- **Smart column mapping** — AI suggests how to map columns with different names across the two files; works with 10+ columns
- **Flexible date parsing** — handles any date format (`MM/DD/YY`, `DD-MM-YYYY`, `YYYY-MM-DD`, etc.) using pandas auto-detection
- **Output control** — only columns you explicitly check/map appear in the output CSV, plus a descriptive `comment` column
- **Chat Q&A** — ask natural language questions about your loaded data and comparison results
- **Exports results** — save the comparison output as a CSV with vendor details and issue comments

---

## 🖥️ Screenshots

The app features a gamer-style dark theme with neon accents, organized into 4 tabs:

| Tab | Purpose |
|-----|---------|
| 📂 **FILES** | Load your Flowchart and Aggregate/Data Feed CSV files |
| 🔗 **MAPPING** | Map columns between files (AI-suggested, user-adjustable) + date validation config |
| 📊 **RESULTS** | View comparison output with stats, data table, and detail panel for comments |
| 💬 **ASK AGENT** | Chat with the AI about your data (disabled during processing, answers only data questions) |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/)
- **Ollama** *(optional, for AI features)* — [Download here](https://ollama.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakshi2112-learner/Data_Validation_App.git
   cd CSVCompareApp
   ```

2. **Create a virtual environment** *(recommended)*
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama** *(optional — skip if you only need rule-based matching)*
   ```bash
   # After installing Ollama from https://ollama.com
   ollama pull phi3:mini
   ```
   > This downloads the Phi-3 Mini model (~2.3 GB). Required only for AI-powered column mapping and chat Q&A.

5. **Run the app**
   ```bash
   python app.py
   ```

---

## 📖 How to Use

1. **Load Files** — In the FILES tab, browse and select your Flowchart (File 1) and Aggregate/Data Feed (File 2) CSV files
2. **Map Columns** — The app auto-suggests column mappings. Review, adjust, and check which columns are key columns for matching
3. **Configure Dates** — Select start date, end date, and flight/range columns for date validation. The app auto-detects date formats
4. **Run Comparison** — Click "RUN COMPARISON" to find missing records and date mismatches
5. **View Results** — See summary stats (total issues, missing records, date mismatches), browse the results table, click a row to see full comment details
6. **Save Output** — Click "EXPORT TO CUSTOM LOCATION" to save the CSV (no auto-download)
7. **Ask Questions** — Use the ASK AGENT tab to ask questions about your data in natural language

---

## 🏗️ Project Structure

```
CSVCompareApp/
├── app.py                  # Main application — Tkinter GUI with 4 tabs
├── comparison_engine.py    # Core comparison logic — missing records,
│                           #   flexible date validation, output CSV
├── agent.py                # Local LLM agent — column mapping suggestions,
│                           #   date role detection, and chat Q&A via Ollama
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from Git
└── README.md               # This file
```

---

## 📦 Dependencies

### Python Libraries (installed via pip)

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | Latest | Loads CSV/Excel files, processes DataFrames, auto-parses dates |
| `ollama` | Latest | Python client for the local Ollama LLM server |
| `openpyxl` | Latest | Reads `.xlsx` Excel files |

### Python Standard Library (built-in, no install needed)

| Library | Purpose |
|---------|---------|
| `tkinter` | Desktop GUI framework — windows, tabs, buttons, file dialogs |
| `threading` | Runs AI tasks in the background so the GUI stays responsive |
| `os` | File path operations — checking existence, getting basenames |
| `re` | Regex — detects date range formats like "Jan - Jun" |
| `datetime` | Date parsing utilities |
| `json` | Parses the LLM's JSON responses for column mapping |
| `typing` | Type hints for cleaner function signatures |

### External Tools (installed separately)

| Tool | Required? | Purpose |
|------|-----------|---------|
| [Ollama](https://ollama.com) | Optional | Local LLM inference server |
| [Phi-3 Mini](https://ollama.com/library/phi3:mini) | Optional | Small language model for AI features |

> **Without Ollama**, the app falls back to rule-based column matching (exact names + synonym dictionary) and keyword-based Q&A. All comparison features work fully without it.

---

## 🤖 How the AI Agent Works

The agent operates with a **two-layer approach**:

### Column Mapping
1. **Rule-based** *(always runs)* — Matches columns by exact name or common synonyms (e.g., `channel` ↔ `channel name`)
2. **LLM-enhanced** *(if Ollama available)* — Sends column lists + sample data to Phi-3 to intelligently match semantically similar names

### Date Validation
- **Auto-detects date roles** — determines which file has min/max date columns vs. flight/range columns
- **Flexible date parsing** — uses `pd.to_datetime()` to parse any date format (MM/DD/YY, DD-MM-YYYY, etc.)
- **Month extraction** — extracts months from start/end dates and compares against flight range (e.g., `Jan - Jun`)

### Chat Q&A
- **With LLM** — Sends your question with data context to the local LLM; restricted to only answer data-related questions
- **Without LLM** — Falls back to comprehensive keyword-based answers (missing records, mismatches, column info, row counts)
- Send button is disabled during processing to prevent duplicate queries

### Privacy
- **100% local** — Ollama runs on `localhost:11434`, never contacts external servers
- **No data shared** — Your CSV files and comparison results never leave your machine
- **No API keys** — No cloud services, no usage costs, works fully offline

---

## ⚠️ Known Limitations

- **Case-insensitive matching** — Column names are normalized to lowercase for comparison
- **Performance on large files** — Row-by-row comparison may be slow on files with 100K+ rows
- **Single delimiter support** — Assumes comma-separated CSV; tab-delimited or semicolon files may not work

---

## 📄 License

This project is for personal/educational use.
