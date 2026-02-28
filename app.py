"""
CSV Compare Agent — Desktop Application
Gamer-style dark theme Tkinter GUI with neon accents.
Fully offline using Ollama + Phi-3-mini.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import pandas as pd

from comparison_engine import (
    get_columns, detect_date_format, load_csv, run_comparison
)
from agent import LocalAgent


# ============================================================
# THEME CONSTANTS
# ============================================================
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BG_INPUT = "#1c2333"
FG_TEXT = "#c9d1d9"
FG_DIM = "#6e7681"
FG_BRIGHT = "#f0f6fc"
NEON_CYAN = "#00e5ff"
NEON_GREEN = "#39ff14"
NEON_PINK = "#ff006e"
NEON_ORANGE = "#ff9100"
NEON_PURPLE = "#b388ff"
ACCENT = NEON_CYAN
ACCENT_HOVER = "#33eaff"
SUCCESS = NEON_GREEN
ERROR = NEON_PINK
WARNING = NEON_ORANGE

FONT_FAMILY = "Consolas"
FONT_TITLE = (FONT_FAMILY, 18, "bold")
FONT_HEADING = (FONT_FAMILY, 13, "bold")
FONT_BODY = (FONT_FAMILY, 10)
FONT_SMALL = (FONT_FAMILY, 9)
FONT_MONO = (FONT_FAMILY, 10)
FONT_CHAT = (FONT_FAMILY, 10)


class CSVCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ CSV Compare Agent")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(800, 550)
        self.root.resizable(True, True)

        # Make root grid responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # State
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.cols1 = []
        self.cols2 = []
        self.mapping_vars = {}  # {col1: StringVar for col2 selection}
        self.date_vars = {}     # Date column selections
        self.agent = None
        self.df1 = None
        self.df2 = None
        self.output_df = None
        self.output_path = ""

        # Init agent in background
        self._init_agent_async()

        # Style
        self._setup_styles()

        # Build UI
        self._build_header()
        self._build_notebook()
        self._build_status_bar()

    # ============================================================
    # STYLES
    # ============================================================
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Dark.TLabel", background=BG_DARK, foreground=FG_TEXT, font=FONT_BODY)
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT, font=FONT_BODY)
        style.configure("Heading.TLabel", background=BG_DARK, foreground=ACCENT, font=FONT_HEADING)
        style.configure("Title.TLabel", background=BG_DARK, foreground=FG_BRIGHT, font=FONT_TITLE)
        style.configure("Dim.TLabel", background=BG_DARK, foreground=FG_DIM, font=FONT_SMALL)
        style.configure("Success.TLabel", background=BG_DARK, foreground=SUCCESS, font=FONT_BODY)
        style.configure("Error.TLabel", background=BG_DARK, foreground=ERROR, font=FONT_BODY)

        style.configure("Accent.TButton",
                        background=ACCENT, foreground=BG_DARK,
                        font=FONT_BODY, padding=(15, 8))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_HOVER)])

        style.configure("Dark.TButton",
                        background=BG_CARD, foreground=FG_TEXT,
                        font=FONT_BODY, padding=(12, 6))
        style.map("Dark.TButton",
                  background=[("active", BG_INPUT)])

        # Notebook
        style.configure("Dark.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BG_CARD, foreground=FG_DIM,
                        font=FONT_BODY, padding=(20, 8))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", ACCENT)])

        # Combobox
        style.configure("Dark.TCombobox",
                        fieldbackground="white", background=BG_CARD,
                        foreground="black", font=FONT_BODY,
                        selectbackground=ACCENT, selectforeground="black")

        # Fix combobox dropdown list colors (popdown listbox)
        self.root.option_add("*TCombobox*Listbox.background", "white")
        self.root.option_add("*TCombobox*Listbox.foreground", "black")
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "black")
        self.root.option_add("*TCombobox*Listbox.font", FONT_BODY)

    # ============================================================
    # HEADER
    # ============================================================
    def _build_header(self):
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill="x", padx=20, pady=(15, 5))

        ttk.Label(header, text="⚡ CSV COMPARE AGENT",
                  style="Title.TLabel").pack(side="left")

        self.agent_status_label = ttk.Label(
            header, text="● Agent: Loading...",
            style="Dim.TLabel"
        )
        self.agent_status_label.pack(side="right")

    # ============================================================
    # NOTEBOOK (Tabs)
    # ============================================================
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Tab 1: Files
        self.tab_files = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(self.tab_files, text="  📂 FILES  ")
        self._build_files_tab()

        # Tab 2: Column Mapping
        self.tab_mapping = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(self.tab_mapping, text="  🔗 MAPPING  ")
        self._build_mapping_tab()

        # Tab 3: Results
        self.tab_results = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(self.tab_results, text="  📊 RESULTS  ")
        self._build_results_tab()

        # Tab 4: Chat
        self.tab_chat = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(self.tab_chat, text="  💬 ASK AGENT  ")
        self._build_chat_tab()

    # ============================================================
    # TAB 1: FILE SELECTION
    # ============================================================
    def _build_files_tab(self):
        container = ttk.Frame(self.tab_files, style="Dark.TFrame")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # File 1
        ttk.Label(container, text="FILE 1 — Flowchart",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, 5))
        f1_frame = ttk.Frame(container, style="Dark.TFrame")
        f1_frame.pack(fill="x", pady=(0, 5))

        self.f1_entry = tk.Entry(f1_frame, textvariable=self.file1_path,
                                 bg=BG_INPUT, fg=FG_TEXT, font=FONT_MONO,
                                 insertbackground=ACCENT, bd=0,
                                 highlightthickness=1, highlightcolor=ACCENT,
                                 highlightbackground=BG_CARD)
        self.f1_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))

        ttk.Button(f1_frame, text="Browse",
                   style="Dark.TButton",
                   command=lambda: self._browse_file(self.file1_path)).pack(side="right")

        # Skip rows for File 1
        f1_skip_frame = ttk.Frame(container, style="Dark.TFrame")
        f1_skip_frame.pack(fill="x", pady=(0, 15))
        tk.Label(f1_skip_frame, text="Header row skip (Auto or number):",
                 bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL).pack(side="left")
        self.f1_skip_var = tk.StringVar(value="Auto")
        tk.Entry(f1_skip_frame, textvariable=self.f1_skip_var,
                 bg=BG_INPUT, fg=FG_TEXT, font=FONT_SMALL,
                 insertbackground=ACCENT, bd=0, width=8,
                 highlightthickness=1, highlightcolor=BG_CARD,
                 highlightbackground=BG_CARD).pack(side="left", padx=(10, 0), ipady=4)

        # File 2
        ttk.Label(container, text="FILE 2 — Aggregate / Data Feed",
                  style="Heading.TLabel").pack(anchor="w", pady=(10, 5))
        f2_frame = ttk.Frame(container, style="Dark.TFrame")
        f2_frame.pack(fill="x", pady=(0, 5))

        self.f2_entry = tk.Entry(f2_frame, textvariable=self.file2_path,
                                 bg=BG_INPUT, fg=FG_TEXT, font=FONT_MONO,
                                 insertbackground=ACCENT, bd=0,
                                 highlightthickness=1, highlightcolor=ACCENT,
                                 highlightbackground=BG_CARD)
        self.f2_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))

        ttk.Button(f2_frame, text="Browse",
                   style="Dark.TButton",
                   command=lambda: self._browse_file(self.file2_path)).pack(side="right")

        # Skip rows for File 2
        f2_skip_frame = ttk.Frame(container, style="Dark.TFrame")
        f2_skip_frame.pack(fill="x", pady=(0, 20))
        tk.Label(f2_skip_frame, text="Header row skip (Auto or number):",
                 bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL).pack(side="left")
        self.f2_skip_var = tk.StringVar(value="Auto")
        tk.Entry(f2_skip_frame, textvariable=self.f2_skip_var,
                 bg=BG_INPUT, fg=FG_TEXT, font=FONT_SMALL,
                 insertbackground=ACCENT, bd=0, width=8,
                 highlightthickness=1, highlightcolor=BG_CARD,
                 highlightbackground=BG_CARD).pack(side="left", padx=(10, 0), ipady=4)

        # Analyze button
        self.analyze_btn = ttk.Button(
            container, text="⚡  ANALYZE FILES",
            style="Accent.TButton",
            command=self._analyze_files
        )
        self.analyze_btn.pack(pady=(10, 0))

        # Column preview area
        self.col_preview_frame = ttk.Frame(container, style="Dark.TFrame")
        self.col_preview_frame.pack(fill="both", expand=True, pady=(15, 0))

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV & Excel files", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ]
        )
        if path:
            var.set(path)

    def _analyze_files(self):
        f1 = self.file1_path.get().strip()
        f2 = self.file2_path.get().strip()

        if not f1 or not f2:
            self._set_status("Please select both files.", ERROR)
            return

        if not os.path.exists(f1):
            self._set_status(f"File not found: {f1}", ERROR)
            return
        if not os.path.exists(f2):
            self._set_status(f"File not found: {f2}", ERROR)
            return

        # Check if both files are the same
        if os.path.abspath(f1) == os.path.abspath(f2):
            messagebox.showwarning("Same File", "Both files are the same! Please select two different files.")
            return

        self._set_status("Analyzing files...", ACCENT)

        # Parse skip rows
        def _parse_skip(var_val):
            v = var_val.strip().lower()
            if v == "auto" or v == "":
                return None  # auto-detect
            try:
                return int(v)
            except ValueError:
                return None

        skip1 = _parse_skip(self.f1_skip_var.get())
        skip2 = _parse_skip(self.f2_skip_var.get())

        try:
            self.cols1 = get_columns(f1, skip_rows=skip1)
            self.cols2 = get_columns(f2, skip_rows=skip2)
            self.df1 = load_csv(f1, skip_rows=skip1)
            self.df2 = load_csv(f2, skip_rows=skip2)

            # Show preview
            for w in self.col_preview_frame.winfo_children():
                w.destroy()

            preview = ttk.Frame(self.col_preview_frame, style="Dark.TFrame")
            preview.pack(fill="x")

            # File 1 columns
            f1_label = ttk.Label(preview, text=f"File 1: {len(self.cols1)} columns — {', '.join(self.cols1)}",
                                 style="Dark.TLabel", wraplength=500)
            f1_label.pack(anchor="w", pady=2)

            # File 2 columns
            f2_label = ttk.Label(preview, text=f"File 2: {len(self.cols2)} columns — {', '.join(self.cols2)}",
                                 style="Dark.TLabel", wraplength=500)
            f2_label.pack(anchor="w", pady=2)

            # Auto-suggest mapping
            self._populate_mapping_tab()

            self._set_status("Files analyzed. Go to MAPPING tab to configure.", SUCCESS)
            self.notebook.select(self.tab_mapping)

        except Exception as e:
            self._set_status(f"Error: {e}", ERROR)

    # ============================================================
    # TAB 2: COLUMN MAPPING
    # ============================================================
    def _build_mapping_tab(self):
        # Scrollable mapping tab using Canvas
        self.mapping_canvas = tk.Canvas(self.tab_mapping, bg=BG_DARK,
                                        highlightthickness=0)
        self.mapping_scrollbar = ttk.Scrollbar(self.tab_mapping, orient="vertical",
                                                command=self.mapping_canvas.yview)
        self.mapping_container = ttk.Frame(self.mapping_canvas, style="Dark.TFrame")

        self.mapping_container.bind(
            "<Configure>",
            lambda e: self.mapping_canvas.configure(
                scrollregion=self.mapping_canvas.bbox("all")
            )
        )

        self.mapping_canvas_window = self.mapping_canvas.create_window(
            (0, 0), window=self.mapping_container, anchor="nw"
        )

        # Make inner frame resize with canvas width
        self.mapping_canvas.bind("<Configure>", lambda e: self.mapping_canvas.itemconfig(
            self.mapping_canvas_window, width=e.width
        ))

        self.mapping_canvas.configure(yscrollcommand=self.mapping_scrollbar.set)

        self.mapping_scrollbar.pack(side="right", fill="y")
        self.mapping_canvas.pack(side="left", fill="both", expand=True,
                                  padx=20, pady=20)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            self.mapping_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.mapping_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(self.mapping_container,
                  text="Load files first in the FILES tab.",
                  style="Dim.TLabel").pack(pady=20)

    def _populate_mapping_tab(self):
        # Clear existing
        for w in self.mapping_container.winfo_children():
            w.destroy()

        ttk.Label(self.mapping_container,
                  text="COLUMN MAPPING",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(self.mapping_container,
                  text="Match File 1 columns → File 2 columns. Agent suggested matches below.",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 15))

        # Get agent suggestions
        suggested = {}
        if self.agent:
            suggested = self.agent.suggest_column_mapping(
                self.cols1, self.cols2,
                df1=self.df1, df2=self.df2,
            )

        # Key columns mapping
        key_frame = ttk.Frame(self.mapping_container, style="Dark.TFrame")
        key_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(key_frame, text="KEY COLUMNS (used for matching records)",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, 5))

        self.mapping_vars = {}
        self.key_checks = {}

        mapping_scroll = ttk.Frame(self.mapping_container, style="Dark.TFrame")
        mapping_scroll.pack(fill="x")

        options = ["-- Not Mapped --"] + self.cols2

        for i, col1 in enumerate(self.cols1):
            row_frame = ttk.Frame(mapping_scroll, style="Dark.TFrame")
            row_frame.pack(fill="x", pady=3)

            # Checkbox for key column
            key_var = tk.BooleanVar(value=(col1 in suggested))
            self.key_checks[col1] = key_var
            cb = tk.Checkbutton(row_frame, variable=key_var,
                                bg=BG_DARK, fg=ACCENT, selectcolor=BG_INPUT,
                                activebackground=BG_DARK, activeforeground=ACCENT)
            cb.pack(side="left", padx=(0, 5))

            # File 1 column label
            lbl = tk.Label(row_frame, text=col1, bg=BG_DARK, fg=NEON_PURPLE,
                           font=FONT_MONO, width=25, anchor="w")
            lbl.pack(side="left", padx=(0, 10))

            # Arrow
            tk.Label(row_frame, text="→", bg=BG_DARK, fg=FG_DIM,
                     font=FONT_BODY).pack(side="left", padx=5)

            # Dropdown for file 2 column
            map_var = tk.StringVar(value=suggested.get(col1, "-- Not Mapped --"))
            self.mapping_vars[col1] = map_var

            combo = ttk.Combobox(row_frame, textvariable=map_var,
                                 values=options, state="readonly",
                                 style="Dark.TCombobox", width=30)
            combo.pack(side="left", padx=(10, 0))

        # Separator line
        sep = tk.Frame(self.mapping_container, bg=ACCENT, height=1)
        sep.pack(fill="x", pady=(20, 10))

        # Date configuration
        ttk.Label(self.mapping_container,
                  text="DATE VALIDATION",
                  style="Heading.TLabel").pack(anchor="w", pady=(10, 5))
        ttk.Label(self.mapping_container,
                  text="Select date columns for range comparison.",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 10))

        date_frame = ttk.Frame(self.mapping_container, style="Dark.TFrame")
        date_frame.pack(fill="x")

        # Auto-detect date roles using classify_date_role
        f1_role = None
        f2_role = None
        if self.agent:
            f1_role = self.agent.classify_date_role(self.file1_path.get())
            f2_role = self.agent.classify_date_role(self.file2_path.get())

        # Determine which file has min/max dates and which has flight/range
        # The file with min_max role provides start/end date columns
        # The file with range role provides the flight column
        default_f1_start = "-- None --"
        default_f1_end = "-- None --"
        default_f2_range = "-- None --"

        if f1_role and f2_role:
            if f1_role["role"] == "min_max" and f2_role["role"] == "range":
                # File 1 has min/max, File 2 has flight — standard case
                default_f1_start = f1_role.get("start_col") or "-- None --"
                default_f1_end = f1_role.get("end_col") or "-- None --"
                default_f2_range = f2_role.get("range_col") or "-- None --"
            elif f1_role["role"] == "range" and f2_role["role"] == "min_max":
                # Swapped: File 1 has flight, File 2 has min/max
                # We still use File 1 start/end dropdowns for the min/max file
                default_f1_start = f2_role.get("start_col") or "-- None --"
                default_f1_end = f2_role.get("end_col") or "-- None --"
                default_f2_range = f1_role.get("range_col") or "-- None --"
            elif f1_role["role"] == "min_max":
                default_f1_start = f1_role.get("start_col") or "-- None --"
                default_f1_end = f1_role.get("end_col") or "-- None --"
            elif f2_role["role"] == "min_max":
                default_f1_start = f2_role.get("start_col") or "-- None --"
                default_f1_end = f2_role.get("end_col") or "-- None --"

        # File 1 start date
        r1 = ttk.Frame(date_frame, style="Dark.TFrame")
        r1.pack(fill="x", pady=3)
        tk.Label(r1, text="Start Date Column:", bg=BG_DARK, fg=FG_TEXT,
                 font=FONT_BODY, width=20, anchor="w").pack(side="left")
        self.date_vars["f1_start"] = tk.StringVar(
            value=default_f1_start
        )
        ttk.Combobox(r1, textvariable=self.date_vars["f1_start"],
                     values=["-- None --"] + self.cols1 + self.cols2, state="readonly",
                     style="Dark.TCombobox", width=25).pack(side="left", padx=(10, 0))

        # File 1 end date
        r2 = ttk.Frame(date_frame, style="Dark.TFrame")
        r2.pack(fill="x", pady=3)
        tk.Label(r2, text="End Date Column:", bg=BG_DARK, fg=FG_TEXT,
                 font=FONT_BODY, width=20, anchor="w").pack(side="left")
        self.date_vars["f1_end"] = tk.StringVar(
            value=default_f1_end
        )
        ttk.Combobox(r2, textvariable=self.date_vars["f1_end"],
                     values=["-- None --"] + self.cols1 + self.cols2, state="readonly",
                     style="Dark.TCombobox", width=25).pack(side="left", padx=(10, 0))

        # File 2 range column
        r3 = ttk.Frame(date_frame, style="Dark.TFrame")
        r3.pack(fill="x", pady=3)
        tk.Label(r3, text="Flight/Range Column:", bg=BG_DARK, fg=FG_TEXT,
                 font=FONT_BODY, width=20, anchor="w").pack(side="left")
        self.date_vars["f2_range"] = tk.StringVar(
            value=default_f2_range
        )
        ttk.Combobox(r3, textvariable=self.date_vars["f2_range"],
                     values=["-- None --"] + self.cols1 + self.cols2, state="readonly",
                     style="Dark.TCombobox", width=25).pack(side="left", padx=(10, 0))

        # Run comparison button
        ttk.Button(self.mapping_container,
                   text="🚀  RUN COMPARISON",
                   style="Accent.TButton",
                   command=self._run_comparison).pack(pady=(20, 20))

    # ============================================================
    # RUN COMPARISON
    # ============================================================
    def _run_comparison(self):
        # Build key mapping from checked + mapped columns
        key_mapping = {}
        for col1, check_var in self.key_checks.items():
            if check_var.get():
                mapped_to = self.mapping_vars[col1].get()
                if mapped_to and mapped_to != "-- Not Mapped --":
                    key_mapping[col1] = mapped_to

        if not key_mapping:
            self._set_status("Select at least one key column pair for matching.", ERROR)
            return

        # Build date config — simple: user picked the 3 columns, just detect format and compare
        date_config = None
        f1_start = self.date_vars.get("f1_start", tk.StringVar()).get()
        f1_end = self.date_vars.get("f1_end", tk.StringVar()).get()
        f2_range = self.date_vars.get("f2_range", tk.StringVar()).get()

        if (f1_start != "-- None --" and f1_end != "-- None --"
                and f2_range != "-- None --"):
            from comparison_engine import detect_date_format

            # Get samples for start/end date — check both loaded dataframes
            start_samples = []
            if self.df1 is not None and f1_start in self.df1.columns:
                start_samples = self.df1[f1_start].dropna().head(5).astype(str).tolist()
            elif self.df2 is not None and f1_start in self.df2.columns:
                start_samples = self.df2[f1_start].dropna().head(5).astype(str).tolist()

            # Get samples for flight/range — check both loaded dataframes
            range_samples = []
            if self.df1 is not None and f2_range in self.df1.columns:
                range_samples = self.df1[f2_range].dropna().head(5).astype(str).tolist()
            elif self.df2 is not None and f2_range in self.df2.columns:
                range_samples = self.df2[f2_range].dropna().head(5).astype(str).tolist()

            date_config = {
                "file1_start_col": f1_start,
                "file1_end_col": f1_end,
                "file1_date_format": detect_date_format(start_samples),
                "file2_range_col": f2_range,
                "file2_date_format": detect_date_format(range_samples),
            }
            print(f"[Date Config] {date_config}")

        # Columns for output = ALL checked columns (whether mapped or not)
        output_cols = [col1 for col1, check_var in self.key_checks.items() if check_var.get()]

        # Output path
        script_dir = os.path.dirname(os.path.abspath(self.file1_path.get()))
        self.output_path = os.path.join(script_dir, "comparison_output.csv")

        self._set_status("Running comparison...", ACCENT)

        def do_compare():
            try:
                result = run_comparison(
                    file1=self.file1_path.get(),
                    file2=self.file2_path.get(),
                    key_mapping=key_mapping,
                    output_columns=output_cols,
                    date_config=date_config,
                    output_path=self.output_path,
                )

                # Load output for display
                if os.path.exists(self.output_path):
                    self.output_df = pd.read_csv(self.output_path).fillna("")
                    # Remove auto-downloaded file — user will choose to save later
                    os.remove(self.output_path)

                self.root.after(0, lambda: self._show_results(result))
                # After results are shown, prompt user to save
                self.root.after(500, self._prompt_save_results)

            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}", ERROR))

        threading.Thread(target=do_compare, daemon=True).start()

    # ============================================================
    # TAB 3: RESULTS
    # ============================================================
    def _build_results_tab(self):
        self.results_container = ttk.Frame(self.tab_results, style="Dark.TFrame")
        self.results_container.pack(fill="both", expand=True, padx=20, pady=20)

        ttk.Label(self.results_container,
                  text="Run a comparison first.",
                  style="Dim.TLabel").pack(pady=20)

    def _show_results(self, result_msg):
        self._set_status(result_msg, SUCCESS)
        self.notebook.select(self.tab_results)

        # Clear
        for w in self.results_container.winfo_children():
            w.destroy()

        ttk.Label(self.results_container,
                  text="COMPARISON RESULTS",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, 5))

        status_text = "✅ " + result_msg if "complete" in result_msg.lower() else "❌ " + result_msg
        ttk.Label(self.results_container, text=status_text,
                  style="Success.TLabel" if "complete" in result_msg.lower() else "Error.TLabel",
                  wraplength=700).pack(anchor="w", pady=(0, 15))

        if self.output_df is not None and not self.output_df.empty:
            # Summary stats
            stats_frame = ttk.Frame(self.results_container, style="Dark.TFrame")
            stats_frame.pack(fill="x", pady=(0, 10))

            total = len(self.output_df)
            missing_count = len(self.output_df[self.output_df["comment"].str.contains("Missing from", case=False, na=False)])
            mismatch_count = len(self.output_df[self.output_df["comment"].str.contains("mismatch", na=False)])

            for label, value, color in [
                ("Total Issues", total, ACCENT),
                ("Missing Records", missing_count, WARNING),
                ("Date Mismatches", mismatch_count, NEON_PINK),
            ]:
                stat_card = tk.Frame(stats_frame, bg=BG_CARD, padx=15, pady=8)
                stat_card.pack(side="left", padx=(0, 10))
                tk.Label(stat_card, text=str(value), bg=BG_CARD, fg=color,
                         font=(FONT_FAMILY, 20, "bold")).pack()
                tk.Label(stat_card, text=label, bg=BG_CARD, fg=FG_DIM,
                         font=FONT_SMALL).pack()

            # Table
            table_frame = ttk.Frame(self.results_container, style="Dark.TFrame")
            table_frame.pack(fill="both", expand=True, pady=(10, 0))

            # Treeview for data
            columns = list(self.output_df.columns)
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

            # Style the treeview
            style = ttk.Style()
            style.configure("Treeview",
                            background=BG_INPUT, foreground=FG_TEXT,
                            fieldbackground=BG_INPUT, font=FONT_SMALL,
                            rowheight=25)
            style.configure("Treeview.Heading",
                            background=BG_CARD, foreground=ACCENT,
                            font=FONT_BODY)
            style.map("Treeview", background=[("selected", BG_CARD)])

            for col in columns:
                tree.heading(col, text=col.upper())
                if col == "comment":
                    tree.column(col, width=500, minwidth=300, anchor="w", stretch=True)
                else:
                    tree.column(col, width=150, minwidth=80, anchor="w", stretch=False)

            for _, row in self.output_df.iterrows():
                values = ["" if str(v).strip().lower() == "nan" else v for v in row]
                tree.insert("", "end", values=values)

            # Scrollbars
            yscrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            xscrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)

            yscrollbar.pack(side="right", fill="y")
            xscrollbar.pack(side="bottom", fill="x")
            tree.pack(side="left", fill="both", expand=True)

            # Detail panel — shows full comment for the selected row
            detail_frame = ttk.Frame(self.results_container, style="Dark.TFrame")
            detail_frame.pack(fill="x", pady=(8, 0))

            ttk.Label(detail_frame, text="SELECTED ROW COMMENT:",
                      style="Heading.TLabel").pack(anchor="w")

            comment_detail = tk.Text(
                detail_frame,
                bg=BG_INPUT, fg=NEON_GREEN, font=FONT_MONO,
                bd=0, wrap="word", height=4,
                highlightthickness=1, highlightcolor=BG_CARD,
                highlightbackground=BG_CARD,
                state="disabled",
            )
            comment_detail.pack(fill="x", pady=(4, 0))

            def on_row_select(event):
                selected = tree.selection()
                if selected:
                    values = tree.item(selected[0], "values")
                    comment_text = values[-1] if values else ""
                    comment_detail.configure(state="normal")
                    comment_detail.delete("1.0", "end")
                    comment_detail.insert("1.0", comment_text)
                    comment_detail.configure(state="disabled")

            tree.bind("<<TreeviewSelect>>", on_row_select)

            # Export button
            ttk.Button(self.results_container,
                       text="💾  EXPORT TO CUSTOM LOCATION",
                       style="Dark.TButton",
                       command=self._export_results).pack(pady=(10, 0))

        else:
            ttk.Label(self.results_container,
                      text="No issues found! Files match perfectly.",
                      style="Success.TLabel").pack(pady=10)

    def _prompt_save_results(self):
        """Ask user if they want to save results after comparison."""
        if self.output_df is not None and not self.output_df.empty:
            save = messagebox.askyesno(
                "Save Results",
                "Comparison complete! Do you want to save the output file?"
            )
            if save:
                self._export_results()

    def _export_results(self):
        if self.output_df is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                ],
                initialfile="comparison_output.csv"
            )
            if path:
                if path.lower().endswith('.xlsx'):
                    self.output_df.to_excel(path, index=False)
                else:
                    self.output_df.to_csv(path, index=False)
                self._set_status(f"Exported to: {path}", SUCCESS)

    # ============================================================
    # TAB 4: CHAT / Q&A
    # ============================================================
    def _build_chat_tab(self):
        container = ttk.Frame(self.tab_chat, style="Dark.TFrame")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        ttk.Label(container, text="ASK THE AGENT",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(container,
                  text="Ask questions about your loaded files and comparison results.",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 10))

        # Chat history
        self.chat_display = scrolledtext.ScrolledText(
            container,
            bg=BG_INPUT, fg=FG_TEXT, font=FONT_CHAT,
            insertbackground=ACCENT, bd=0,
            highlightthickness=1, highlightcolor=BG_CARD,
            highlightbackground=BG_CARD,
            wrap="word", state="disabled", height=18
        )
        self.chat_display.pack(fill="both", expand=True, pady=(0, 10))

        # Configure tags for colored messages
        self.chat_display.tag_configure("user", foreground=NEON_CYAN)
        self.chat_display.tag_configure("agent", foreground=NEON_GREEN)
        self.chat_display.tag_configure("system", foreground=FG_DIM)

        # Input area
        input_frame = ttk.Frame(container, style="Dark.TFrame")
        input_frame.pack(fill="x")

        self.chat_input = tk.Entry(
            input_frame,
            bg=BG_INPUT, fg=FG_TEXT, font=FONT_CHAT,
            insertbackground=ACCENT, bd=0,
            highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground=BG_CARD
        )
        self.chat_input.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self.chat_input.bind("<Return>", lambda e: self._send_chat())

        self.send_btn = ttk.Button(input_frame, text="Send",
                   style="Accent.TButton",
                   command=self._send_chat)
        self.send_btn.pack(side="right")

        # Welcome message
        self._append_chat("AGENT", "Hello! Load your files and run a comparison, then ask me anything about the data.", "system")

    def _send_chat(self):
        question = self.chat_input.get().strip()
        if not question:
            return

        self.chat_input.delete(0, "end")
        self._append_chat("YOU", question, "user")

        # Disable input while processing
        self.send_btn.configure(state="disabled")
        self.chat_input.configure(state="disabled")

        # Build dataframes context
        dfs = {}
        if self.df1 is not None:
            dfs["File 1"] = self.df1
        if self.df2 is not None:
            dfs["File 2"] = self.df2
        if self.output_df is not None:
            dfs["Output"] = self.output_df

        if not dfs:
            self._append_chat("AGENT", "Please load files first before asking questions.", "system")
            self.send_btn.configure(state="normal")
            self.chat_input.configure(state="normal")
            return

        def do_query():
            try:
                if self.agent:
                    answer = self.agent.query_data(question, dfs)
                else:
                    answer = "Agent is not available. Please wait for it to initialize."
            except Exception as e:
                answer = f"Error: {e}"

            def on_response():
                self._append_chat("AGENT", answer, "agent")
                self.send_btn.configure(state="normal")
                self.chat_input.configure(state="normal")
                self.chat_input.focus_set()

            self.root.after(0, on_response)

        threading.Thread(target=do_query, daemon=True).start()
        self._append_chat("AGENT", "Thinking...", "system")

    def _append_chat(self, sender, message, tag):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"\n{sender}: ", tag)
        self.chat_display.insert("end", f"{message}\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    # ============================================================
    # STATUS BAR
    # ============================================================
    def _build_status_bar(self):
        self.status_bar = tk.Frame(self.root, bg=BG_CARD, height=30)
        self.status_bar.pack(fill="x", side="bottom")

        self.status_label = tk.Label(
            self.status_bar, text="Ready",
            bg=BG_CARD, fg=FG_DIM, font=FONT_SMALL,
            anchor="w", padx=20
        )
        self.status_label.pack(fill="x", ipady=4)

    def _set_status(self, text, color=FG_DIM):
        self.status_label.configure(text=text, fg=color)

    # ============================================================
    # AGENT INIT
    # ============================================================
    def _init_agent_async(self):
        def init():
            try:
                self.agent = LocalAgent()
                status_text = "● Agent: Online" if self.agent.available else "● Agent: Offline (rule-based)"
                status_color = SUCCESS if self.agent.available else WARNING
            except Exception:
                self.agent = LocalAgent.__new__(LocalAgent)
                self.agent.available = False
                self.agent.model_name = "phi3:mini"
                status_text = "● Agent: Offline (rule-based)"
                status_color = WARNING

            self.root.after(0, lambda: self.agent_status_label.configure(
                text=status_text, foreground=status_color
            ))

        threading.Thread(target=init, daemon=True).start()


# ============================================================
# MAIN
# ============================================================
def main():
    root = tk.Tk()
    app = CSVCompareApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
