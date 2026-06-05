"""Generate LaTeX tables from publishable results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "---"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    bold_best_col: str | None = None,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(df.columns) + "}",
        "\\hline",
        " & ".join(df.columns) + " \\\\",
        "\\hline",
    ]
    best_idx = None
    if bold_best_col and bold_best_col in df.columns:
        try:
            best_idx = df[bold_best_col].astype(float).idxmax()
        except Exception:
            best_idx = None
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = str(row[c])
            if idx == best_idx and c == bold_best_col:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def write_table_bundle(results_root: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    from physionet_mi.paths import baseline_runs_dir, find_project_root

    fixed = baseline_runs_dir(find_project_root())
    fixed = fixed / "pipeline_comparison.csv"
    if fixed.exists():
        df = pd.read_csv(fixed)
        tex = dataframe_to_latex(
            df.round(4),
            "Fixed hold-out results (subject-disjoint 80/20).",
            "tab:fixed_holdout",
            bold_best_col="accuracy",
        )
        p = output_dir / "table_fixed_holdout_results.tex"
        p.write_text(tex, encoding="utf-8")
        df.to_csv(output_dir / "table_fixed_holdout_results.csv", index=False)
        written.append(p)

    for name, glob_pat, cap, lbl in [
        ("repeated_holdout", "**/repeated_holdout_summary.csv", "Repeated hold-out summary.", "tab:repeated_holdout"),
        ("groupkfold", "**/groupkfold_summary.csv", "GroupKFold summary.", "tab:groupkfold"),
        ("ea_gain", "**/ea_gain_summary.csv", "EA gain summary.", "tab:ea_gain"),
    ]:
        matches = list(results_root.glob(glob_pat))
        if matches:
            df = pd.read_csv(matches[0])
            tex = dataframe_to_latex(df.round(4), cap, lbl)
            p = output_dir / f"table_{name}_results.tex"
            p.write_text(tex, encoding="utf-8")
            df.to_csv(output_dir / f"table_{name}_results.csv", index=False)
            written.append(p)

    note = output_dir / "table_notes.txt"
    note.write_text(
        "All tables: subject-independent evaluation; no subject overlap between train and test.\n"
        "Metrics computed on held-out test subjects only.\n",
        encoding="utf-8",
    )
    return written
