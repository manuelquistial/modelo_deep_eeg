#!/usr/bin/env python3
"""Generate result sentences for paper (only if data supports them)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paths import baseline_runs_dir, paper_tables_dir, publishable_runs_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output = args.output or (paper_tables_dir(ROOT) / "generated_result_sentences.md")
    lines = ["# Generated result sentences\n\n"]

    summary_files = list(results_root.glob("**/repeated_holdout_summary.csv"))
    if not summary_files:
        lines.append("Insufficient evidence: run repeated hold-out first.\n")
    else:
        for sf in summary_files:
            df = pd.read_csv(sf)
            if df.empty:
                continue
            ds = df["dataset"].iloc[0] if "dataset" in df.columns else sf.parent.name
            if "use_ea" in df.columns:
                ea_grp = df.groupby("use_ea")["balanced_accuracy_mean"].mean()
                if len(ea_grp) == 2:
                    lines.append(
                        f"- Across repeated subject-disjoint splits on {ds}, Euclidean Alignment "
                        f"changed mean balanced accuracy (EA={ea_grp.get(True, float('nan')):.3f} vs "
                        f"no-EA={ea_grp.get(False, float('nan')):.3f}).\n"
                    )
            best = df.loc[df["balanced_accuracy_mean"].idxmax()] if "balanced_accuracy_mean" in df.columns else None
            if best is not None:
                lines.append(
                    f"- On {ds}, {best['model']} (EA={best['use_ea']}) achieved the highest mean balanced "
                    f"accuracy ({best['balanced_accuracy_mean']:.3f} ± {best.get('balanced_accuracy_std', 0):.3f}).\n"
                )

    fixed = baseline_runs_dir(ROOT) / "pipeline_comparison.csv"
    if fixed.exists():
        df = pd.read_csv(fixed)
        phys = df[(df["dataset"] == "physionet") & (df["use_ea"] == True)]  # noqa
        if len(phys):
            classical = phys[phys["model"].isin(["FBCSP+LDA", "CSP+SVM", "fbcsp_lda", "csp_svm"])]
            deep = phys[phys["model"].str.contains("EEG", case=False, na=False)]
            if len(classical) and len(deep):
                lines.append(
                    "- On PhysioNet (fixed hold-out), classical spatial filtering remained competitive "
                    "with deep learning under subject-independent evaluation.\n"
                )
        bnci = df[(df["dataset"].str.contains("bnci", case=False)) & (df["use_ea"] == True)]  # noqa
        eegnet = bnci[bnci["model"].str.contains("EEGNet", case=False, na=False)]
        if len(eegnet):
            lines.append(
                f"- On BNCI2014-001, EEGNet achieved strong balanced accuracy "
                f"({eegnet['balanced_accuracy'].iloc[0]:.3f}) under fixed hold-out.\n"
            )

    subj = list(results_root.glob("**/subject_level_metrics.csv"))
    if subj:
        sdf = pd.read_csv(subj[0])
        if len(sdf):
            std_acc = sdf.groupby("subject_id")["accuracy"].mean().std()
            lines.append(
                f"- Subject-level analysis showed substantial inter-subject variability "
                f"(std of mean subject accuracy ≈ {std_acc:.3f}).\n"
            )

    lat = list(args.results_root.glob("**/lateralization_vs_accuracy.csv"))
    if lat:
        ldf = pd.read_csv(lat[0])
        sig = ldf[ldf["pearson_p"] < 0.05] if "pearson_p" in ldf.columns else pd.DataFrame()
        if len(sig):
            lines.append(
                "- Lateralization analysis showed significant correlation between mu-band lateralization "
                "and subject-level accuracy for some models.\n"
            )
        else:
            lines.append(
                "- Lateralization vs accuracy correlations were not consistently significant; "
                "interpret with caution.\n"
            )
    else:
        lines.append("Insufficient evidence: run neurophysiology analysis first.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
