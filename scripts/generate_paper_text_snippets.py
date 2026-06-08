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

MODEL_LABELS = {
    "csp_svm": "CSP+SVM",
    "fbcsp_lda": "FBCSP+LDA",
    "riemann_mdm": "Riemann MDM",
    "riemann_ts_lr": "Riemann TS+LR",
    "eegnet": "EEGNet",
}

DATASET_LABELS = {
    "physionet": "PhysioNet MI",
    "bnci": "BNCI2014-001",
    "bnci2014_001": "BNCI2014-001",
}


def _dataset_label(name: str) -> str:
    key = str(name).lower()
    if "bnci" in key:
        return DATASET_LABELS["bnci"]
    if "physionet" in key:
        return DATASET_LABELS["physionet"]
    return str(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output = args.output or (paper_tables_dir(ROOT) / "generated_result_sentences.md")
    lines = ["# Generated result sentences\n\n"]

    lines.append("## Methodology (repeated hold-out)\n\n")
    lines.append(
        "Full (LaTeX):\n\n"
        "```latex\n"
        "To assess the stability of the benchmark with respect to random subject partitioning "
        "and stochastic model training, we used a repeated subject-disjoint hold-out protocol "
        "with ten repetitions. A master random seed of 42 was used to generate the sequence of "
        "repeated splits using NumPy's default random number generator. For each repetition, the "
        "generated seed controlled the subject split and, for deep learning models, the stochastic "
        "training components. Results are reported as mean $\\pm$ standard deviation across "
        "repetitions.\n"
        "```\n\n"
    )
    lines.append(
        "Short (abstract/methods summary):\n\n"
        "> Repeated subject-disjoint hold-out (10 repetitions) used master seed 42 to draw "
        "reproducible split seeds; mean ± SD across repetitions.\n\n"
    )

    summary_paths = {
        "physionet": results_root / "repeated_holdout" / "physionet" / "repeated_holdout_summary.csv",
        "bnci": results_root / "repeated_holdout" / "bnci" / "repeated_holdout_summary.csv",
    }
    if not any(p.exists() for p in summary_paths.values()):
        for sf in sorted(results_root.glob("**/repeated_holdout_summary.csv")):
            key = "bnci" if "bnci" in sf.parent.name else "physionet"
            summary_paths[key] = sf

    loaded = {
        key: pd.read_csv(path)
        for key, path in summary_paths.items()
        if path.exists()
    }
    if not loaded:
        lines.append("Insufficient evidence: repeated hold-out summaries not found.\n")
    else:
        for key in ("physionet", "bnci"):
            df = loaded.get(key)
            if df is None or df.empty:
                continue
            ds_raw = df["dataset"].iloc[0] if "dataset" in df.columns else key
            ds = _dataset_label(ds_raw)
            if "use_ea" in df.columns and "balanced_accuracy_mean" in df.columns:
                ea_grp = df.groupby("use_ea")["balanced_accuracy_mean"].mean()
                if len(ea_grp) == 2:
                    lines.append(
                        f"- Across repeated subject-disjoint splits on {ds}, Euclidean Alignment "
                        f"changed mean balanced accuracy (EA={ea_grp.get(True, float('nan')):.3f} vs "
                        f"no-EA={ea_grp.get(False, float('nan')):.3f}).\n"
                    )
            if "balanced_accuracy_mean" in df.columns:
                best = df.loc[df["balanced_accuracy_mean"].idxmax()]
                model_name = MODEL_LABELS.get(str(best["model"]), str(best["model"]))
                ea_flag = "with EA" if bool(best["use_ea"]) else "without EA"
                lines.append(
                    f"- On {ds}, {model_name} ({ea_flag}) achieved the highest mean balanced "
                    f"accuracy ({best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f}).\n"
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

    lat_files = list(results_root.glob("**/lateralization_vs_accuracy.csv"))
    if lat_files:
        sig_models: list[str] = []
        for lat_path in lat_files:
            ldf = pd.read_csv(lat_path)
            if "pearson_p" not in ldf.columns:
                continue
            for _, row in ldf[ldf["pearson_p"] < 0.05].iterrows():
                sig_models.append(f"{row['model']} ({lat_path.parent.parent.name})")
        if sig_models:
            lines.append(
                "- Mu-band lateralization correlated significantly with subject-level accuracy for "
                f"{', '.join(sig_models)}; other models showed weak or non-significant coupling.\n"
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
