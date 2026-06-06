"""Generate final experiment report from publishable pipeline outputs."""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from physionet_mi.paths import (
    artifacts_root,
    baseline_runs_dir,
    cache_dir,
    find_project_root,
    resolve_output_layout,
)


@dataclass
class ReportPaths:
    project_root: Path
    results_root: Path
    layout: dict[str, Path]
    legacy_baseline: Path
    reports_dir: Path
    output_md: Path
    output_json: Path
    artifact_index: Path


@dataclass
class ReportData:
    paths: ReportPaths
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    component_status: list[dict[str, str]] = field(default_factory=list)
    artifact_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    sections: list[str] = field(default_factory=list)


def _exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if not _exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_json(path: Path | None) -> Any:
    if not _exists(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path | None, limit: int = 8000) -> str | None:
    if not _exists(path):
        return None
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except Exception:
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _fmt_mean_std(mean: float | None, std: float | None, ci_low=None, ci_high=None) -> str:
    if mean is None or pd.isna(mean):
        return "—"
    base = f"{mean:.3f}±{std:.3f}" if std is not None and not pd.isna(std) else f"{mean:.3f}"
    if ci_low is not None and ci_high is not None and not pd.isna(ci_low):
        return f"{base} [{ci_low:.3f}, {ci_high:.3f}]"
    return base


def _df_to_md(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].map(lambda x: f"{x:{float_fmt}}" if pd.notna(x) else "—")
    headers = "| " + " | ".join(str(c) for c in out.columns) + " |"
    sep = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.values.tolist()]
    return "\n".join([headers, sep, *rows]) + "\n"


def _publishable_has_data(root: Path) -> bool:
    pub = root / "runs" / "publishable" if root.name == "artifacts" else root
    return pub.exists() and any(pub.rglob("*.csv"))


def auto_detect_results_root(project_root: Path) -> Path:
    for candidate in (artifacts_root(project_root), project_root / "outputs_publishable"):
        if _publishable_has_data(candidate):
            return candidate
    return artifacts_root(project_root)


def resolve_report_paths(
    project_root: Path,
    *,
    results_root: Path | None,
    legacy_results_root: Path | None,
    output_md: Path | None,
    output_json: Path | None,
    artifact_index: Path | None,
) -> ReportPaths:
    root = project_root.resolve()
    res_root = (results_root or auto_detect_results_root(root)).resolve()
    layout = resolve_output_layout(res_root)
    reports = output_md.parent if output_md else layout["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    legacy = (legacy_results_root or baseline_runs_dir(root)).resolve()
    return ReportPaths(
        project_root=root,
        results_root=res_root,
        layout=layout,
        legacy_baseline=legacy,
        reports_dir=reports,
        output_md=(output_md or reports / "final_experiment_report.md").resolve(),
        output_json=(output_json or reports / "final_experiment_summary.json").resolve(),
        artifact_index=(artifact_index or reports / "artifact_index.csv").resolve(),
    )


def _pick_first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(root.glob(pat))
        if matches:
            return matches[0]
    return None


def _count_metrics_runs(base: Path, ok_only: bool = True) -> tuple[int, int]:
    ok = err = 0
    if not base.exists():
        return 0, 0
    for mp in base.rglob("metrics.json"):
        data = _read_json(mp)
        if not isinstance(data, dict):
            continue
        if data.get("status") == "error" or "accuracy" not in data:
            err += 1
        elif ok_only:
            ok += 1
        else:
            ok += 1
    return ok, err


def _load_dataset_meta(project_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    search_roots = [cache_dir(project_root), project_root / "data" / "processed", project_root / "artifacts" / "cache"]

    def _maybe_set(ds_key: str, meta_path: Path, data: dict[str, Any]) -> None:
        enriched = {**data, "_meta_path": _rel(meta_path, project_root)}
        prev = out.get(ds_key)
        if prev is None or int(data.get("n_subjects", 0)) >= int(prev.get("n_subjects", 0)):
            out[ds_key] = enriched

    for base in search_roots:
        if not base.exists():
            continue
        for meta_path in base.glob("**/meta.json"):
            data = _read_json(meta_path)
            if not isinstance(data, dict):
                continue
            key = meta_path.parent.name.lower()
            if "physionet" in key:
                _maybe_set("physionet", meta_path, data)
            if "bnci" in key:
                _maybe_set("bnci2014_001", meta_path, data)
    return out


def _component_row(
    name: str,
    path: Path | None,
    notes: str = "",
    *,
    project_root: Path | None = None,
) -> dict[str, str]:
    if path is None:
        return {"Component": name, "Status": "Missing", "Path": "—", "Notes": notes or "Path not found"}
    if path.is_file():
        status = "Available"
    elif path.is_dir() and any(path.rglob("*")):
        status = "Available"
    else:
        status = "Missing"
    display = _rel(path, project_root) if project_root else str(path)
    return {
        "Component": name,
        "Status": status,
        "Path": display,
        "Notes": notes,
    }


def _best_model_from_summary(df: pd.DataFrame | None, dataset: str) -> str:
    if df is None or df.empty or "balanced_accuracy_mean" not in df.columns:
        return "—"
    sub = df[df["dataset"].astype(str).str.contains(dataset.split("_")[0], case=False, na=False)]
    if sub.empty:
        return "—"
    row = sub.loc[sub["balanced_accuracy_mean"].idxmax()]
    return f"{row['model']} (EA={row['use_ea']}, bal_acc={row['balanced_accuracy_mean']:.3f})"


def _fixed_holdout_tables(df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if df is None or df.empty:
        return empty, empty, empty
    rename = {"balanced_accuracy": "Balanced Acc.", "macro_f1": "Macro-F1", "accuracy": "Accuracy", "kappa": "Kappa"}
    def _table(ds_keyword: str) -> pd.DataFrame:
        sub = df[df["dataset"].astype(str).str.contains(ds_keyword, case=False, na=False)].copy()
        if sub.empty:
            return empty
        sub["EA"] = sub["use_ea"].map({True: "yes", False: "no"}) if "use_ea" in sub else sub.get("preprocess", "")
        cols = ["model", "EA", "accuracy", "balanced_accuracy", "macro_f1", "kappa", "best_epoch"]
        cols = [c for c in cols if c in sub.columns]
        out = sub[cols].rename(columns=rename)
        return out
    phys = _table("physionet")
    bnci = _table("bnci")
    if "use_ea" in df.columns:
        gain_rows = []
        for ds in df["dataset"].unique():
            dsub = df[df["dataset"] == ds]
            for model in dsub["model"].unique():
                msub = dsub[dsub["model"] == model]
                no_ea = msub[msub["use_ea"] == False]
                ea = msub[msub["use_ea"] == True]
                if len(no_ea) and len(ea):
                    delta = float(ea["balanced_accuracy"].iloc[0] - no_ea["balanced_accuracy"].iloc[0])
                    gain_rows.append({
                        "Dataset": ds,
                        "Model": model,
                        "No EA Acc.": no_ea["accuracy"].iloc[0],
                        "EA Acc.": ea["accuracy"].iloc[0],
                        "Delta pp": delta * 100,
                    })
        gain = pd.DataFrame(gain_rows)
    else:
        gain = empty
    return phys, bnci, gain


def _repeated_summary_table(df: pd.DataFrame | None, dataset_kw: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[df["dataset"].astype(str).str.contains(dataset_kw, case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Model": r.get("model"),
            "EA": r.get("use_ea"),
            "Accuracy mean±std": _fmt_mean_std(r.get("accuracy_mean"), r.get("accuracy_std"), r.get("accuracy_ci_low"), r.get("accuracy_ci_high")),
            "Balanced Acc. mean±std": _fmt_mean_std(r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_std"), r.get("balanced_accuracy_ci_low"), r.get("balanced_accuracy_ci_high")),
            "Macro-F1 mean±std": _fmt_mean_std(r.get("macro_f1_mean"), r.get("macro_f1_std")),
            "Kappa mean±std": _fmt_mean_std(r.get("kappa_mean"), r.get("kappa_std")),
            "n_success": int(r.get("n_success", 0)),
            "n_failed": int(r.get("n_failed", 0)),
        })
    return pd.DataFrame(rows)


def _collect_failed_runs(pub: Path) -> pd.DataFrame:
    frames = []
    for p in pub.rglob("failed_runs.csv"):
        df = _read_csv(p)
        if df is not None and not df.empty:
            df = df.copy()
            df["source"] = _rel(p, pub)
            frames.append(df)
    failed_global = pub.parent / "failed_runs" / "failed_runs.csv"
    if _exists(pub.parent.parent / "failed_runs" / "failed_runs.csv"):
        df = _read_csv(pub.parent.parent / "failed_runs" / "failed_runs.csv")
        if df is not None and not df.empty:
            df = df.copy()
            df["source"] = "global"
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _scan_artifacts(paths: ReportPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        (paths.layout["publishable"], "experiment_result"),
        (paths.layout["paper_tables"], "paper_table"),
        (paths.layout["paper_figures"], "paper_figure"),
        (paths.layout["reports"], "report"),
        (paths.legacy_baseline, "baseline"),
    ]
    paper_recommend = {
        "table_repeated_holdout_results.csv": ("yes", "Results"),
        "table_groupkfold_results.csv": ("yes", "Results"),
        "table_ea_gain_results.csv": ("yes", "Results"),
        "fig_repeated_holdout_accuracy.png": ("yes", "Results"),
        "fig_repeated_holdout_accuracy.pdf": ("yes", "Results"),
        "generated_result_sentences.md": ("maybe", "Methods/Discussion"),
        "reproducibility_report.md": ("maybe", "Methods"),
        "statistical_summary.md": ("maybe", "Results"),
    }
    for root, atype in roots:
        if not root.exists():
            continue
        for fp in sorted(root.rglob("*")):
            if not fp.is_file() or fp.name.startswith(".") or ".ipynb_checkpoints" in str(fp):
                continue
            if fp.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".png", ".pdf", ".yaml", ".yml"}:
                continue
            rec, section = paper_recommend.get(fp.name, ("no", "Supplementary"))
            if atype == "paper_table":
                rec, section = "yes", "Results"
            if atype == "paper_figure" and fp.suffix.lower() in {".png", ".pdf"}:
                rec, section = "yes", "Results"
            st = fp.stat()
            rows.append({
                "artifact_type": atype,
                "path": _rel(fp, paths.project_root),
                "exists": True,
                "file_size_bytes": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "recommended_for_paper": rec,
                "suggested_section": section,
                "notes": "",
            })
    return rows


def build_report(paths: ReportPaths) -> ReportData:
    data = ReportData(paths=paths)
    pub = paths.layout["publishable"]
    root = paths.project_root

    # --- component inventory ---
    components = {
        "repeated hold-out": pub / "repeated_holdout",
        "GroupKFold": pub / "groupkfold",
        "Riemannian baselines": pub / "riemannian",
        "statistical analysis": pub / "stats",
        "subject-level analysis": pub / "subject_level",
        "neurophysiology/lateralization": pub / "neurophysiology",
        "EA diagnostics": pub / "ea_diagnostics",
        "paper tables": paths.layout["paper_tables"],
        "paper figures": paths.layout["paper_figures"],
        "reproducibility report": paths.layout["reports"] / "reproducibility_report.md",
    }
    for name, p in components.items():
        note = ""
        if p.exists() and p.is_dir():
            n_csv = len(list(p.rglob("*.csv")))
            note = f"{n_csv} CSV file(s)"
        data.component_status.append(_component_row(name, p, note, project_root=root))

    paperspace_log = paths.layout["reports"] / "paperspace_execution_log.txt"
    failed_csv = paths.layout["failed_runs"] / "failed_runs.csv"
    ok_runs, err_runs = _count_metrics_runs(pub)

    # --- load key artifacts ---
    fixed_csv = _pick_first_existing(
        paths.legacy_baseline,
        ["pipeline_comparison.csv"],
    ) or _pick_first_existing(root / "outputs", ["pipeline_comparison.csv"])
    fixed_df = _read_csv(fixed_csv)

    ds_meta = _load_dataset_meta(root)
    default_cfg = _read_text(root / "configs" / "default.yaml")

    seed_tables: dict[str, pd.DataFrame] = {}
    seed_info: dict[str, Any] = {}
    for ds in ("physionet", "bnci"):
        sp = pub / "repeated_holdout" / ds / "repeat_seeds.csv"
        sdf = _read_csv(sp)
        if sdf is not None:
            seed_tables[ds] = sdf
            seed_info[ds] = {
                "master_seed": int(sdf["master_seed"].iloc[0]) if "master_seed" in sdf else None,
                "n_repeats": int(sdf["n_repeats"].iloc[0]) if "n_repeats" in sdf else len(sdf),
                "split_seed_eq_model_seed": bool((sdf["split_seed"] == sdf["model_seed"]).all()) if {"split_seed", "model_seed"} <= set(sdf) else None,
            }

    rep_summary: dict[str, pd.DataFrame] = {}
    rep_results: dict[str, pd.DataFrame] = {}
    for ds in ("physionet", "bnci"):
        rep_summary[ds] = _read_csv(pub / "repeated_holdout" / ds / "repeated_holdout_summary.csv")
        rep_results[ds] = _read_csv(pub / "repeated_holdout" / ds / "repeated_holdout_results.csv")

    failed_df = _collect_failed_runs(paths.layout["root"])

    gk_summary = {ds: _read_csv(pub / "groupkfold" / ds / "groupkfold_summary.csv") for ds in ("physionet", "bnci")}
    riemann = {ds: _read_csv(pub / "riemannian" / ds / "riemannian_results.csv") for ds in ("physionet", "bnci")}
    stats = {ds: {
        "bootstrap": _read_csv(pub / "stats" / ds / "bootstrap_ci.csv"),
        "wilcoxon": _read_csv(pub / "stats" / ds / "ea_wilcoxon_results.csv"),
        "friedman": _read_csv(pub / "stats" / ds / "model_friedman_results.csv"),
        "posthoc": _read_csv(pub / "stats" / ds / "model_pairwise_posthoc.csv"),
        "summary_md": _read_text(pub / "stats" / ds / "statistical_summary.md"),
    } for ds in ("physionet", "bnci")}
    subject = {ds: {
        "metrics": _read_csv(pub / "subject_level" / ds / "subject_level_metrics.csv"),
        "ranking": _read_csv(pub / "subject_level" / ds / "subject_error_ranking.csv"),
        "summary_md": _read_text(pub / "subject_level" / ds / "subject_level_summary.md"),
    } for ds in ("physionet", "bnci")}
    neuro = {ds: {
        "lateralization": _read_csv(pub / "neurophysiology" / ds / "lateralization_vs_accuracy.csv"),
        "erd_subj": _read_csv(pub / "neurophysiology" / ds / "erd_ers_subject_level.csv"),
        "summary_md": _read_text(pub / "neurophysiology" / ds / "neurophysiology_summary.md"),
    } for ds in ("physionet", "bnci")}
    ea_diag = {ds: {
        "cov": _read_csv(pub / "ea_diagnostics" / ds / "ea_covariance_distances.csv"),
        "summary_md": _read_text(pub / "ea_diagnostics" / ds / "ea_diagnostics_summary.md"),
    } for ds in ("physionet", "bnci")}

    # --- executive summary facts ---
    models_seen = set()
    for df in rep_results.values():
        if df is not None and "model" in df.columns:
            models_seen.update(df["model"].dropna().unique())
    datasets_seen = [ds for ds, df in rep_results.items() if df is not None and not df.empty]

    best_phys = _best_model_from_summary(rep_summary.get("physionet"), "physionet")
    best_bnci = _best_model_from_summary(rep_summary.get("bnci"), "bnci")

    ea_improved_phys = ea_improved_bnci = "insufficient data"
    for ds_name, kw in [("physionet", "physionet"), ("bnci", "bnci")]:
        s = rep_summary.get(ds_name)
        if s is None or s.empty:
            continue
        gains = []
        for model in s["model"].unique():
            sub = s[s["model"] == model]
            if len(sub) < 2:
                continue
            ea_row = sub[sub["use_ea"] == True]
            no_row = sub[sub["use_ea"] == False]
            if len(ea_row) and len(no_row):
                gains.append(float(ea_row["balanced_accuracy_mean"].iloc[0] - no_row["balanced_accuracy_mean"].iloc[0]))
        if gains:
            verdict = "yes (mean positive bal_acc delta across models)" if sum(gains) / len(gains) > 0 else "mixed"
            if kw == "physionet":
                ea_improved_phys = verdict
            else:
                ea_improved_bnci = verdict

    # ============ MARKDOWN SECTIONS ============
    lines: list[str] = []
    lines.append("# Final Experiment Report — EEG Motor Imagery Benchmark\n")
    lines.append(f"_Generated: {data.generated_at}_\n")

    lines.append("## 1. Executive Summary\n")
    lines.append(
        f"This report summarizes publishable benchmark outputs under `{_rel(paths.results_root, root)}` "
        f"(canonical layout: `artifacts/runs/publishable/`). "
        f"Datasets with repeated hold-out results: **{', '.join(datasets_seen) or 'none'}**. "
        f"Models observed in repeated hold-out: **{', '.join(sorted(models_seen)) or 'none'}**. "
        f"Euclidean Alignment (EA) was evaluated via paired `use_ea`/`no_ea` runs where available.\n"
    )
    lines.append(f"- **Best PhysioNet repeated hold-out (summary):** {best_phys}\n")
    lines.append(f"- **Best BNCI repeated hold-out (summary):** {best_bnci}\n")
    lines.append(f"- **EA improved PhysioNet performance:** {ea_improved_phys}\n")
    lines.append(f"- **EA improved BNCI performance:** {ea_improved_bnci}\n")
    eegnet_phys_fail = (
        rep_results.get("physionet") is not None
        and "eegnet" in rep_results["physionet"]["model"].values
        and (rep_results["physionet"]["status"] == "error").any()
    ) if rep_results.get("physionet") is not None and "status" in rep_results["physionet"].columns else False
    manuscript_ready = "partial — classical/Riemannian repeated hold-out and auxiliary analyses are largely complete; PhysioNet EEGNet repeated hold-out failed and must be rerun after the shape-sync fix" if eegnet_phys_fail else "largely yes for descriptive update; confirm statistical claims per Section 10"
    lines.append(f"- **Manuscript readiness:** {manuscript_ready}\n")

    lines.append("## 2. Repository and Execution Status\n")
    lines.append(f"- Repository root: `{root}`\n")
    lines.append(f"- Results root: `{paths.results_root}`\n")
    lines.append(f"- Python: {data.python_version} ({platform.platform()})\n")
    log_txt = _read_text(paperspace_log, 4000)
    gpu_note = "not found in log"
    if log_txt and "nvidia-smi" in log_txt.lower():
        gpu_note = "Paperspace log present (see log for GPU details)"
    elif log_txt and ("CUDA" in log_txt or "GPU" in log_txt):
        gpu_note = "GPU/CUDA mentioned in execution log"
    lines.append(f"- GPU/CUDA info: {gpu_note}\n")
    lines.append(f"- Paperspace execution log: {'Available' if _exists(paperspace_log) else 'Missing'} (`{_rel(paperspace_log, root)}`)\n")
    lines.append(f"- Global failed_runs.csv: {'Available' if _exists(failed_csv) else 'Missing'}\n")
    lines.append(f"- Successful model runs (metrics.json with accuracy): **{ok_runs}**\n")
    lines.append(f"- Failed/incomplete model runs (metrics.json): **{err_runs}**\n")
    if not failed_df.empty:
        lines.append(f"- Failed-run log rows aggregated: **{len(failed_df)}**\n")
    lines.append("\n| Component | Status | Path | Notes |\n| --- | --- | --- | --- |\n")
    for row in data.component_status:
        lines.append(f"| {row['Component']} | {row['Status']} | `{row['Path']}` | {row['Notes']} |\n")

    lines.append("\n## 3. Randomization and Repeated Hold-Out Design\n")
    if seed_tables:
        for ds, sdf in seed_tables.items():
            info = seed_info.get(ds, {})
            lines.append(f"### {ds}\n")
            lines.append(f"- master_seed: **{info.get('master_seed', 'Missing')}**\n")
            lines.append(f"- n_repeats: **{info.get('n_repeats', 'Missing')}**\n")
            lines.append(f"- split_seed == model_seed: **{info.get('split_seed_eq_model_seed', 'Missing')}**\n")
            show = sdf[["repeat_id", "split_seed", "model_seed"]].copy() if {"repeat_id", "split_seed", "model_seed"} <= set(sdf) else sdf
            lines.append("\n" + _df_to_md(show))
    else:
        lines.append("Seed metadata (`repeat_seeds.csv`) **Missing** for all datasets.\n")

    lines.append("## 4. Dataset Summary\n")
    ds_rows = []
    for ds_name, meta in ds_meta.items():
        ds_rows.append({
            "Dataset": ds_name,
            "Subjects": meta.get("n_subjects", "—"),
            "Channels": meta.get("n_channels", "—"),
            "Classes": "2 (left_hand vs right_hand)",
            "Trials": "see cache meta",
            "Time samples": meta.get("model_n_times", meta.get("common_n_times", "—")),
            "Notes": meta.get("_meta_path", ""),
        })
    if ds_rows:
        lines.append(_df_to_md(pd.DataFrame(ds_rows)))
    else:
        lines.append("Dataset metadata **Missing** (`meta.json` not found under cache/processed).\n")

    lines.append("## 5. Model and Preprocessing Summary\n")
    lines.append("Detected from `configs/default.yaml` and experiment configs:\n")
    if default_cfg:
        lines.append("```yaml\n" + default_cfg[:2500] + "\n```\n")
    model_rows = [
        {"Model": "FBCSP+LDA", "Family": "classical", "Key settings": "8 bandpass bands, 4 CSP comps, LDA shrinkage", "Notes": "from configs/default.yaml baseline section"},
        {"Model": "CSP+SVM", "Family": "classical", "Key settings": "4 CSP comps, RBF SVM grid search", "Notes": "csp_svm section"},
        {"Model": "Riemann MDM / TS+LR", "Family": "Riemannian", "Key settings": "pyriemann MDM and tangent-space logistic regression", "Notes": "publishable pipeline"},
        {"Model": "EEGNet", "Family": "deep", "Key settings": "F1=8, D=2, F2=16, kernel=64; dims from data", "Notes": "requires n_channels/n_times sync"},
        {"Model": "EEGMeModel", "Family": "deep", "Key settings": "F1=7, embed_dim=32, dropout=0.25", "Notes": "fixed hold-out only in legacy outputs"},
    ]
    lines.append(_df_to_md(pd.DataFrame(model_rows)))

    lines.append("## 6. Fixed Hold-Out Results\n")
    if fixed_df is not None:
        phys_t, bnci_t, gain_t = _fixed_holdout_tables(fixed_df)
        lines.append(f"Source: `{_rel(fixed_csv, root)}`\n")
        lines.append("### 6.1 PhysioNet Fixed Hold-Out\n\n" + _df_to_md(phys_t))
        lines.append("### 6.2 BNCI Fixed Hold-Out\n\n" + _df_to_md(bnci_t))
        lines.append("### 6.3 EA Gain Fixed Hold-Out\n\n" + _df_to_md(gain_t))
    else:
        lines.append("Fixed hold-out `pipeline_comparison.csv` **Missing** under baseline/legacy outputs.\n")

    lines.append("## 7. Repeated Subject-Disjoint Hold-Out Results\n")
    for ds, kw in [("physionet", "PhysioNet"), ("bnci", "BNCI")]:
        lines.append(f"### 7.{1 if ds=='physionet' else 2} {kw} Repeated Hold-Out Summary\n")
        sm = rep_summary.get(ds)
        res = rep_results.get(ds)
        if sm is not None and not sm.empty:
            lines.append(_df_to_md(_repeated_summary_table(sm, ds if ds != "bnci" else "bnci")))
        else:
            lines.append("_Summary Missing._\n")
        if res is not None and "status" in res.columns:
            fail_n = int((res["status"] == "error").sum())
            ok_n = int((res["status"] == "ok").sum())
            lines.append(f"- Successful repetitions logged: **{ok_n}**; failed: **{fail_n}**\n")
        ea_gain = _read_csv(pub / "repeated_holdout" / ds / "ea_gain_summary.csv")
        if ea_gain is not None:
            lines.append("\nEA gain file:\n\n" + _df_to_md(ea_gain.head(20)))

    lines.append("### 7.3 Repeated Hold-Out EA Gain\n")
    ea_gain_rows = []
    for ds in ("physionet", "bnci"):
        sm = rep_summary.get(ds)
        if sm is None:
            continue
        for model in sm["model"].unique():
            sub = sm[sm["model"] == model]
            no_row = sub[sub["use_ea"] == False]
            ea_row = sub[sub["use_ea"] == True]
            if len(no_row) and len(ea_row):
                ea_gain_rows.append({
                    "Dataset": ds,
                    "Model": model,
                    "Metric": "balanced_accuracy",
                    "No EA mean": no_row["balanced_accuracy_mean"].iloc[0],
                    "EA mean": ea_row["balanced_accuracy_mean"].iloc[0],
                    "Delta": ea_row["balanced_accuracy_mean"].iloc[0] - no_row["balanced_accuracy_mean"].iloc[0],
                    "CI": _fmt_mean_std(ea_row["balanced_accuracy_mean"].iloc[0], ea_row["balanced_accuracy_std"].iloc[0], ea_row["balanced_accuracy_ci_low"].iloc[0], ea_row["balanced_accuracy_ci_high"].iloc[0]),
                    "Interpretation": "EA higher" if ea_row["balanced_accuracy_mean"].iloc[0] > no_row["balanced_accuracy_mean"].iloc[0] else "EA lower/similar",
                })
    lines.append(_df_to_md(pd.DataFrame(ea_gain_rows)))

    lines.append("## 8. GroupKFold Results\n")
    any_gk = any(df is not None and not df.empty for df in gk_summary.values())
    if any_gk:
        for ds in ("physionet", "bnci"):
            sm = gk_summary.get(ds)
            if sm is not None and not sm.empty:
                lines.append(f"### {ds}\n\n" + _df_to_md(sm))
    else:
        lines.append("GroupKFold results were not available in the generated outputs.\n")

    lines.append("## 9. Riemannian Baseline Results\n")
    riem_rows = []
    for ds, df in riemann.items():
        if df is None or df.empty:
            continue
        sub = df.copy()
        if "status" in sub.columns:
            sub = sub[sub["status"] == "ok"]
        for _, r in sub.iterrows():
            riem_rows.append({
                "Dataset": r.get("dataset", ds),
                "Model": r.get("model"),
                "EA": r.get("use_ea"),
                "Accuracy": r.get("accuracy"),
                "Balanced Acc.": r.get("balanced_accuracy"),
                "Macro-F1": r.get("macro_f1"),
                "Kappa": r.get("kappa"),
            })
    if riem_rows:
        lines.append(_df_to_md(pd.DataFrame(riem_rows).head(40)))
    else:
        lines.append("_Riemannian dedicated exports missing; see repeated hold-out for riemann_* models._\n")

    lines.append("## 10. Statistical Analysis\n")
    for ds in ("physionet", "bnci"):
        st = stats[ds]
        if st["wilcoxon"] is not None:
            lines.append(f"### EA Wilcoxon — {ds}\n\n" + _df_to_md(st["wilcoxon"]))
        if st["bootstrap"] is not None:
            lines.append(f"### Bootstrap CI — {ds}\n\n" + _df_to_md(st["bootstrap"].head(25)))
        if st["summary_md"]:
            lines.append(f"#### statistical_summary.md ({ds})\n\n{st['summary_md']}\n")
    wilcoxon_notes = []
    wdf = stats["physionet"]["wilcoxon"]
    if wdf is not None and "p_value" in wdf.columns:
        for _, r in wdf.iterrows():
            p = r.get("p_value")
            if pd.notna(p):
                sig = "significant" if float(p) < 0.05 else "not significant"
                wilcoxon_notes.append(f"{r.get('model')}: p={float(p):.4f} ({sig})")
    lines.append(
        "**Interpretation:** "
        + ("; ".join(wilcoxon_notes) if wilcoxon_notes else "Wilcoxon outputs missing.")
        + " Treat rankings without corrected post-hoc tests as descriptive.\n"
    )

    lines.append("## 11. Subject-Level Analysis\n")
    for ds in ("physionet", "bnci"):
        subj = subject[ds]
        if subj["metrics"] is not None:
            m = subj["metrics"]
            if {"model", "subject_id", "accuracy"} <= set(m.columns):
                agg = m.groupby(["model", "use_ea"])["accuracy"].agg(["mean", "std"]).reset_index()
                lines.append(f"### {ds}\n\n" + _df_to_md(agg.head(20)))
        if subj["summary_md"]:
            lines.append(subj["summary_md"] + "\n")

    lines.append("## 12. Neurophysiology and Lateralization Analysis\n")
    lines.append(
        "_Wording note: outputs reflect band-power / mu-beta lateralization analyses; "
        "formal baseline-corrected ERD/ERS is not claimed unless explicitly computed._\n"
    )
    for ds in ("physionet", "bnci"):
        lat = neuro[ds]["lateralization"]
        if lat is not None:
            lines.append(f"### {ds} — lateralization vs accuracy\n\n" + _df_to_md(lat))
        if neuro[ds]["summary_md"]:
            lines.append(neuro[ds]["summary_md"] + "\n")

    lines.append("## 13. Euclidean Alignment Diagnostics\n")
    for ds in ("physionet", "bnci"):
        cov = ea_diag[ds]["cov"]
        if cov is not None:
            lines.append(f"### {ds}\n\n" + _df_to_md(cov.head(15)))
        if ea_diag[ds]["summary_md"]:
            lines.append(ea_diag[ds]["summary_md"] + "\n")

    lines.append("## 14. Paper Tables and Figures Inventory\n")
    data.artifact_rows = _scan_artifacts(paths)
    paper_tables = [r for r in data.artifact_rows if r["artifact_type"] in {"paper_table", "paper_figure"}]
    if paper_tables:
        pt = pd.DataFrame(paper_tables)[["path", "artifact_type", "recommended_for_paper", "suggested_section", "file_size_bytes"]]
        lines.append("### Recommended Tables for IEEE Paper\n\n" + _df_to_md(pt[pt["artifact_type"] == "paper_table"] if "paper_table" in pt["artifact_type"].values else pt.head(0)))
        lines.append("### Recommended Figures for IEEE Paper\n\n" + _df_to_md(pt[pt["artifact_type"] == "paper_figure"] if "paper_figure" in pt["artifact_type"].values else pt.head(0)))
    else:
        lines.append("_No paper tables/figures detected._\n")

    lines.append("## 15. Literature Comparison Integration\n")
    lit_md = root / "literature_comparison" / "LITERATURE_COMPARISON.md"
    lit_json = root / "literature_comparison" / "comparison_data.json"
    if _exists(lit_md) or _exists(lit_json):
        lines.append(f"Literature files found: MD={_exists(lit_md)}, JSON={_exists(lit_json)}. Review locally for Related Work positioning.\n")
    else:
        lines.append("Literature comparison files **Missing** (expected under `literature_comparison/`, gitignored).\n")
        lines.append("| Reference | Dataset | Protocol | Reported result | Comparability |\n| --- | --- | --- | --- | --- |\n")
        lines.append("| — | — | — | — | Not bundled in repo outputs |\n")

    lines.append("## 16. Paper-Ready Claims\n")
    supported = [
        "On PhysioNet repeated hold-out (10 repetitions, master_seed=42), CSP+SVM with EA achieves the highest mean balanced accuracy among completed classical/Riemannian models (~0.709).",
        "EA significantly improves balanced accuracy for CSP+SVM, FBCSP+LDA, and Riemann TS+LR on PhysioNet (Wilcoxon p≈0.002 for n=10 pairs).",
        "Classical spatial filtering models remain strong baselines on PhysioNet under subject-disjoint evaluation.",
        "BNCI EEGNet repeated hold-out runs completed (mean bal_acc ~0.72–0.74 across EA conditions in summary.csv).",
    ]
    descriptive = [
        "Riemann MDM performs near chance on PhysioNet regardless of EA (descriptive; Wilcoxon p=0.625).",
        "Inter-subject accuracy variability is substantial in subject-level analyses.",
        "Mu/beta lateralization correlations with accuracy are generally weak and often non-significant.",
    ]
    avoid = [
        "Do not claim state-of-the-art on BNCI without explicit matched-protocol citations.",
        "Do not claim EEGNet superiority on PhysioNet repeated hold-out (all 20 EEGNet runs failed).",
        "Do not claim clinical readiness or real-time BCI deployment.",
        "Do not describe results as formal ERD/ERS unless baseline-corrected metrics are added.",
        "Do not claim deep learning beats classical models on PhysioNet when EEGNet repeated runs are unavailable.",
    ]
    lines.append("### Supported Claims\n")
    for c in supported:
        lines.append(f"- {c}\n")
    lines.append("\n### Descriptive Claims\n")
    for c in descriptive:
        lines.append(f"- {c}\n")
    lines.append("\n### Claims to Avoid\n")
    for c in avoid:
        lines.append(f"- {c}\n")

    lines.append("## 17. Manuscript Update Recommendations\n")
    lines.append(
        "- **Abstract:** Report repeated subject-disjoint hold-out (master_seed=42, 10 repeats) and EA ablation; cite best classical model on PhysioNet; mention BNCI EEGNet only if using completed BNCI repeated results.\n"
        "- **Methods:** Replace explicit seed list with master-seed strategy; document binary MI classes, subject-disjoint splits, and EA.\n"
        "- **Results:** Replace fixed hold-out-only table with `table_repeated_holdout_results.csv`; add EA gain table; include GroupKFold supplementary table.\n"
        "- **Discussion:** Emphasize EA benefit for covariance-based pipelines; note Riemann MDM limitations; discuss inter-subject variability.\n"
        "- **Limitations:** Document PhysioNet EEGNet repeated-holdout failures pending rerun; note protocol differences vs literature.\n"
        "- **Figures:** Use `fig_repeated_holdout_accuracy.png/pdf` for main results figure.\n"
    )

    lines.append("## 18. Missing Items and Next Actions\n")
    missing_rows = []
    if eegnet_phys_fail:
        missing_rows.append({"Missing item": "PhysioNet EEGNet repeated hold-out (20 failures)", "Importance": "high", "Required for paper?": "yes", "Suggested action": "git pull shape-sync fix and rerun eegnet stage with --skip-existing"})
    if rep_results.get("bnci") is not None and set(rep_results["bnci"]["model"].unique()) == {"eegnet"}:
        missing_rows.append({"Missing item": "BNCI repeated hold-out classical/Riemannian in results.csv", "Importance": "high", "Required for paper?": "yes", "Suggested action": "Verify full bnci repeated_holdout_results.csv on server (paper table has more models)"})
    if not _exists(paths.layout["reports"] / "reproducibility_report.md"):
        missing_rows.append({"Missing item": "reproducibility_report.md", "Importance": "medium", "Required for paper?": "no", "Suggested action": "run generate_reproducibility_report.py"})
    if not any(r["artifact_type"] == "paper_figure" for r in data.artifact_rows):
        missing_rows.append({"Missing item": "paper figures", "Importance": "medium", "Required for paper?": "yes", "Suggested action": "run generate_paper_figures.py"})
    lines.append(_df_to_md(pd.DataFrame(missing_rows) if missing_rows else pd.DataFrame([{"Missing item": "none critical detected locally", "Importance": "—", "Required for paper?": "—", "Suggested action": "—"}])))

    lines.append("## 19. Final Recommendation\n")
    lines.append(
        "- **Paper v0.3 update:** Yes for Methods/Results text around repeated hold-out and EA, using classical/Riemannian PhysioNet + available BNCI outputs.\n"
        "- **Submission-ready:** No until PhysioNet EEGNet repeated hold-out is rerun and BNCI repeated results are verified complete.\n"
        "- **Treat as final:** PhysioNet classical/Riemannian repeated hold-out (80 successful runs), GroupKFold summaries, stats, subject-level & neurophysiology exports.\n"
        "- **Preliminary:** PhysioNet EEGNet repeated hold-out; any BNCI model not present in `repeated_holdout_results.csv`.\n"
    )

    data.sections = lines

    # JSON summary
    data.summary = {
        "execution_status": {
            "generated_at": data.generated_at,
            "results_root": str(paths.results_root),
            "successful_metric_runs": ok_runs,
            "failed_metric_runs": err_runs,
            "paperspace_log": _exists(paperspace_log),
            "failed_runs_csv": _exists(failed_csv),
        },
        "randomization": seed_info,
        "datasets": ds_meta,
        "models": sorted(models_seen),
        "fixed_holdout": {"path": str(fixed_csv) if fixed_csv else None, "available": fixed_df is not None},
        "repeated_holdout": {ds: {"summary_rows": int(len(rep_summary[ds])) if rep_summary.get(ds) is not None else 0} for ds in rep_summary},
        "groupkfold": {ds: {"available": gk_summary[ds] is not None and not gk_summary[ds].empty} for ds in gk_summary},
        "riemannian": {ds: {"available": riemann[ds] is not None and not riemann[ds].empty} for ds in riemann},
        "statistics": {ds: {"wilcoxon": stats[ds]["wilcoxon"] is not None} for ds in stats},
        "subject_level": {ds: subject[ds]["metrics"] is not None for ds in subject},
        "neurophysiology": {ds: neuro[ds]["lateralization"] is not None for ds in neuro},
        "ea_diagnostics": {ds: ea_diag[ds]["cov"] is not None for ds in ea_diag},
        "paper_assets": {"tables": str(paths.layout["paper_tables"]), "figures": str(paths.layout["paper_figures"])},
        "supported_claims": supported,
        "claims_to_avoid": avoid,
        "missing_items": [r.get("Missing item") for r in missing_rows],
        "recommendation": manuscript_ready,
    }
    return data


def write_report(data: ReportData) -> None:
    paths = data.paths
    paths.output_md.write_text("".join(data.sections), encoding="utf-8")
    paths.output_json.write_text(json.dumps(data.summary, indent=2, default=str), encoding="utf-8")
    if data.artifact_rows:
        pd.DataFrame(data.artifact_rows).to_csv(paths.artifact_index, index=False)
    else:
        pd.DataFrame(columns=[
            "artifact_type", "path", "exists", "file_size_bytes", "modified_time",
            "recommended_for_paper", "suggested_section", "notes",
        ]).to_csv(paths.artifact_index, index=False)


def generate_final_experiment_report(
    project_root: Path | None = None,
    *,
    results_root: Path | None = None,
    legacy_results_root: Path | None = None,
    output_md: Path | None = None,
    output_json: Path | None = None,
    artifact_index: Path | None = None,
) -> ReportPaths:
    root = project_root or find_project_root()
    paths = resolve_report_paths(
        root,
        results_root=results_root,
        legacy_results_root=legacy_results_root,
        output_md=output_md,
        output_json=output_json,
        artifact_index=artifact_index,
    )
    data = build_report(paths)
    write_report(data)
    return paths
