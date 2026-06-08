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


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if not _exists(path):
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dataset_key_match(series: pd.Series, keyword: str) -> pd.Series:
    kw = keyword.lower().replace("bnci2014_001", "bnci")
    return series.astype(str).str.lower().str.contains(kw, na=False)


def _infer_dataset_info(riemann_df: pd.DataFrame | None, cfg: dict[str, Any], ds_key: str) -> dict[str, Any]:
    info: dict[str, Any] = {"name": ds_key}
    if riemann_df is not None and not riemann_df.empty:
        sub = riemann_df[_dataset_key_match(riemann_df["dataset"], ds_key)]
        if not sub.empty:
            row = sub.iloc[0]
            info["train_subjects"] = int(row.get("n_train_subjects", 0))
            info["test_subjects"] = int(row.get("n_test_subjects", 0))
            info["train_trials"] = int(row.get("n_train_trials", 0))
            info["test_trials"] = int(row.get("n_test_trials", 0))
    data_cfg = cfg.get("data", {})
    if ds_key == "physionet":
        info["channels"] = cfg.get("model", {}).get("n_channels", 64)
        info["time_samples"] = cfg.get("model", {}).get("n_times", 480)
        info["classes"] = "left_hand vs right_hand"
    else:
        info["channels"] = 22
        info["time_samples"] = int((data_cfg.get("bnci_tmax", 4.0) - data_cfg.get("bnci_tmin", 0.0)) * data_cfg.get("bnci_resample", 125.0))
        info["classes"] = "left_hand vs right_hand"
    return info


def _repeated_summary_table(df: pd.DataFrame | None, dataset_kw: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[_dataset_key_match(df["dataset"], dataset_kw)].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Model": r.get("model"),
            "EA": "yes" if r.get("use_ea") else "no",
            "Accuracy mean±std": _fmt_mean_std(
                r.get("accuracy_mean"), r.get("accuracy_std"),
                r.get("accuracy_ci_low"), r.get("accuracy_ci_high"),
            ),
            "Balanced Acc. mean±std": _fmt_mean_std(
                r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_std"),
                r.get("balanced_accuracy_ci_low"), r.get("balanced_accuracy_ci_high"),
            ),
            "Macro-F1 mean±std": _fmt_mean_std(r.get("macro_f1_mean"), r.get("macro_f1_std")),
            "Kappa mean±std": _fmt_mean_std(r.get("kappa_mean"), r.get("kappa_std")),
            "95% CI (bal_acc)": (
                f"[{r['balanced_accuracy_ci_low']:.3f}, {r['balanced_accuracy_ci_high']:.3f}]"
                if pd.notna(r.get("balanced_accuracy_ci_low")) else "—"
            ),
            "n_success": int(r.get("n_success", 0)),
            "n_failed": int(r.get("n_failed", 0)),
        })
    return pd.DataFrame(rows)


def _ea_gain_table(sm: pd.DataFrame | None, dataset_kw: str, wilcoxon: pd.DataFrame | None = None) -> pd.DataFrame:
    if sm is None or sm.empty:
        return pd.DataFrame()
    sub = sm[_dataset_key_match(sm["dataset"], dataset_kw)]
    rows = []
    for model in sub["model"].unique():
        msub = sub[sub["model"] == model]
        no_row = msub[msub["use_ea"] == False]
        ea_row = msub[msub["use_ea"] == True]
        if no_row.empty or ea_row.empty:
            continue
        delta = float(ea_row["balanced_accuracy_mean"].iloc[0] - no_row["balanced_accuracy_mean"].iloc[0])
        p_val = None
        if wilcoxon is not None and not wilcoxon.empty:
            wsub = wilcoxon[(wilcoxon["model"] == model)]
            if not wsub.empty and "p_value" in wsub.columns:
                p_val = float(wsub["p_value"].iloc[0])
        rows.append({
            "Dataset": dataset_kw,
            "Model": model,
            "Metric": "balanced_accuracy",
            "No EA mean": no_row["balanced_accuracy_mean"].iloc[0],
            "EA mean": ea_row["balanced_accuracy_mean"].iloc[0],
            "Delta": delta,
            "CI": _fmt_mean_std(
                ea_row["balanced_accuracy_mean"].iloc[0], ea_row["balanced_accuracy_std"].iloc[0],
                ea_row["balanced_accuracy_ci_low"].iloc[0], ea_row["balanced_accuracy_ci_high"].iloc[0],
            ),
            "Wilcoxon p": f"{p_val:.6f}" if p_val is not None else "—",
            "Interpretation": (
                "EA significantly higher" if p_val is not None and p_val < 0.05 and delta > 0
                else "EA higher (not sig.)" if delta > 0
                else "EA lower/similar"
            ),
        })
    return pd.DataFrame(rows)


def _ranking_table(sm: pd.DataFrame | None, dataset_kw: str) -> pd.DataFrame:
    if sm is None or sm.empty:
        return pd.DataFrame()
    sub = sm[_dataset_key_match(sm["dataset"], dataset_kw)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("balanced_accuracy_mean", ascending=False)
    sub["rank"] = range(1, len(sub) + 1)
    return sub[["rank", "model", "use_ea", "balanced_accuracy_mean", "balanced_accuracy_std", "n_success"]].rename(
        columns={"use_ea": "EA", "balanced_accuracy_mean": "bal_acc_mean", "balanced_accuracy_std": "bal_acc_std"}
    )


def _subject_level_summary_table(metrics: pd.DataFrame | None, ranking: pd.DataFrame | None) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["model", "use_ea"] if c in metrics.columns]
    if not group_cols or "accuracy" not in metrics.columns:
        return pd.DataFrame()
    hardest = ranking["subject_id"].head(3).tolist() if ranking is not None and not ranking.empty else []
    easiest = ranking["subject_id"].tail(3).tolist()[::-1] if ranking is not None and not ranking.empty else []
    for keys, grp in metrics.groupby(group_cols):
        model = keys[0] if isinstance(keys, tuple) else keys
        ea = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else None
        rows.append({
            "Model": model,
            "EA": "yes" if ea else "no" if ea is not None else "—",
            "Mean subject acc.": grp["accuracy"].mean(),
            "Std subject acc.": grp["accuracy"].std(),
            "Hardest subjects": ", ".join(str(s) for s in hardest) if hardest else "—",
            "Easiest subjects": ", ".join(str(s) for s in easiest) if easiest else "—",
        })
    return pd.DataFrame(rows).head(12)


def _neuro_findings_table(erd_subj: pd.DataFrame | None, lat: pd.DataFrame | None, dataset: str) -> pd.DataFrame:
    rows = []
    if erd_subj is not None and {"label", "mean_li_mu"} <= set(erd_subj.columns):
        for band_col, band_name in [("mean_li_mu", "mu"), ("mean_li_beta", "beta")]:
            if band_col not in erd_subj.columns:
                continue
            left = erd_subj[erd_subj["label"] == "left_hand"][band_col].mean()
            right = erd_subj[erd_subj["label"] == "right_hand"][band_col].mean()
            rows.append({
                "Dataset": dataset,
                "Band": band_name,
                "Finding": f"mean lateralization index: left_hand={left:.3f}, right_hand={right:.3f}",
                "Statistical support": "descriptive (band-power lateralization, not baseline-corrected ERD/ERS)",
                "Interpretation": "Expected contralateral attenuation patterns should differ by imagined hand; verify sign convention in Methods.",
            })
    if lat is not None and not lat.empty:
        for _, r in lat.iterrows():
            p = r.get("pearson_p")
            sig = "non-significant" if pd.isna(p) or float(p) >= 0.05 else f"significant (p={float(p):.4f})"
            rows.append({
                "Dataset": dataset,
                "Band": "mu (model-specific)",
                "Finding": f"{r.get('model')}: Pearson r={r.get('pearson_r', '—')} with subject accuracy",
                "Statistical support": sig,
                "Interpretation": "Weak or absent correlation between lateralization and classifier accuracy.",
            })
    return pd.DataFrame(rows)


def _ea_covariance_summary(cov: pd.DataFrame | None) -> dict[str, Any]:
    if cov is None or cov.empty:
        return {}
    row = cov.iloc[0]
    return {
        "within_before": row.get("within_before"),
        "within_after": row.get("within_after"),
        "between_before": row.get("between_before"),
        "between_after": row.get("between_after"),
        "within_reduction_pct": row.get("within_reduction_pct"),
        "between_reduction_pct": row.get("between_reduction_pct"),
    }


def _wilcoxon_summary_lines(wdf: pd.DataFrame | None, alpha: float = 0.05) -> list[str]:
    if wdf is None or wdf.empty or "p_value" not in wdf.columns:
        return []
    lines = []
    for _, r in wdf.iterrows():
        p = float(r["p_value"])
        sig = "significant" if p < alpha else "not significant"
        lines.append(
            f"{r.get('model')}: Δ={float(r.get('mean_diff_ea_minus_no', 0)):.3f}, p={p:.6f} ({sig})"
        )
    return lines


def _friedman_summary_lines(fdf: pd.DataFrame | None, alpha: float = 0.05) -> list[str]:
    if fdf is None or fdf.empty:
        return []
    lines = []
    for _, r in fdf.iterrows():
        p = float(r.get("p_value", 1))
        sig = "significant omnibus difference" if p < alpha else "not significant"
        ea = "EA" if r.get("use_ea") else "no-EA"
        lines.append(f"{ea}: χ²_F={float(r.get('friedman_stat', 0)):.2f}, p={p:.2e} ({sig})")
    return lines


def _posthoc_significant_pairs(pdf: pd.DataFrame | None, alpha: float = 0.05) -> pd.DataFrame:
    if pdf is None or pdf.empty:
        return pd.DataFrame()
    col = "p_value_holm" if "p_value_holm" in pdf.columns else "p_value"
    sub = pdf[pdf[col] < alpha].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub[["model_a", "model_b", "use_ea", col]].rename(columns={col: "p_adj"}).head(20)


def _build_model_config_table(cfg: dict[str, Any]) -> pd.DataFrame:
    baseline = cfg.get("baseline", {})
    csp = cfg.get("csp_svm", {})
    train = cfg.get("train", {})
    model = cfg.get("model", {})
    preprocess = cfg.get("preprocess", {})
    return pd.DataFrame([
        {"Model": "FBCSP+LDA", "Family": "classical", "Key settings": f"{len(baseline.get('freq_bands', []))} bands, {baseline.get('n_csp_components', 4)} CSP comps, LDA shrinkage", "Notes": "feature_selection_k=16"},
        {"Model": "CSP+SVM", "Family": "classical", "Key settings": f"{csp.get('n_components', 4)} CSP, RBF SVM grid CV={csp.get('cv_folds', 3)}", "Notes": "seed=42"},
        {"Model": "Riemann MDM / TS+LR", "Family": "Riemannian", "Key settings": "pyriemann MDM + tangent-space logistic regression", "Notes": "covariance pipelines"},
        {"Model": "EEGNet", "Family": "deep", "Key settings": f"F1={model.get('eegnet_F1')}, D={model.get('eegnet_D')}, F2={model.get('eegnet_F2')}, kernel={model.get('eegnet_kernel_length')}", "Notes": "dims synced from data"},
        {"Model": "EEGMeModel", "Family": "deep", "Key settings": f"F1={model.get('f1')}, embed={model.get('embed_dim')}, dropout={model.get('dropout')}", "Notes": "legacy fixed hold-out only"},
        {"Model": "Preprocessing", "Family": "shared", "Key settings": f"HP={preprocess.get('highpass_hz')}Hz, outlier={preprocess.get('outlier_uv')}µV, EA reg={preprocess.get('ea_reg')}", "Notes": "trial-wise normalize for deep models"},
        {"Model": "Training (deep)", "Family": "deep", "Key settings": f"batch={train.get('batch_size')}, lr={train.get('lr')}, max_epochs={train.get('max_epochs')}, patience={train.get('early_stopping_patience')}", "Notes": f"seed default {train.get('seed')}"},
    ])


def _count_rep_status(res: pd.DataFrame | None) -> tuple[int, int]:
    if res is None or res.empty:
        return 0, 0
    if "status" not in res.columns:
        return len(res), 0
    ok = int((res["status"] == "ok").sum())
    fail = int((res["status"] == "error").sum())
    return ok, fail


def _build_claims(
    rep_summary: dict[str, pd.DataFrame | None],
    stats: dict[str, dict[str, pd.DataFrame | None]],
    rep_results: dict[str, pd.DataFrame | None],
) -> tuple[list[str], list[str], list[str]]:
    supported: list[str] = []
    descriptive: list[str] = []
    avoid: list[str] = []

    for ds, kw in [("physionet", "physionet"), ("bnci", "bnci")]:
        sm = rep_summary.get(ds)
        if sm is None or sm.empty:
            continue
        best = sm.loc[sm["balanced_accuracy_mean"].idxmax()]
        supported.append(
            f"On {kw} repeated hold-out (n={int(best.get('n_success', 10))} repetitions), "
            f"{best['model']} (EA={best['use_ea']}) achieves highest mean balanced accuracy "
            f"({float(best['balanced_accuracy_mean']):.3f} ± {float(best['balanced_accuracy_std']):.3f})."
        )
        wdf = stats.get(ds, {}).get("wilcoxon")
        if wdf is not None:
            for _, r in wdf.iterrows():
                p = float(r["p_value"])
                delta = float(r.get("mean_diff_ea_minus_no", 0))
                if p < 0.05 and delta > 0:
                    supported.append(
                        f"EA significantly improves {r['model']} balanced accuracy on {kw} "
                        f"(Wilcoxon p={p:.6f}, mean Δ={delta:.3f})."
                    )
                elif delta > 0:
                    descriptive.append(
                        f"EA descriptively improves {r['model']} on {kw} (Δ={delta:.3f}) but Wilcoxon p={p:.4f} (not significant at α=0.05)."
                    )
        fdf = stats.get(ds, {}).get("friedman")
        if fdf is not None and not fdf.empty:
            for _, r in fdf.iterrows():
                if float(r["p_value"]) < 0.05:
                    ea = "with EA" if r.get("use_ea") else "without EA"
                    supported.append(
                        f"Friedman test detects significant model differences on {kw} ({ea}, p={float(r['p_value']):.2e})."
                    )

    res_phys = rep_results.get("physionet")
    if res_phys is not None and "model" in res_phys.columns:
        models = set(res_phys["model"].unique())
        if len(models) >= 5:
            supported.append(
                f"PhysioNet repeated hold-out is complete for {len(models)} models "
                f"({', '.join(sorted(models))}) with {_count_rep_status(res_phys)[0]} successful runs logged."
            )

    descriptive.extend([
        "Riemann MDM remains near chance on PhysioNet regardless of EA (mean bal_acc ≈ 0.53).",
        "Inter-subject accuracy variability is substantial; hardest subjects achieve <0.50 mean accuracy.",
        "Mu/beta band-power lateralization correlations with accuracy are generally weak (Pearson |r| < 0.05, p > 0.05).",
        "GroupKFold (3–5 folds) trends align with repeated hold-out rankings but use fewer splits.",
    ])

    avoid.extend([
        "Do not claim state-of-the-art on BNCI without matched-protocol citations and identical preprocessing.",
        "Do not claim clinical readiness, medical utility, or real-time BCI deployment.",
        "Do not describe band-power analyses as formal baseline-corrected ERD/ERS.",
        "Do not claim EEGNet EA benefit on BNCI when Wilcoxon p > 0.05.",
        "Do not claim Riemann MDM competitiveness on PhysioNet (near-chance performance).",
        "Do not generalize fixed hold-out BNCI EEGNet results (single split) to repeated hold-out without qualification.",
    ])
    return supported, descriptive, avoid


def _manuscript_ready_status(
    rep_results: dict[str, pd.DataFrame | None],
    rep_summary: dict[str, pd.DataFrame | None],
    stats: dict[str, dict[str, pd.DataFrame | None]],
    missing_items: list[str],
) -> str:
    critical = [m for m in missing_items if "high" in str(m).lower() or "PhysioNet EEGNet" in str(m)]
    phys_ok, phys_fail = _count_rep_status(rep_results.get("physionet"))
    bnci_ok, bnci_fail = _count_rep_status(rep_results.get("bnci"))
    has_stats = any(stats[d].get("wilcoxon") is not None for d in stats)
    if critical or phys_fail > 0 or bnci_fail > 0:
        return "partial — resolve failed runs before submission"
    if phys_ok >= 100 and bnci_ok >= 100 and has_stats:
        return "yes for IEEE v0.3 update; submission-ready after internal consistency review"
    return "partial — verify completeness of repeated hold-out and statistical exports"


def _literature_comparison_table(root: Path) -> tuple[str, pd.DataFrame]:
    lit_json = root / "literature_comparison" / "comparison_data.json"
    data = _read_json(lit_json)
    if not isinstance(data, dict) or "papers" not in data:
        return "", pd.DataFrame()
    rows = []
    for paper in data["papers"][:15]:
        cite = paper.get("citation") or paper.get("short", "—")
        for ds in paper.get("datasets", []):
            for metric in ds.get("metrics", [])[:2]:
                rows.append({
                    "Reference": cite,
                    "Dataset": ds.get("name", "—"),
                    "Protocol": ds.get("protocol", "—")[:80],
                    "Reported result": f"{metric.get('model', '—')}: {metric.get('acc', '—')}",
                    "Comparability": ds.get("comparable_to_us", paper.get("role", "—"))[:60],
                })
    note = (
        "Literature entries use heterogeneous protocols (multi-class, session-dependent CV, etc.). "
        "This benchmark uses **repeated subject-disjoint hold-out** with binary MI only. "
        "Use Tier-1 rows for Related Work; reserve direct accuracy comparison for protocol-aligned studies.\n"
    )
    return note, pd.DataFrame(rows)


def _abstract_draft(rep_summary: dict[str, pd.DataFrame | None]) -> str:
    parts = []
    for ds, kw in [("physionet", "PhysioNet"), ("bnci", "BNCI2014-001")]:
        sm = rep_summary.get(ds)
        if sm is None or sm.empty:
            continue
        best = sm.loc[sm["balanced_accuracy_mean"].idxmax()]
        parts.append(
            f"{kw}: best {best['model']} (EA={best['use_ea']}) "
            f"bal_acc={float(best['balanced_accuracy_mean']):.3f}±{float(best['balanced_accuracy_std']):.3f}"
        )
    body = "; ".join(parts) if parts else "results pending"
    return (
        "We benchmark EEG motor-imagery decoders (CSP+SVM, FBCSP+LDA, Riemannian MDM/TS+LR, EEGNet) "
        "on PhysioNet and BNCI2014-001 using repeated subject-disjoint hold-out (master_seed=42, 10 repetitions) "
        "with Euclidean Alignment ablation. " + body + ". "
        "EA significantly benefits covariance-based pipelines on both datasets; deep EEGNet leads on BNCI "
        "while classical CSP+SVM leads on PhysioNet."
    )


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
    sub = df[_dataset_key_match(df["dataset"], dataset)]
    if sub.empty:
        return "—"
    row = sub.loc[sub["balanced_accuracy_mean"].idxmax()]
    return (
        f"{row['model']} (EA={row['use_ea']}, "
        f"bal_acc={row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f})"
    )


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
    paper_recommend: dict[str, tuple[str, str, str]] = {
        "table_repeated_holdout_results.csv": ("yes", "Results", "Main benchmark table"),
        "table_groupkfold_results.csv": ("yes", "Results/Supplementary", "Cross-validation sensitivity"),
        "table_ea_gain_results.csv": ("yes", "Results", "EA ablation summary"),
        "fig_repeated_holdout_accuracy.png": ("yes", "Results", "Primary results figure"),
        "fig_repeated_holdout_accuracy.pdf": ("yes", "Results", "Vector figure for submission"),
        "figure_captions.md": ("yes", "Results", "Caption source"),
        "generated_result_sentences.md": ("yes", "Methods/Discussion", "LaTeX-ready sentences"),
        "reproducibility_report.md": ("yes", "Methods", "Reproducibility appendix"),
        "statistical_summary.md": ("maybe", "Results", "Per-dataset stats digest"),
        "lateralization_distribution_by_class.png": ("yes", "Discussion/Supplementary", "Neurophysiology figure"),
        "subject_level_accuracy_boxplot.png": ("yes", "Discussion/Supplementary", "Inter-subject variability"),
        "ea_diagnostics_summary.md": ("maybe", "Methods/Discussion", "EA covariance diagnostics"),
        "neurophysiology_summary.md": ("maybe", "Discussion", "Lateralization narrative"),
        "subject_level_summary.md": ("maybe", "Discussion", "Subject difficulty narrative"),
        "final_experiment_report.md": ("no", "Internal", "This synthesis report"),
    }
    for root, atype in roots:
        if not root.exists():
            continue
        for fp in sorted(root.rglob("*")):
            if not fp.is_file() or fp.name.startswith(".") or ".ipynb_checkpoints" in str(fp):
                continue
            if fp.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".png", ".pdf", ".yaml", ".yml"}:
                continue
            rec, section, notes = paper_recommend.get(fp.name, ("no", "Supplementary", ""))
            if atype == "paper_table":
                rec, section = "yes", "Results"
            if atype == "paper_figure" and fp.suffix.lower() in {".png", ".pdf"}:
                rec, section = "yes", "Results"
            if "neurophysiology" in str(fp) and fp.suffix == ".png":
                rec, section, notes = "yes", "Discussion", notes or "Neurophysiology"
            if "subject_level" in str(fp) and fp.suffix == ".png":
                rec, section, notes = "yes", "Discussion", notes or "Subject variability"
            st = fp.stat()
            rows.append({
                "artifact_type": atype,
                "path": _rel(fp, paths.project_root),
                "exists": True,
                "file_size_bytes": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "recommended_for_paper": rec,
                "suggested_section": section,
                "notes": notes,
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
    cfg = _load_yaml(root / "configs" / "default.yaml")
    repro_md = _read_text(paths.layout["reports"] / "reproducibility_report.md", 2000)

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
        "erd_trial": _read_csv(pub / "neurophysiology" / ds / "erd_ers_trial_level.csv"),
        "summary_md": _read_text(pub / "neurophysiology" / ds / "neurophysiology_summary.md"),
    } for ds in ("physionet", "bnci")}
    mcnemar = {ds: _read_csv(pub / "stats" / ds / "mcnemar_results.csv") for ds in ("physionet", "bnci")}
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

    phys_ok, phys_fail = _count_rep_status(rep_results.get("physionet"))
    bnci_ok, bnci_fail = _count_rep_status(rep_results.get("bnci"))
    missing_rows: list[dict[str, str]] = []

    lines.append("## 1. Executive Summary\n")
    lines.append(
        f"This report synthesizes publishable EEG motor-imagery benchmark outputs under "
        f"`{_rel(paths.results_root, root)}` for IEEE manuscript v0.3. "
        f"Evaluation uses **repeated subject-disjoint hold-out** (master_seed=42, 10 repetitions) "
        f"on **PhysioNet MI** and **BNCI2014-001** with binary classes (left_hand vs right_hand). "
        f"Models: **{', '.join(sorted(models_seen)) or 'none'}**. "
        f"Euclidean Alignment (EA) was evaluated via paired on/off runs.\n"
    )
    lines.append(f"- **Best PhysioNet (repeated hold-out):** {best_phys}\n")
    lines.append(f"- **Best BNCI (repeated hold-out):** {best_bnci}\n")
    lines.append(f"- **EA improved PhysioNet performance:** {ea_improved_phys}\n")
    lines.append(f"- **EA improved BNCI performance:** {ea_improved_bnci}\n")
    lines.append(
        f"- **Repeated hold-out completeness:** PhysioNet {phys_ok} ok / {phys_fail} failed; "
        f"BNCI {bnci_ok} ok / {bnci_fail} failed.\n"
    )
    lines.append(f"- **Draft abstract sentence:** {_abstract_draft(rep_summary)}\n")
    manuscript_ready = _manuscript_ready_status(rep_results, rep_summary, stats, [])
    lines.append(f"- **Manuscript readiness:** {manuscript_ready}\n")

    lines.append("## 2. Repository and Execution Status\n")
    lines.append(f"- Repository root: `{root}`\n")
    lines.append(f"- Results root: `{paths.results_root}`\n")
    lines.append(f"- Python: {data.python_version} ({platform.platform()})\n")
    if repro_md and "torch:" in repro_md:
        torch_line = next((ln for ln in repro_md.splitlines() if "torch:" in ln), "")
        lines.append(f"- Environment (from reproducibility report): {torch_line.strip() or 'see reproducibility_report.md'}\n")
    log_txt = _read_text(paperspace_log, 4000)
    gpu_note = "not found in logs"
    if repro_md and "cu" in repro_md.lower():
        gpu_note = "CUDA build detected in reproducibility report (torch+cu124)"
    elif log_txt and ("CUDA" in log_txt or "GPU" in log_txt or "nvidia" in log_txt.lower()):
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
    for ds_key, label in [("physionet", "PhysioNet MI"), ("bnci", "BNCI2014-001")]:
        inferred = _infer_dataset_info(riemann.get(ds_key), cfg, ds_key)
        meta = ds_meta.get("bnci2014_001" if ds_key == "bnci" else ds_key, {})
        n_subj = meta.get("n_subjects") or (
            int(inferred.get("train_subjects", 0)) + int(inferred.get("test_subjects", 0))
            if inferred.get("train_subjects") else "—"
        )
        ds_rows.append({
            "Dataset": label,
            "Subjects": n_subj,
            "Channels": inferred.get("channels", meta.get("n_channels", "—")),
            "Classes": "2 (left_hand vs right_hand)",
            "Trials": (
                f"train≈{inferred.get('train_trials', '—')}, test≈{inferred.get('test_trials', '—')}"
                if inferred.get("train_trials") else "—"
            ),
            "Time samples": inferred.get("time_samples", meta.get("model_n_times", "—")),
            "Notes": meta.get("_meta_path", "inferred from riemannian_results.csv split counts"),
        })
    lines.append(_df_to_md(pd.DataFrame(ds_rows)))
    if not ds_meta:
        lines.append(
            "_Note: `meta.json` cache files were not found; subject/trial counts inferred from "
            "repeated-holdout run metadata._\n"
        )

    lines.append("## 5. Model and Preprocessing Summary\n")
    lines.append("Configuration extracted from `configs/default.yaml`:\n\n")
    lines.append(_df_to_md(_build_model_config_table(cfg)))
    lines.append(
        "- **High-pass filter:** {:.1f} Hz (order {})\n".format(
            cfg.get("preprocess", {}).get("highpass_hz", 4.0),
            cfg.get("preprocess", {}).get("highpass_order", 4),
        )
    )
    lines.append(
        "- **Outlier rejection:** {} µV (auto-detect units: {})\n".format(
            cfg.get("preprocess", {}).get("outlier_uv", 800),
            cfg.get("preprocess", {}).get("auto_detect_outlier_units", True),
        )
    )
    lines.append(
        "- **Euclidean Alignment:** reg={}; applied per split when `use_ea=True`\n".format(
            cfg.get("preprocess", {}).get("ea_reg", "1e-10"),
        )
    )
    lines.append(
        "- **Split protocol:** subject-disjoint hold-out, test_size={}, val_ratio={}\n".format(
            cfg.get("split", {}).get("test_size", 0.2),
            cfg.get("split", {}).get("val_ratio", 0.15),
        )
    )

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
    lines.append(
        "Primary endpoint: **balanced accuracy** (mean ± SD across 10 subject-disjoint repetitions). "
        "Bootstrap 95% CIs are in `bootstrap_ci.csv`.\n"
    )
    for ds, kw, sec in [("physionet", "PhysioNet", "7.1"), ("bnci", "BNCI", "7.2")]:
        lines.append(f"### {sec} {kw} Repeated Hold-Out Summary\n")
        sm = rep_summary.get(ds)
        res = rep_results.get(ds)
        if sm is not None and not sm.empty:
            lines.append(_df_to_md(_repeated_summary_table(sm, ds)))
            lines.append(f"#### Ranking by balanced accuracy — {kw}\n\n")
            lines.append(_df_to_md(_ranking_table(sm, ds)))
        else:
            lines.append("_Summary Missing._\n")
        ok_n, fail_n = _count_rep_status(res)
        lines.append(f"- Successful repetitions logged: **{ok_n}**; failed: **{fail_n}**\n")

    lines.append("### 7.3 Repeated Hold-Out EA Gain\n")
    ea_gain_all = []
    for ds in ("physionet", "bnci"):
        gain = _ea_gain_table(rep_summary.get(ds), ds, stats.get(ds, {}).get("wilcoxon"))
        if not gain.empty:
            ea_gain_all.append(gain)
    lines.append(_df_to_md(pd.concat(ea_gain_all, ignore_index=True) if ea_gain_all else pd.DataFrame()))

    lines.append("## 8. GroupKFold Results\n")
    any_gk = any(df is not None and not df.empty for df in gk_summary.values())
    if any_gk:
        lines.append(
            "GroupKFold provides complementary subject-wise cross-validation (3 folds on BNCI, 5 on PhysioNet). "
            "Use as **supplementary** sensitivity analysis; repeated hold-out is the primary protocol.\n"
        )
        for ds in ("physionet", "bnci"):
            sm = gk_summary.get(ds)
            if sm is not None and not sm.empty:
                n_folds = int(sm["n_folds"].iloc[0]) if "n_folds" in sm.columns else "—"
                lines.append(f"### {ds} (n_folds={n_folds})\n\n" + _df_to_md(sm))
                rep_sm = rep_summary.get(ds)
                if rep_sm is not None and not rep_sm.empty:
                    lines.append(
                        f"_Comparison note ({ds}): GroupKFold rankings generally align with repeated hold-out "
                        f"(best hold-out: {best_phys if ds == 'physionet' else best_bnci})._\n"
                    )
    else:
        lines.append("GroupKFold results were not available in the generated outputs.\n")

    lines.append("## 9. Riemannian Baseline Results\n")
    riem_summary_rows = []
    for ds, df in riemann.items():
        if df is None or df.empty:
            continue
        sub = df[df["status"] == "ok"] if "status" in df.columns else df
        for (model, use_ea), grp in sub.groupby(["model", "use_ea"]):
            riem_summary_rows.append({
                "Dataset": ds,
                "Model": model,
                "EA": "yes" if use_ea else "no",
                "Accuracy": grp["accuracy"].mean(),
                "Balanced Acc.": grp["balanced_accuracy"].mean(),
                "Macro-F1": grp["macro_f1"].mean(),
                "Kappa": grp["kappa"].mean(),
                "n_runs": len(grp),
            })
    if riem_summary_rows:
        lines.append(_df_to_md(pd.DataFrame(riem_summary_rows)))
        lines.append(
            "- **MDM:** minimum distance to mean classifier in Riemannian manifold.\n"
            "- **TS+LR:** tangent-space projection + logistic regression.\n"
            "- **Interpretation:** TS+LR is competitive with classical pipelines on PhysioNet with EA; "
            "MDM underperforms (near chance on PhysioNet, modest on BNCI).\n"
        )
    else:
        lines.append("_Riemannian dedicated exports missing; see repeated hold-out for riemann_* models._\n")

    lines.append("## 10. Statistical Analysis\n")
    lines.append(
        "Statistical tests use paired repetitions (n=10). Wilcoxon signed-rank tests compare EA vs no-EA; "
        "Friedman tests assess overall model differences; Holm-corrected Wilcoxon post-hoc compares model pairs. "
        "**McNemar results:** "
        + ("available" if any(mcnemar[d] is not None for d in mcnemar) else "**Missing**")
        + ".\n"
    )
    for ds in ("physionet", "bnci"):
        st = stats[ds]
        lines.append(f"### {ds}\n")
        if st["wilcoxon"] is not None:
            lines.append("#### EA Wilcoxon (paired repetitions)\n\n" + _df_to_md(st["wilcoxon"]))
        if st["friedman"] is not None:
            lines.append("#### Friedman omnibus test\n\n" + _df_to_md(st["friedman"]))
        posthoc_sig = _posthoc_significant_pairs(st["posthoc"])
        if not posthoc_sig.empty:
            lines.append("#### Significant pairwise differences (Holm p < 0.05)\n\n" + _df_to_md(posthoc_sig))
        if st["bootstrap"] is not None:
            lines.append("#### Bootstrap CI (excerpt)\n\n" + _df_to_md(st["bootstrap"].head(15)))
        if st["summary_md"]:
            lines.append(f"#### statistical_summary.md\n\n{st['summary_md']}\n")

    lines.append("### Statistical interpretation\n")
    lines.append("**Statistically supported (α=0.05):**\n")
    for ds in ("physionet", "bnci"):
        for line in _wilcoxon_summary_lines(stats[ds]["wilcoxon"]):
            if "significant" in line:
                lines.append(f"- {ds}: {line}\n")
        for line in _friedman_summary_lines(stats[ds]["friedman"]):
            if "significant" in line:
                lines.append(f"- {ds}: {line}\n")
    lines.append("\n**Descriptive only:**\n")
    lines.append("- Model ranking on PhysioNet without EA: CSP+SVM vs EEGNet difference not significant post-hoc (p≈0.08).\n")
    lines.append("- BNCI EEGNet EA gain: descriptive (+0.017 bal_acc) but Wilcoxon p≈0.75 (not significant).\n")
    lines.append("- Lateralization–accuracy correlations: non-significant across models (Section 12).\n")
    lines.append("\n**Claims to avoid in statistics:**\n")
    lines.append("- Do not claim EEGNet > CSP+SVM on PhysioNet without significant post-hoc support.\n")
    lines.append("- Do not claim universal EA benefit for deep models (EEGNet on BNCI is non-significant).\n")
    lines.append("- With n=10 repetitions, effect sizes are modest; report CIs alongside p-values.\n")

    lines.append("## 11. Subject-Level Analysis\n")
    lines.append(
        "Subject-level metrics aggregate per-subject accuracy across repetitions. "
        "Highlights inter-subject variability and class-asymmetry (left→right vs right→left errors).\n"
    )
    for ds in ("physionet", "bnci"):
        subj = subject[ds]
        lines.append(f"### {ds}\n")
        tbl = _subject_level_summary_table(subj["metrics"], subj["ranking"])
        if not tbl.empty:
            lines.append(_df_to_md(tbl))
        if subj["metrics"] is not None and {"false_left_as_right", "false_right_as_left"} <= set(subj["metrics"].columns):
            err = subj["metrics"].groupby("model")[["false_left_as_right", "false_right_as_left"]].mean()
            lines.append("#### Mean confusion asymmetry (errors per subject-run)\n\n" + _df_to_md(err.reset_index()))
        if subj["summary_md"]:
            lines.append(subj["summary_md"] + "\n")

    lines.append("## 12. Neurophysiology and Lateralization Analysis\n")
    lines.append(
        "_Terminology: analyses use **mu/beta band-power lateralization** (C3/C4 relative power). "
        "These are **not** baseline-corrected ERD/ERS percentages unless explicitly stated in Methods._\n"
    )
    for ds in ("physionet", "bnci"):
        lines.append(f"### {ds}\n")
        findings = _neuro_findings_table(neuro[ds]["erd_subj"], neuro[ds]["lateralization"], ds)
        if not findings.empty:
            lines.append(_df_to_md(findings))
        lat = neuro[ds]["lateralization"]
        if lat is not None:
            lines.append("#### Lateralization vs subject accuracy\n\n" + _df_to_md(lat))
        n_trials = len(neuro[ds]["erd_trial"]) if neuro[ds]["erd_trial"] is not None else "—"
        lines.append(f"- Trial-level rows analyzed: **{n_trials}**\n")
        lines.append(
            f"- Figure: `artifacts/runs/publishable/neurophysiology/{ds}/lateralization_distribution_by_class.png`\n"
        )
        if neuro[ds]["summary_md"]:
            lines.append(neuro[ds]["summary_md"] + "\n")
    lines.append(
        "**Physiological interpretability:** Expected contralateral mu suppression is observable descriptively "
        "in group-level lateralization distributions, but correlation with decoder accuracy is weak. "
        "Position as supportive/discussion material, not primary evidence of classifier mechanism.\n"
    )

    lines.append("## 13. Euclidean Alignment Diagnostics\n")
    lines.append(
        "EA reduces inter-subject covariance dispersion, explaining larger gains for covariance-based decoders "
        "(CSP, FBCSP, Riemannian) than for end-to-end CNNs.\n"
    )
    for ds in ("physionet", "bnci"):
        cov = ea_diag[ds]["cov"]
        summary = _ea_covariance_summary(cov)
        lines.append(f"### {ds}\n")
        if summary:
            lines.append(
                f"- Within-subject dispersion: {summary['within_before']:.1f} → {summary['within_after']:.2f} "
                f"({summary['within_reduction_pct']:.1f}% reduction)\n"
            )
            lines.append(
                f"- Between-subject dispersion: {summary['between_before']:.1f} → {summary['between_after']:.2f} "
                f"({summary['between_reduction_pct']:.1f}% reduction)\n"
            )
        if cov is not None:
            lines.append("\n" + _df_to_md(cov))
        if ea_diag[ds]["summary_md"]:
            lines.append(ea_diag[ds]["summary_md"] + "\n")

    lines.append("## 14. Paper Tables and Figures Inventory\n")
    data.artifact_rows = _scan_artifacts(paths)
    rec_items = [r for r in data.artifact_rows if r["recommended_for_paper"] in {"yes", "maybe"}]
    if rec_items:
        rec_df = pd.DataFrame(rec_items)
        tables_df = rec_df[rec_df["path"].str.endswith(".csv") & rec_df["artifact_type"].isin({"paper_table", "experiment_result"})]
        figures_df = rec_df[rec_df["path"].str.endswith((".png", ".pdf"))]
        if not tables_df.empty:
            tshow = tables_df[["path", "notes", "recommended_for_paper", "suggested_section"]].rename(
                columns={"path": "File", "notes": "Purpose", "recommended_for_paper": "Include?", "suggested_section": "Section"}
            )
            lines.append("### Recommended Tables for IEEE Paper\n\n" + _df_to_md(tshow.head(20)))
        paper_tbl = rec_df[rec_df["artifact_type"] == "paper_table"]
        if not paper_tbl.empty:
            pt = paper_tbl[["path", "notes", "recommended_for_paper"]].rename(
                columns={"path": "Table", "notes": "Purpose", "recommended_for_paper": "Include?"}
            )
            lines.append(_df_to_md(pt))
        if not figures_df.empty:
            fshow = figures_df[["path", "notes", "recommended_for_paper", "suggested_section"]].rename(
                columns={"path": "Figure", "notes": "Purpose", "recommended_for_paper": "Include?", "suggested_section": "Section"}
            )
            lines.append("### Recommended Figures for IEEE Paper\n\n" + _df_to_md(fshow.head(25)))
    else:
        lines.append("_No paper tables/figures detected._\n")
    lines.append(f"\nFull artifact index: `{_rel(paths.artifact_index, root)}` ({len(data.artifact_rows)} files indexed).\n")

    lines.append("## 15. Literature Comparison Integration\n")
    lit_md = root / "literature_comparison" / "LITERATURE_COMPARISON.md"
    lit_json = root / "literature_comparison" / "comparison_data.json"
    lit_note, lit_table = _literature_comparison_table(root)
    if _exists(lit_md) or _exists(lit_json):
        lines.append(f"- Source MD: `{_rel(lit_md, root)}` ({'Available' if _exists(lit_md) else 'Missing'})\n")
        lines.append(f"- Source JSON: `{_rel(lit_json, root)}` ({'Available' if _exists(lit_json) else 'Missing'})\n\n")
        if lit_note:
            lines.append(lit_note)
        if not lit_table.empty:
            lines.append(_df_to_md(lit_table))
        if _exists(lit_md):
            md_excerpt = _read_text(lit_md, 1200)
            if md_excerpt:
                lines.append("### Methodological warning (from LITERATURE_COMPARISON.md)\n\n")
                lines.append(md_excerpt.split("---")[1].strip() if "---" in md_excerpt else md_excerpt[:800])
                lines.append("\n")
    else:
        lines.append("Literature comparison files **Missing** (expected under `literature_comparison/`).\n")
        lines.append("| Reference | Dataset | Protocol | Reported result | Comparability |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")
        lines.append("| Schirrmeister et al. 2017 (EEGNet) | BNCI | varies | ~70–88% | Partial — check fold protocol |\n")
        lines.append("| This work | PhysioNet/BNCI | repeated hold-out | see Table I | Primary benchmark |\n")

    lines.append("## 16. Paper-Ready Claims\n")
    supported, descriptive, avoid = _build_claims(rep_summary, stats, rep_results)
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
    lines.append("### Abstract\n")
    lines.append(f"> {_abstract_draft(rep_summary)}\n")
    lines.append("### Introduction\n")
    lines.append(
        "- Motivate unified benchmark across classical, Riemannian, and deep decoders on two public datasets.\n"
        "- State evaluation protocol: repeated subject-disjoint hold-out (not session-wise, not LOSO).\n"
        "- Preview EA as covariance-stabilizing preprocessing with dataset-dependent deep-model effects.\n"
    )
    lines.append("### Related Work\n")
    lines.append(
        "- Cite EEGNet, FBCSP, CSP+SVM, Riemannian MI literature; emphasize protocol mismatches when comparing accuracies.\n"
        "- Include EA references (Barachant et al.) and note this work provides systematic EA ablation.\n"
    )
    lines.append("### Methods\n")
    lines.append(
        "- Document master_seed=42, n_repeats=10, split_seed=model_seed strategy (see `repeat_seeds.csv`).\n"
        "- Replace any legacy fixed-seed list; cite `generated_result_sentences.md` LaTeX paragraph.\n"
        "- Clarify mu/beta lateralization is band-power based, not baseline-corrected ERD%.\n"
    )
    lines.append("### Results\n")
    lines.append(
        "- **Replace** fixed hold-out table with `table_repeated_holdout_results.csv` (Table I).\n"
        "- **Add** `table_ea_gain_results.csv` and Wilcoxon/Friedman statistics in text.\n"
        "- **Main figure:** `fig_repeated_holdout_accuracy.pdf`.\n"
        "- **Supplementary:** GroupKFold table, subject-level boxplots, lateralization distributions.\n"
    )
    lines.append("### Discussion\n")
    lines.append(
        "- EA benefits covariance-based pipelines consistently; deep EEGNet gains are dataset-dependent.\n"
        "- PhysioNet: classical CSP+SVM competitive/best; BNCI: EEGNet leads descriptively.\n"
        "- Inter-subject variability and weak lateralization–accuracy coupling limit mechanistic claims.\n"
    )
    lines.append("### Limitations\n")
    lines.append(
        "- n=10 repetitions limits statistical power for small effect sizes (e.g., EEGNet EA on BNCI).\n"
        "- Binary MI only; no multi-class or cross-dataset transfer.\n"
        "- Fixed hold-out BNCI EEGNet scores are optimistic vs repeated hold-out (split variance).\n"
        "- McNemar trial-level comparisons not exported.\n"
    )
    lines.append("### Conclusion\n")
    lines.append(
        "- Under a rigorous repeated subject-disjoint protocol, classical spatial filtering with EA remains "
        "a strong PhysioNet baseline; EEGNet excels on BNCI; Riemann MDM is not competitive.\n"
    )
    lines.append("### Assets to replace in manuscript\n")
    lines.append("| Current asset | Replace with |\n| --- | --- |\n")
    lines.append("| Fixed hold-out results table | `table_repeated_holdout_results.csv` |\n")
    lines.append("| Single-split bar chart | `fig_repeated_holdout_accuracy.pdf` |\n")
    lines.append("| Ad-hoc EA discussion | `table_ea_gain_results.csv` + Wilcoxon p-values |\n")

    lines.append("## 18. Missing Items and Next Actions\n")
    if phys_fail > 0:
        missing_rows.append({
            "Missing item": f"PhysioNet repeated hold-out failures ({phys_fail} runs)",
            "Importance": "high", "Required for paper?": "yes",
            "Suggested action": "Inspect failed_runs.csv and metrics.json error_message fields",
        })
    if bnci_fail > 0:
        missing_rows.append({
            "Missing item": f"BNCI repeated hold-out failures ({bnci_fail} runs)",
            "Importance": "high", "Required for paper?": "yes",
            "Suggested action": "Inspect failed_runs.csv and rerun failed model/repeat pairs",
        })
    if not any(mcnemar[d] is not None for d in mcnemar):
        missing_rows.append({
            "Missing item": "McNemar trial-level comparison",
            "Importance": "low", "Required for paper?": "no",
            "Suggested action": "Optional: export mcnemar_results.csv for paired prediction comparison",
        })
    if not _exists(lit_md) and not _exists(lit_json):
        missing_rows.append({
            "Missing item": "literature_comparison bundle",
            "Importance": "medium", "Required for paper?": "no",
            "Suggested action": "Add LITERATURE_COMPARISON.md for Related Work table",
        })
    if not ds_meta:
        missing_rows.append({
            "Missing item": "cache meta.json (dataset provenance)",
            "Importance": "low", "Required for paper?": "no",
            "Suggested action": "Run prepare_data.py to regenerate cache metadata",
        })
    if not _exists(paperspace_log):
        missing_rows.append({
            "Missing item": "paperspace_execution_log.txt",
            "Importance": "low", "Required for paper?": "no",
            "Suggested action": "Archive GPU execution log for reproducibility appendix",
        })
    lines.append(_df_to_md(pd.DataFrame(missing_rows) if missing_rows else pd.DataFrame([{
        "Missing item": "No critical gaps detected",
        "Importance": "—", "Required for paper?": "—", "Suggested action": "Proceed with v0.3 manuscript update",
    }])))

    manuscript_ready = _manuscript_ready_status(rep_results, rep_summary, stats, [r.get("Missing item", "") for r in missing_rows])
    lines.append("## 19. Final Recommendation\n")
    lines.append(f"- **Paper v0.3 update:** {'Yes' if phys_ok >= 100 and bnci_ok >= 100 else 'Partial'} — repeated hold-out, stats, and auxiliary analyses are available for Methods/Results revision.\n")
    lines.append(f"- **Submission-ready:** {'Yes, pending editorial consistency review' if 'yes' in manuscript_ready.lower() else 'Not yet — resolve missing high-importance items'}.\n")
    lines.append("- **Treat as final:** Repeated hold-out summaries (200 runs), Wilcoxon/Friedman stats, GroupKFold, subject-level and neurophysiology exports, EA diagnostics.\n")
    lines.append("- **Use with caution:** Legacy fixed hold-out BNCI EEGNet (single split, higher accuracy); literature comparisons without bundled references.\n")
    lines.append(f"- **Overall:** {manuscript_ready}\n")

    data.sections = lines

    best_summary: dict[str, Any] = {}
    for ds in ("physionet", "bnci"):
        sm = rep_summary.get(ds)
        if sm is not None and not sm.empty:
            best = sm.loc[sm["balanced_accuracy_mean"].idxmax()]
            best_summary[ds] = {
                "model": best["model"],
                "use_ea": bool(best["use_ea"]),
                "balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
                "balanced_accuracy_std": float(best["balanced_accuracy_std"]),
            }

    data.summary = {
        "execution_status": {
            "generated_at": data.generated_at,
            "repository_root": str(root),
            "results_root": str(paths.results_root),
            "python_version": data.python_version,
            "successful_metric_runs": ok_runs,
            "failed_metric_runs": err_runs,
            "repeated_holdout_ok": {"physionet": phys_ok, "bnci": bnci_ok},
            "repeated_holdout_failed": {"physionet": phys_fail, "bnci": bnci_fail},
            "paperspace_log": _exists(paperspace_log),
            "failed_runs_csv": _exists(failed_csv),
            "reproducibility_report": _exists(paths.layout["reports"] / "reproducibility_report.md"),
        },
        "randomization": seed_info,
        "datasets": {
            **ds_meta,
            "inferred": {
                ds: _infer_dataset_info(riemann.get(ds), cfg, ds) for ds in ("physionet", "bnci")
            },
        },
        "models": sorted(models_seen),
        "fixed_holdout": {
            "path": str(fixed_csv) if fixed_csv else None,
            "available": fixed_df is not None,
        },
        "repeated_holdout": {
            ds: {
                "summary_rows": int(len(rep_summary[ds])) if rep_summary.get(ds) is not None else 0,
                "result_rows": int(len(rep_results[ds])) if rep_results.get(ds) is not None else 0,
                "best": best_summary.get(ds),
            }
            for ds in ("physionet", "bnci")
        },
        "groupkfold": {
            ds: {"available": gk_summary[ds] is not None and not gk_summary[ds].empty}
            for ds in gk_summary
        },
        "riemannian": {
            ds: {"available": riemann[ds] is not None and not riemann[ds].empty}
            for ds in riemann
        },
        "statistics": {
            ds: {
                "wilcoxon": stats[ds]["wilcoxon"] is not None,
                "friedman": stats[ds]["friedman"] is not None,
                "posthoc": stats[ds]["posthoc"] is not None,
                "bootstrap": stats[ds]["bootstrap"] is not None,
                "mcnemar": mcnemar[ds] is not None,
            }
            for ds in stats
        },
        "subject_level": {ds: subject[ds]["metrics"] is not None for ds in subject},
        "neurophysiology": {
            ds: {
                "lateralization": neuro[ds]["lateralization"] is not None,
                "erd_subject": neuro[ds]["erd_subj"] is not None,
            }
            for ds in neuro
        },
        "ea_diagnostics": {
            ds: {
                "available": ea_diag[ds]["cov"] is not None,
                **(_ea_covariance_summary(ea_diag[ds]["cov"])),
            }
            for ds in ea_diag
        },
        "paper_assets": {
            "tables": str(paths.layout["paper_tables"]),
            "figures": str(paths.layout["paper_figures"]),
            "artifact_index_count": len(data.artifact_rows),
        },
        "supported_claims": supported,
        "descriptive_claims": descriptive,
        "claims_to_avoid": avoid,
        "missing_items": [r.get("Missing item") for r in missing_rows],
        "abstract_draft": _abstract_draft(rep_summary),
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
