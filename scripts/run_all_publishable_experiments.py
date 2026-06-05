#!/usr/bin/env python3
"""Orchestrate the full publishable experiment pipeline (Paperspace-ready, resumable)."""

from __future__ import annotations

import argparse
import csv
import importlib
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

DEEP_MODELS = frozenset({"eegnet", "eegme"})
RIEMANN_MODELS = frozenset({"riemann_mdm", "riemann_ts_lr", "riemann_fgmdm", "riemann_ts_svm"})

STAGE_ORDER = [
    "verify_env",
    "verify_data",
    "repeated_holdout",
    "groupkfold",
    "riemannian",
    "stats",
    "subject_level",
    "neurophysiology",
    "ea_diagnostics",
    "paper_tables",
    "paper_figures",
    "reproducibility_report",
    "paper_text_snippets",
    "implementation_summary",
]

STAGE_CHOICES = ["all", *STAGE_ORDER, "data", "paper"]


class ExecutionLogger:
    """Mirror stdout to a persistent log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = message.rstrip("\n")
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def section(self, title: str) -> None:
        bar = "=" * 72
        self.write("")
        self.write(bar)
        self.write(title)
        self.write(bar)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_failed_run(
    failed_csv: Path,
    *,
    stage: str,
    dataset: str,
    command: str,
    returncode: int,
    error_message: str,
) -> None:
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _now_iso(),
        "stage": stage,
        "dataset": dataset,
        "command": command,
        "returncode": returncode,
        "error_message": error_message,
    }
    write_header = not failed_csv.exists()
    with failed_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_cmd(
    cmd: list[str],
    *,
    logger: ExecutionLogger,
    stage: str,
    dataset: str,
    dry_run: bool,
    failed_csv: Path,
    continue_on_error: bool = True,
) -> int:
    """Run a subprocess; log failures and optionally continue."""
    cmd_str = " ".join(cmd)
    logger.write(f"$ {cmd_str}")
    if dry_run:
        return 0
    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    except Exception as exc:
        _append_failed_run(
            failed_csv,
            stage=stage,
            dataset=dataset,
            command=cmd_str,
            returncode=-1,
            error_message=str(exc),
        )
        logger.write(f"ERROR launching command: {exc}")
        if not continue_on_error:
            raise
        return -1

    if proc.stdout:
        logger.write(proc.stdout.rstrip())
    if proc.stderr:
        logger.write(proc.stderr.rstrip())

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "non-zero exit").strip()[:2000]
        _append_failed_run(
            failed_csv,
            stage=stage,
            dataset=dataset,
            command=cmd_str,
            returncode=proc.returncode,
            error_message=err,
        )
        logger.write(f"FAILED (exit {proc.returncode}): {cmd_str}")
        if not continue_on_error:
            sys.exit(proc.returncode)
    else:
        logger.write(f"OK: {cmd_str}")
    return proc.returncode


def _ea_flag(ea: str) -> str:
    if ea in ("true", "false", "both"):
        return ea
    raise ValueError(f"--ea must be true|false|both, got {ea!r}")


def _resolve_models(
    models: list[str],
    *,
    only_classical: bool,
    skip_deep: bool,
) -> list[str]:
    out = list(models)
    if only_classical:
        out = [m for m in out if m not in DEEP_MODELS]
    if skip_deep:
        out = [m for m in out if m not in DEEP_MODELS]
    return out


def _dataset_key(ds: str) -> str:
    return "bnci2014_001" if ds == "bnci" else ds


def verify_environment(logger: ExecutionLogger) -> None:
    logger.section("STAGE 1: Verify environment and dependencies")
    logger.write(f"Python: {sys.version.split()[0]} on {platform.platform()}")

    required = ["numpy", "scipy", "sklearn", "pandas", "mne", "matplotlib", "torch"]
    optional = ["moabb", "pyriemann"]
    for pkg in required:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            ver = getattr(mod, "__version__", "unknown")
            logger.write(f"  [ok] {pkg} {ver}")
        except ImportError:
            logger.write(f"  [MISSING] {pkg}")

    for pkg in optional:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.write(f"  [ok] {pkg} {ver}")
        except ImportError:
            logger.write(f"  [optional missing] {pkg}")

    try:
        import torch

        if torch.cuda.is_available():
            logger.write(f"  [ok] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.write("  [warn] CUDA not available (deep models will use CPU)")
    except Exception as exc:
        logger.write(f"  [warn] torch CUDA check failed: {exc}")

    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            logger.write(proc.stdout.splitlines()[0] if proc.stdout else "nvidia-smi: ok")
        else:
            logger.write("nvidia-smi: not available")
    except FileNotFoundError:
        logger.write("nvidia-smi: not found")


def _cache_marker_paths(dataset: str) -> list[Path]:
    ds = _dataset_key(dataset)
    if ds == "physionet":
        patterns = [
            "data/processed/physionet_lr_ea_*",
            "data/processed/physionet_lr_no_ea_*",
        ]
    else:
        patterns = [
            "data/processed/bnci*_ea_*",
            "data/processed/bnci*_no_ea_*",
        ]
    markers: list[Path] = []
    for pat in patterns:
        for d in ROOT.glob(pat):
            if (d / "subject_dict_raw.joblib").exists() or (d / "meta.json").exists():
                markers.append(d)
    return markers


def verify_or_prepare_data(
    datasets: list[str],
    *,
    logger: ExecutionLogger,
    dry_run: bool,
    failed_csv: Path,
    force_prepare: bool,
) -> None:
    logger.section("STAGE 2: Verify or prepare cached datasets")
    config_map = {
        "physionet": ["preprocess_ea.yaml", "preprocess_no_ea.yaml"],
        "bnci": ["preprocess_bnci_ea.yaml", "preprocess_bnci_no_ea.yaml"],
    }

    for ds in datasets:
        markers = _cache_marker_paths(ds)
        if markers and not force_prepare:
            logger.write(f"  [{ds}] cache found ({len(markers)} dirs); skipping prepare_data")
            for m in markers[:3]:
                logger.write(f"    - {m.relative_to(ROOT)}")
            continue

        logger.write(f"  [{ds}] cache missing or --force-prepare; running prepare_data")
        for cfg in config_map.get(ds, []):
            run_cmd(
                [PY, "scripts/prepare_data.py", "--config", f"configs/{cfg}"],
                logger=logger,
                stage="verify_data",
                dataset=ds,
                dry_run=dry_run,
                failed_csv=failed_csv,
            )


def _aggregate_riemannian_from_holdout(
    holdout_csv: Path,
    riemann_csv: Path,
    logger: ExecutionLogger,
) -> bool:
    """Copy Riemannian rows from repeated hold-out results if present."""
    if not holdout_csv.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(holdout_csv)
        if df.empty or "model" not in df.columns:
            return False
        riemann_df = df[df["model"].isin(RIEMANN_MODELS)]
        if riemann_df.empty:
            return False
        riemann_csv.parent.mkdir(parents=True, exist_ok=True)
        riemann_df.to_csv(riemann_csv, index=False)
        logger.write(f"Aggregated {len(riemann_df)} Riemannian rows -> {riemann_csv}")
        return True
    except Exception as exc:
        logger.write(f"Could not aggregate Riemannian results: {exc}")
        return False


def generate_implementation_summary(output_root: Path, logger: ExecutionLogger) -> None:
    logger.section("STAGE 14: Generate implementation summary")
    out = output_root / "reports" / "implementation_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Implementation Summary\n",
        f"Generated: {_now_iso()}\n",
        "## Output root\n",
        f"`{output_root}`\n",
        "## Completed artifacts\n",
    ]

    checks = [
        ("Repeated hold-out", "repeated_holdout/*/repeated_holdout_results.csv"),
        ("GroupKFold", "groupkfold/*/groupkfold_results.csv"),
        ("Riemannian", "riemannian/*/riemannian_results.csv"),
        ("Statistics", "stats/*/bootstrap_ci.csv"),
        ("Subject-level", "subject_level/*/subject_level_metrics.csv"),
        ("Neurophysiology", "neurophysiology/*/erd_ers_trial_level.csv"),
        ("EA diagnostics", "ea_diagnostics/*/ea_covariance_distances.csv"),
        ("Paper tables", "paper_tables/*.tex"),
        ("Paper figures", "paper_figures/*.pdf"),
        ("Reproducibility", "reports/reproducibility_report.md"),
        ("Failed runs", "failed_runs/failed_runs.csv"),
    ]
    for label, pattern in checks:
        matches = list(output_root.glob(pattern))
        status = f"{len(matches)} file(s)" if matches else "not found"
        lines.append(f"- **{label}**: {status}\n")

    lines.append("\n## Resume command\n")
    lines.append("```bash\n")
    lines.append("./run_paperspace_publishable_experiments.sh\n")
    lines.append("```\n")

    out.write_text("".join(lines), encoding="utf-8")
    logger.write(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run publishable EEG-MI experiments in resumable stages (Paperspace-ready)."
    )
    p.add_argument(
        "--stage",
        default="all",
        choices=STAGE_CHOICES,
        help="Pipeline stage (all runs every stage in order).",
    )
    p.add_argument("--datasets", nargs="+", default=["physionet", "bnci"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["fbcsp_lda", "csp_svm", "riemann_mdm", "riemann_ts_lr", "eegnet", "eegme"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--ea", default="both", help="EA mode for training scripts: true|false|both")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--skip-deep", action="store_true", help="Exclude deep models where supported")
    p.add_argument("--skip-existing", action="store_true", help="Skip runs whose outputs already exist")
    p.add_argument("--only-classical", action="store_true", help="Classical models only")
    p.add_argument("--groupkfold-deep", action="store_true", help="Include deep models in GroupKFold")
    p.add_argument("--force-prepare", action="store_true", help="Always run prepare_data")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs_publishable",
        help="Root directory for all publishable outputs",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Execution log path (default: <output-root>/reports/paperspace_execution_log.txt)",
    )
    args = p.parse_args()

    output_root = args.output_root.resolve()
    log_path = (args.log_file or output_root / "reports" / "paperspace_execution_log.txt").resolve()
    failed_csv = output_root / "failed_runs" / "failed_runs.csv"
    logger = ExecutionLogger(log_path)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "failed_runs").mkdir(parents=True, exist_ok=True)

    ea = _ea_flag(args.ea)
    models = _resolve_models(args.models, only_classical=args.only_classical, skip_deep=args.skip_deep)

    if args.stage == "all":
        stages = STAGE_ORDER
    elif args.stage == "paper":
        stages = [
            "paper_tables",
            "paper_figures",
            "reproducibility_report",
            "paper_text_snippets",
            "implementation_summary",
        ]
    elif args.stage == "data":
        stages = ["verify_data"]
    else:
        stages = [args.stage]

    logger.section("PUBLISHABLE EEG-MI PIPELINE")
    logger.write(f"Started: {_now_iso()}")
    logger.write(f"Stage(s): {', '.join(stages)}")
    logger.write(f"Output root: {output_root}")
    logger.write(f"Datasets: {args.datasets}")
    logger.write(f"Models: {models}")
    logger.write(f"Seeds: {args.seeds}")
    logger.write(f"EA: {ea}")
    logger.write(f"Skip existing: {args.skip_existing}")

    if "verify_env" in stages:
        verify_environment(logger)

    if "verify_data" in stages:
        verify_or_prepare_data(
            args.datasets,
            logger=logger,
            dry_run=args.dry_run,
            failed_csv=failed_csv,
            force_prepare=args.force_prepare,
        )

    for ds in args.datasets:
        if "repeated_holdout" in stages:
            logger.section(f"STAGE 3: Repeated hold-out — {ds}")
            out = output_root / "repeated_holdout" / ds
            cmd = [
                PY,
                "scripts/run_repeated_holdout.py",
                "--dataset",
                ds,
                "--seeds",
                *[str(s) for s in args.seeds],
                "--models",
                *models,
                "--ea",
                ea,
                "--output-dir",
                str(out),
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            if args.only_classical:
                cmd.append("--only-classical")
            run_cmd(
                cmd,
                logger=logger,
                stage="repeated_holdout",
                dataset=ds,
                dry_run=args.dry_run,
                failed_csv=failed_csv,
            )

        if "groupkfold" in stages:
            logger.section(f"STAGE 4: GroupKFold — {ds}")
            gk_models = models
            if not args.groupkfold_deep:
                gk_models = [m for m in gk_models if m not in DEEP_MODELS]
            n_splits = args.n_splits if ds == "physionet" else min(args.n_splits, 3)
            out = output_root / "groupkfold" / ds
            cmd = [
                PY,
                "scripts/run_groupkfold.py",
                "--dataset",
                ds,
                "--n-splits",
                str(n_splits),
                "--models",
                *gk_models,
                "--ea",
                ea,
                "--output-dir",
                str(out),
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            if not args.groupkfold_deep:
                cmd.append("--skip-deep")
            run_cmd(
                cmd,
                logger=logger,
                stage="groupkfold",
                dataset=ds,
                dry_run=args.dry_run,
                failed_csv=failed_csv,
            )

        if "riemannian" in stages:
            logger.section(f"STAGE 5: Riemannian baselines — {ds}")
            riemann_out = output_root / "riemannian" / ds
            riemann_csv = riemann_out / "riemannian_results.csv"
            holdout_csv = output_root / "repeated_holdout" / ds / "repeated_holdout_results.csv"

            if args.skip_existing and riemann_csv.exists():
                logger.write(f"Skipping Riemannian stage; exists: {riemann_csv}")
            elif _aggregate_riemannian_from_holdout(holdout_csv, riemann_csv, logger):
                pass
            else:
                cmd = [
                    PY,
                    "scripts/run_riemannian_baselines.py",
                    "--dataset",
                    ds,
                    "--ea",
                    ea,
                    "--seeds",
                    *[str(s) for s in args.seeds],
                    "--output-dir",
                    str(riemann_out),
                ]
                if args.skip_existing:
                    cmd.append("--skip-existing")
                run_cmd(
                    cmd,
                    logger=logger,
                    stage="riemannian",
                    dataset=ds,
                    dry_run=args.dry_run,
                    failed_csv=failed_csv,
                )
                if holdout_csv.exists() and not riemann_csv.exists():
                    _aggregate_riemannian_from_holdout(holdout_csv, riemann_csv, logger)

        if "stats" in stages:
            logger.section(f"STAGE 6: Statistical analysis — {ds}")
            res = output_root / "repeated_holdout" / ds / "repeated_holdout_results.csv"
            if res.exists() or args.dry_run:
                run_cmd(
                    [
                        PY,
                        "scripts/run_statistical_analysis.py",
                        "--results",
                        str(res),
                        "--output-dir",
                        str(output_root / "stats" / ds),
                    ],
                    logger=logger,
                    stage="stats",
                    dataset=ds,
                    dry_run=args.dry_run,
                    failed_csv=failed_csv,
                )
            else:
                logger.write(f"Skipping stats for {ds}: missing {res}")

        if "subject_level" in stages:
            logger.section(f"STAGE 7: Subject-level analysis — {ds}")
            pred_root = output_root / "repeated_holdout" / ds
            if pred_root.exists() or args.dry_run:
                run_cmd(
                    [
                        PY,
                        "scripts/run_subject_level_analysis.py",
                        "--predictions-root",
                        str(pred_root),
                        "--output-dir",
                        str(output_root / "subject_level" / ds),
                    ],
                    logger=logger,
                    stage="subject_level",
                    dataset=ds,
                    dry_run=args.dry_run,
                    failed_csv=failed_csv,
                )
            else:
                logger.write(f"Skipping subject-level for {ds}: missing {pred_root}")

        if "neurophysiology" in stages:
            logger.section(f"STAGE 8: Neurophysiology — {ds}")
            run_cmd(
                [
                    PY,
                    "scripts/run_neurophysiology_analysis.py",
                    "--dataset",
                    ds,
                    "--ea",
                    ea,
                    "--output-dir",
                    str(output_root / "neurophysiology" / ds),
                ],
                logger=logger,
                stage="neurophysiology",
                dataset=ds,
                dry_run=args.dry_run,
                failed_csv=failed_csv,
            )

        if "ea_diagnostics" in stages:
            logger.section(f"STAGE 9: EA diagnostics — {ds}")
            run_cmd(
                [
                    PY,
                    "scripts/run_ea_diagnostics.py",
                    "--dataset",
                    ds,
                    "--output-dir",
                    str(output_root / "ea_diagnostics" / ds),
                ],
                logger=logger,
                stage="ea_diagnostics",
                dataset=ds,
                dry_run=args.dry_run,
                failed_csv=failed_csv,
            )

    if "paper_tables" in stages:
        logger.section("STAGE 10: Generate paper tables")
        run_cmd(
            [
                PY,
                "scripts/generate_paper_tables.py",
                "--results-root",
                str(output_root),
                "--output-dir",
                str(output_root / "paper_tables"),
            ],
            logger=logger,
            stage="paper_tables",
            dataset="all",
            dry_run=args.dry_run,
            failed_csv=failed_csv,
        )

    if "paper_figures" in stages:
        logger.section("STAGE 11: Generate paper figures")
        run_cmd(
            [
                PY,
                "scripts/generate_paper_figures.py",
                "--results-root",
                str(output_root),
                "--output-dir",
                str(output_root / "paper_figures"),
            ],
            logger=logger,
            stage="paper_figures",
            dataset="all",
            dry_run=args.dry_run,
            failed_csv=failed_csv,
        )

    if "reproducibility_report" in stages:
        logger.section("STAGE 12: Generate reproducibility report")
        run_cmd(
            [
                PY,
                "scripts/generate_reproducibility_report.py",
                "--results-root",
                str(output_root),
                "--output",
                str(output_root / "reports" / "reproducibility_report.md"),
            ],
            logger=logger,
            stage="reproducibility_report",
            dataset="all",
            dry_run=args.dry_run,
            failed_csv=failed_csv,
        )

    if "paper_text_snippets" in stages:
        logger.section("STAGE 13: Generate paper text snippets")
        run_cmd(
            [
                PY,
                "scripts/generate_paper_text_snippets.py",
                "--results-root",
                str(output_root),
                "--output",
                str(output_root / "paper_tables" / "generated_result_sentences.md"),
            ],
            logger=logger,
            stage="paper_text_snippets",
            dataset="all",
            dry_run=args.dry_run,
            failed_csv=failed_csv,
        )

    if "implementation_summary" in stages:
        if not args.dry_run:
            generate_implementation_summary(output_root, logger)
        else:
            logger.write("DRY-RUN: would write implementation_summary.md")

    logger.section("PIPELINE FINISHED")
    logger.write(f"Finished: {_now_iso()}")
    logger.write(f"Log: {log_path}")
    logger.write(f"Failed runs (if any): {failed_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
