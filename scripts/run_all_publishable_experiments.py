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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable

from physionet_mi.evaluation.random_seeds import resolve_repeat_seeds  # noqa: E402
from physionet_mi.paths import (  # noqa: E402
    artifacts_root,
    cache_dir,
    ensure_artifact_tree,
    resolve_output_layout,
)

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
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
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

    assert proc.stdout is not None
    output_lines: list[str] = []
    for line in proc.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)

    returncode = proc.wait()
    if output_lines:
        logger.write("".join(output_lines).rstrip())

    if returncode != 0:
        err = "".join(output_lines).strip()[:2000] or "non-zero exit"
        _append_failed_run(
            failed_csv,
            stage=stage,
            dataset=dataset,
            command=cmd_str,
            returncode=returncode,
            error_message=err,
        )
        logger.write(f"FAILED (exit {returncode}): {cmd_str}")
        print(f"\n!!! COMMAND FAILED (exit {returncode}): {cmd_str}\n", flush=True)
        if not continue_on_error:
            sys.exit(returncode)
    else:
        logger.write(f"OK: {cmd_str}")
    return returncode


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
    cache_root = cache_dir(ROOT)
    if ds == "physionet":
        patterns = [
            "physionet_lr_ea_*",
            "physionet_lr_no_ea_*",
        ]
    else:
        patterns = [
            "bnci*_ea_*",
            "bnci*_no_ea_*",
        ]
    markers: list[Path] = []
    for pat in patterns:
        for d in cache_root.glob(pat):
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


EXECUTION_COMPLETE_NAME = "EXECUTION_COMPLETE.txt"
EXECUTION_FAILED_NAME = "EXECUTION_FAILED.txt"


@dataclass
class PipelineContext:
    stage: str
    stages: list[str]
    datasets: list[str]
    models: list[str]
    master_seed: int | None
    n_repeats: int
    legacy_explicit_seeds: bool
    explicit_seeds: list[int] | None
    ea: str
    skip_existing: bool
    dry_run: bool
    layout: dict[str, Path]
    started_at: str
    started_monotonic: float = field(default_factory=monotonic)
    log_path: Path | None = None


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _count_failed_runs(failed_csv: Path) -> int:
    if not failed_csv.exists():
        return 0
    try:
        with failed_csv.open(encoding="utf-8") as fh:
            rows = sum(1 for _ in csv.DictReader(fh))
        return rows
    except Exception:
        return -1


def write_execution_marker(
    *,
    status: str,
    ctx: PipelineContext,
    failed_csv: Path,
    error_message: str = "",
) -> Path:
    """Write a plain-text marker so Paperspace runs can be verified after shutdown."""
    reports = ctx.layout["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    finished_at = _now_iso()
    duration = _format_duration(monotonic() - ctx.started_monotonic)
    failed_count = _count_failed_runs(failed_csv)

    if status == "COMPLETE":
        filename = EXECUTION_COMPLETE_NAME
        headline = "PUBLISHABLE EEG-MI PIPELINE — EXECUTION COMPLETE"
        footer = "Safe to shut down the Paperspace machine."
    else:
        filename = EXECUTION_FAILED_NAME
        headline = "PUBLISHABLE EEG-MI PIPELINE — EXECUTION FAILED"
        footer = "Resume with: ./run_paperspace_publishable_experiments.sh"

    marker = reports / filename
    lines = [
        headline,
        "=" * len(headline),
        f"Status: {status}",
        f"Started (UTC): {ctx.started_at}",
        f"Finished (UTC): {finished_at}",
        f"Duration: {duration}",
        f"Stage requested: {ctx.stage}",
        f"Stages executed: {', '.join(ctx.stages)}",
        f"Artifacts root: {ctx.layout['root']}",
        f"Datasets: {', '.join(ctx.datasets)}",
        f"Models: {', '.join(ctx.models)}",
        (
            f"Seeds (legacy): {', '.join(str(s) for s in ctx.explicit_seeds)}"
            if ctx.legacy_explicit_seeds and ctx.explicit_seeds
            else f"Master seed: {ctx.master_seed}, n_repeats: {ctx.n_repeats}"
        ),
        f"EA mode: {ctx.ea}",
        f"Skip existing: {ctx.skip_existing}",
        f"Dry run: {ctx.dry_run}",
        f"Failed subprocesses logged: {failed_count}",
        f"Failed runs CSV: {failed_csv}",
        f"Execution log: {ctx.log_path}",
        f"Implementation summary: {reports / 'implementation_summary.md'}",
        f"Reproducibility report: {reports / 'reproducibility_report.md'}",
    ]
    if error_message:
        lines.append(f"Error: {error_message}")
    lines.append("")
    lines.append(footer)
    lines.append("")
    marker.write_text("\n".join(lines), encoding="utf-8")
    return marker


def generate_implementation_summary(layout: dict[str, Path], logger: ExecutionLogger) -> None:
    logger.section("STAGE 14: Generate implementation summary")
    out = layout["reports"] / "implementation_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    art_root = layout["root"]

    lines = [
        "# Implementation Summary\n",
        f"Generated: {_now_iso()}\n",
        "## Artifacts root\n",
        f"`{art_root}`\n",
        "## Completed artifacts\n",
    ]

    checks = [
        ("Repeated hold-out", layout["publishable"] / "repeated_holdout"),
        ("GroupKFold", layout["publishable"] / "groupkfold"),
        ("Riemannian", layout["publishable"] / "riemannian"),
        ("Statistics", layout["publishable"] / "stats"),
        ("Subject-level", layout["publishable"] / "subject_level"),
        ("Neurophysiology", layout["publishable"] / "neurophysiology"),
        ("EA diagnostics", layout["publishable"] / "ea_diagnostics"),
        ("Paper tables", layout["paper_tables"]),
        ("Paper figures", layout["paper_figures"]),
        ("Reproducibility", layout["reports"] / "reproducibility_report.md"),
        ("Failed runs", layout["failed_runs"] / "failed_runs.csv"),
    ]
    for label, path in checks:
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = list(path.rglob("*"))
            matches = [m for m in matches if m.is_file()]
        else:
            matches = []
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
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Legacy explicit split seeds. Prefer --master-seed and --n-repeats.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="Master seed for repeated hold-out (default: 42)",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeated hold-out partitions (default: 10)",
    )
    p.add_argument(
        "--separate-model-seeds",
        action="store_true",
        help="Draw independent model seeds from the master RNG",
    )
    p.add_argument("--ea", default="both", help="EA mode for training scripts: true|false|both")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--skip-deep", action="store_true", help="Exclude deep models where supported")
    p.add_argument("--skip-existing", action="store_true", help="Skip runs whose outputs already exist")
    p.add_argument("--only-classical", action="store_true", help="Classical models only")
    p.add_argument("--groupkfold-deep", action="store_true", help="Include deep models in GroupKFold")
    p.add_argument("--force-prepare", action="store_true", help="Always run prepare_data")
    p.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="Parallel CPU workers for classical models per seed/fold (deep models stay on GPU sequentially)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Artifacts root (default: artifacts/). Legacy outputs_publishable/ still supported.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Execution log path (default: <output-root>/reports/paperspace_execution_log.txt)",
    )
    args = p.parse_args()

    try:
        _, master_seed, n_repeats, legacy_seeds = resolve_repeat_seeds(
            explicit_seeds=args.seeds,
            master_seed=args.master_seed,
            n_repeats=args.n_repeats,
            separate_model_seeds=args.separate_model_seeds,
        )
    except ValueError as exc:
        p.error(str(exc))

    output_root = (args.output_root or artifacts_root(ROOT)).resolve()
    layout = resolve_output_layout(output_root)
    log_path = (args.log_file or layout["reports"] / "paperspace_execution_log.txt").resolve()
    failed_csv = layout["failed_runs"] / "failed_runs.csv"
    logger = ExecutionLogger(log_path)

    ensure_artifact_tree(ROOT)
    for key in ("reports", "failed_runs", "publishable", "paper_tables", "paper_figures"):
        layout[key].mkdir(parents=True, exist_ok=True)

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

    started_at = _now_iso()
    ctx = PipelineContext(
        stage=args.stage,
        stages=stages,
        datasets=args.datasets,
        models=models,
        master_seed=master_seed,
        n_repeats=n_repeats,
        legacy_explicit_seeds=legacy_seeds,
        explicit_seeds=args.seeds,
        ea=ea,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        layout=layout,
        started_at=started_at,
        log_path=log_path,
    )

    logger.section("PUBLISHABLE EEG-MI PIPELINE")
    logger.write(f"Started: {started_at}")
    logger.write(f"Stage(s): {', '.join(stages)}")
    logger.write(f"Output root: {output_root}")
    logger.write(f"Datasets: {args.datasets}")
    logger.write(f"Models: {models}")
    if legacy_seeds:
        logger.write(f"Seeds (legacy --seeds): {args.seeds}")
        logger.write(
            "WARNING: Using explicit --seeds. For manuscript experiments, prefer --master-seed and --n-repeats."
        )
    else:
        logger.write(f"Master seed: {master_seed}, n_repeats: {n_repeats}")
    logger.write(f"EA: {ea}")
    logger.write(f"Skip existing: {args.skip_existing}")
    logger.write(f"Parallel jobs (classical): {args.parallel_jobs}")

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
            out = layout["publishable"] / "repeated_holdout" / ds
            cmd = [
                PY,
                "scripts/run_repeated_holdout.py",
                "--dataset",
                ds,
                "--models",
                *models,
                "--ea",
                ea,
                "--output-dir",
                str(out),
            ]
            if legacy_seeds:
                cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
            else:
                cmd.extend(["--master-seed", str(master_seed), "--n-repeats", str(n_repeats)])
            if args.separate_model_seeds:
                cmd.append("--separate-model-seeds")
            if args.skip_existing:
                cmd.append("--skip-existing")
            if args.only_classical:
                cmd.append("--only-classical")
            if args.parallel_jobs > 1:
                cmd.extend(["--parallel-jobs", str(args.parallel_jobs)])
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
            out = layout["publishable"] / "groupkfold" / ds
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
            if master_seed is not None:
                cmd.extend(["--master-seed", str(master_seed)])
            if args.skip_existing:
                cmd.append("--skip-existing")
            if not args.groupkfold_deep:
                cmd.append("--skip-deep")
            if args.parallel_jobs > 1:
                cmd.extend(["--parallel-jobs", str(args.parallel_jobs)])
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
            riemann_out = layout["publishable"] / "riemannian" / ds
            riemann_csv = riemann_out / "riemannian_results.csv"
            holdout_csv = layout["publishable"] / "repeated_holdout" / ds / "repeated_holdout_results.csv"

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
                    "--output-dir",
                    str(riemann_out),
                ]
                if legacy_seeds:
                    cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
                else:
                    cmd.extend(["--master-seed", str(master_seed), "--n-repeats", str(n_repeats)])
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
            res = layout["publishable"] / "repeated_holdout" / ds / "repeated_holdout_results.csv"
            if res.exists() or args.dry_run:
                run_cmd(
                    [
                        PY,
                        "scripts/run_statistical_analysis.py",
                        "--results",
                        str(res),
                        "--output-dir",
                        str(layout["publishable"] / "stats" / ds),
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
            pred_root = layout["publishable"] / "repeated_holdout" / ds
            if pred_root.exists() or args.dry_run:
                run_cmd(
                    [
                        PY,
                        "scripts/run_subject_level_analysis.py",
                        "--predictions-root",
                        str(pred_root),
                        "--output-dir",
                        str(layout["publishable"] / "subject_level" / ds),
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
                    str(layout["publishable"] / "neurophysiology" / ds),
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
                    str(layout["publishable"] / "ea_diagnostics" / ds),
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
                str(layout["results_root"]),
                "--output-dir",
                str(layout["paper_tables"]),
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
                str(layout["results_root"]),
                "--output-dir",
                str(layout["paper_figures"]),
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
                str(layout["results_root"]),
                "--output",
                str(layout["reports"] / "reproducibility_report.md"),
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
                str(layout["results_root"]),
                "--output",
                str(layout["paper_tables"] / "generated_result_sentences.md"),
            ],
            logger=logger,
            stage="paper_text_snippets",
            dataset="all",
            dry_run=args.dry_run,
            failed_csv=failed_csv,
        )

    if "implementation_summary" in stages:
        if not args.dry_run:
            generate_implementation_summary(layout, logger)
        else:
            logger.write("DRY-RUN: would write implementation_summary.md")

    logger.section("PIPELINE FINISHED")
    logger.write(f"Finished: {_now_iso()}")
    logger.write(f"Log: {log_path}")
    logger.write(f"Failed runs (if any): {failed_csv}")

    if not args.dry_run:
        marker = write_execution_marker(
            status="COMPLETE",
            ctx=ctx,
            failed_csv=failed_csv,
        )
        logger.write(f"Execution marker: {marker}")
        print(f"\n>>> PIPELINE COMPLETE — see {marker}\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        try:
            # Best-effort failure marker if argparse/main partially initialized.
            reports = ROOT / "artifacts" / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            marker = reports / EXECUTION_FAILED_NAME
            marker.write_text(
                "\n".join(
                    [
                        "PUBLISHABLE EEG-MI PIPELINE — EXECUTION FAILED",
                        f"Finished (UTC): {_now_iso()}",
                        f"Error: {exc}",
                        "",
                        "Resume with: ./run_paperspace_publishable_experiments.sh",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"\n>>> PIPELINE FAILED — see {marker}\n", flush=True)
        except Exception:
            pass
        sys.exit(1)
