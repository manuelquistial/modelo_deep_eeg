"""Central artifact paths for generated data, caches, and experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACTS_ENV = "PHYSIONET_MI_ARTIFACTS_ROOT"

# Relative paths inside the artifacts root (canonical layout).
REL_RAW = "raw"
REL_CACHE = "cache"
REL_BASELINE_RUNS = "runs/baseline"
REL_PUBLISHABLE_RUNS = "runs/publishable"
REL_PAPER_TABLES = "paper/tables"
REL_PAPER_FIGURES = "paper/figures"
REL_REPORTS = "reports"
REL_FAILED_RUNS = "failed_runs"
REL_SMOKE = "smoke"

# Legacy locations kept for backward compatibility.
LEGACY_RAW = "data/mne"
LEGACY_CACHE = "data/processed"
LEGACY_BASELINE_RUNS = "outputs"
LEGACY_PUBLISHABLE_RUNS = "outputs_publishable"
LEGACY_SMOKE = "outputs_publishable_smoke_test"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* until a directory containing ``pyproject.toml`` is found."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


def artifacts_root(project_root: Path | None = None) -> Path:
    """Return the root folder for all generated artifacts."""
    root = project_root or find_project_root()
    env = os.environ.get(ARTIFACTS_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (root / "artifacts").resolve()


def _pick_existing(project_root: Path, new_rel: str, legacy_rel: str) -> Path:
    """Prefer the canonical artifacts path; fall back to legacy if only that exists."""
    new_path = project_root / "artifacts" / new_rel
    legacy_path = project_root / legacy_rel
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path


def raw_data_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return _pick_existing(root, REL_RAW, LEGACY_RAW)


def cache_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return _pick_existing(root, REL_CACHE, LEGACY_CACHE)


def baseline_runs_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return _pick_existing(root, REL_BASELINE_RUNS, LEGACY_BASELINE_RUNS)


def publishable_runs_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    return _pick_existing(root, REL_PUBLISHABLE_RUNS, LEGACY_PUBLISHABLE_RUNS)


def paper_tables_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    canonical = artifacts_root(root) / REL_PAPER_TABLES
    legacy = publishable_runs_dir(root) / "paper_tables"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def paper_figures_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    canonical = artifacts_root(root) / REL_PAPER_FIGURES
    legacy = publishable_runs_dir(root) / "paper_figures"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def reports_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    canonical = artifacts_root(root) / REL_REPORTS
    legacy = publishable_runs_dir(root) / "reports"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def failed_runs_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    canonical = artifacts_root(root) / REL_FAILED_RUNS
    legacy = publishable_runs_dir(root) / "failed_runs"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def smoke_runs_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    canonical = artifacts_root(root) / REL_SMOKE
    legacy = root / LEGACY_SMOKE
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def default_config_paths(project_root: Path | None = None) -> dict[str, str]:
    """Relative config strings for YAML defaults."""
    root = project_root or find_project_root()
    art = artifacts_root(root)
    try:
        raw_rel = raw_data_dir(root).relative_to(root).as_posix()
    except ValueError:
        raw_rel = f"{art.name}/{REL_RAW}"
    try:
        cache_rel = cache_dir(root).relative_to(root).as_posix()
    except ValueError:
        cache_rel = f"{art.name}/{REL_CACHE}"
    try:
        baseline_rel = baseline_runs_dir(root).relative_to(root).as_posix()
    except ValueError:
        baseline_rel = f"{art.name}/{REL_BASELINE_RUNS}"
    return {
        "mne_data_dir": raw_rel,
        "cache_dir": cache_rel,
        "outputs_dir": baseline_rel,
    }


def resolve_output_layout(output_root: Path) -> dict[str, Path]:
    """Map an output root to publishable/reports/paper subdirectories.

    Supports the canonical ``artifacts/`` layout and the legacy flat
    ``outputs_publishable/`` layout.
    """
    output_root = output_root.resolve()
    canonical_publishable = output_root / REL_PUBLISHABLE_RUNS
    if output_root.name == "artifacts" or canonical_publishable.exists():
        base = output_root if output_root.name == "artifacts" else output_root
        return {
            "root": base,
            "publishable": base / REL_PUBLISHABLE_RUNS,
            "reports": base / REL_REPORTS,
            "paper_tables": base / REL_PAPER_TABLES,
            "paper_figures": base / REL_PAPER_FIGURES,
            "failed_runs": base / REL_FAILED_RUNS,
            "results_root": base / REL_PUBLISHABLE_RUNS,
        }
    return {
        "root": output_root,
        "publishable": output_root,
        "reports": output_root / "reports",
        "paper_tables": output_root / "paper_tables",
        "paper_figures": output_root / "paper_figures",
        "failed_runs": output_root / "failed_runs",
        "results_root": output_root,
    }


def ensure_artifact_tree(project_root: Path | None = None) -> list[Path]:
    """Create the canonical artifacts directory tree."""
    root = project_root or find_project_root()
    art = artifacts_root(root)
    dirs = [
        art / REL_RAW,
        art / REL_CACHE,
        art / REL_BASELINE_RUNS,
        art / REL_PUBLISHABLE_RUNS,
        art / REL_PAPER_TABLES,
        art / REL_PAPER_FIGURES,
        art / REL_REPORTS,
        art / REL_FAILED_RUNS,
        art / REL_SMOKE,
        art / REL_PUBLISHABLE_RUNS / "repeated_holdout",
        art / REL_PUBLISHABLE_RUNS / "groupkfold",
        art / REL_PUBLISHABLE_RUNS / "stats",
        art / REL_PUBLISHABLE_RUNS / "subject_level",
        art / REL_PUBLISHABLE_RUNS / "neurophysiology",
        art / REL_PUBLISHABLE_RUNS / "ea_diagnostics",
        art / REL_PUBLISHABLE_RUNS / "riemannian",
    ]
    created: list[Path] = []
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created
