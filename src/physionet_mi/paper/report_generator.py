"""Reproducibility report generation."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import yaml


def collect_package_versions() -> dict[str, str]:
    pkgs = ["numpy", "scipy", "sklearn", "pandas", "mne", "moabb", "matplotlib", "torch", "pyriemann"]
    versions = {}
    for pkg in pkgs:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def generate_reproducibility_report(results_root: Path, output_path: Path, project_root: Path) -> None:
    lines = [
        "# Reproducibility Report\n",
        f"- Python: {sys.version.split()[0]}\n",
        f"- OS: {platform.platform()}\n",
        "## Package versions\n",
    ]
    for pkg, ver in collect_package_versions().items():
        lines.append(f"- {pkg}: {ver}\n")

    lines.append("\n## Dataset sources\n")
    lines.append("- PhysioNet MI via MOABB `PhysionetMI`\n")
    lines.append("- BNCI2014-001 via MOABB `BNCI2014_001`\n")

    lines.append("\n## Evaluation protocol\n")
    lines.append("- Binary left_hand vs right_hand\n")
    lines.append("- Subject-disjoint hold-out (repeated seeds) and GroupKFold\n")
    lines.append("- EA ablation on/off with per-split preprocessing\n")
    lines.append("- No subject leakage between train/val/test\n")

    cfg_default = project_root / "configs/default.yaml"
    if cfg_default.exists():
        lines.append("\n## Default hyperparameters (`configs/default.yaml`)\n")
        lines.append("```yaml\n")
        lines.append(cfg_default.read_text(encoding="utf-8")[:2000])
        lines.append("\n```\n")

    lines.append("\n## Commands to reproduce\n")
    cmds = [
        "python scripts/prepare_data.py --config configs/preprocess_ea.yaml",
        "python scripts/run_repeated_holdout.py --dataset physionet --seeds 0 1 2 3 4 5 6 7 8 9 --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr --ea both",
        "python scripts/run_statistical_analysis.py --results artifacts/runs/publishable/repeated_holdout/physionet/repeated_holdout_results.csv --output-dir artifacts/runs/publishable/stats/physionet",
        "python scripts/generate_paper_tables.py",
        "python scripts/generate_paper_figures.py",
    ]
    for c in cmds:
        lines.append(f"```bash\n{c}\n```\n")

    lines.append("\n## Output paths\n")
    for p in sorted(results_root.rglob("*.csv"))[:30]:
        lines.append(f"- `{p.relative_to(project_root) if p.is_relative_to(project_root) else p}`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
