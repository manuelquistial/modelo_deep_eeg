"""Reproducibility report generation."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd


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


def _load_repeat_seed_tables(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("**/repeat_seeds.json"))


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
    lines.append("- Repeated subject-disjoint hold-out and GroupKFold\n")
    lines.append("- EA ablation on/off with per-split preprocessing\n")
    lines.append("- No subject leakage between train/val/test\n")

    lines.append("\n## Repeated hold-out randomization\n")
    lines.append(
        "- Strategy: `numpy.random.default_rng(master_seed)` draws `uint32` split/model seeds.\n"
    )
    lines.append(
        "- Default: `master_seed=42`, `n_repeats=10` (not a cherry-picked best split).\n"
    )
    lines.append(
        "- Each repetition stores `repeat_id`, `split_seed`, and `model_seed` in "
        "`repeat_seeds.csv` / `split_metadata.json`.\n"
    )

    seed_files = _load_repeat_seed_tables(results_root)
    if seed_files:
        for sf in seed_files:
            rel = sf.relative_to(project_root) if sf.is_relative_to(project_root) else sf
            lines.append(f"\n### Generated seeds: `{rel}`\n")
            try:
                records = json.loads(sf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                lines.append("- (could not parse JSON)\n")
                continue
            if not records:
                continue
            master = records[0].get("master_seed", "n/a")
            n_rep = records[0].get("n_repeats", len(records))
            lines.append(f"- master_seed: {master}\n")
            lines.append(f"- n_repeats: {n_rep}\n")
            split_seeds = [r.get("split_seed") for r in records]
            model_seeds = [r.get("model_seed") for r in records]
            lines.append(f"- split_seeds: {split_seeds}\n")
            lines.append(f"- model_seeds: {model_seeds}\n")
    else:
        lines.append("- No `repeat_seeds.json` found yet; run repeated hold-out first.\n")

    results_csvs = sorted(results_root.glob("**/repeated_holdout_results.csv"))
    if results_csvs:
        lines.append("\n## Repeated hold-out result metadata\n")
        for rc in results_csvs[:2]:
            try:
                df = pd.read_csv(rc)
            except Exception:
                continue
            cols = [c for c in ("master_seed", "n_repeats", "repeat_id", "split_seed", "model_seed") if c in df.columns]
            if cols:
                rel = rc.relative_to(project_root) if rc.is_relative_to(project_root) else rc
                lines.append(f"- `{rel}` columns: {', '.join(cols)}\n")

    cfg_default = project_root / "configs/default.yaml"
    if cfg_default.exists():
        lines.append("\n## Default hyperparameters (`configs/default.yaml`)\n")
        lines.append("```yaml\n")
        lines.append(cfg_default.read_text(encoding="utf-8")[:2000])
        lines.append("\n```\n")

    lines.append("\n## Commands to reproduce\n")
    cmds = [
        "python scripts/prepare_data.py --config configs/preprocess_ea.yaml",
        (
            "python scripts/run_repeated_holdout.py --dataset physionet "
            "--master-seed 42 --n-repeats 10 "
            "--models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr --ea both "
            "--output-dir artifacts/runs/publishable/repeated_holdout/physionet"
        ),
        (
            "python scripts/run_statistical_analysis.py "
            "--results artifacts/runs/publishable/repeated_holdout/physionet/repeated_holdout_results.csv "
            "--output-dir artifacts/runs/publishable/stats/physionet"
        ),
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
