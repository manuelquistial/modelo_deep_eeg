"""Map dataset × preprocess variant → YAML config path."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_DATASETS = ("physionet", "bnci2014_001")
SUPPORTED_PREPROCESS = ("ea", "no_ea")

CONFIG_MATRIX: dict[str, dict[str, str]] = {
    "physionet": {
        "ea": "configs/preprocess_ea.yaml",
        "no_ea": "configs/preprocess_no_ea.yaml",
    },
    "bnci2014_001": {
        "ea": "configs/preprocess_bnci_ea.yaml",
        "no_ea": "configs/preprocess_bnci_no_ea.yaml",
    },
}


def resolve_config_path(
    dataset: str,
    preprocess: str,
    project_root: Path | None = None,
) -> Path:
    """Return absolute path to experiment config for dataset + preprocess variant."""
    if dataset not in CONFIG_MATRIX:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from {list(CONFIG_MATRIX)}"
        )
    if preprocess not in SUPPORTED_PREPROCESS:
        raise ValueError(
            f"Unknown preprocess {preprocess!r}. Choose from {SUPPORTED_PREPROCESS}"
        )
    rel = CONFIG_MATRIX[dataset][preprocess]
    root = project_root or Path.cwd()
    return (root / rel).resolve()


def iter_benchmark_configs(
    datasets: list[str],
    preprocess_variants: list[str],
    project_root: Path,
) -> list[tuple[str, str, Path]]:
    """Yield (dataset, preprocess, config_path) for each combination."""
    combos: list[tuple[str, str, Path]] = []
    for dataset in datasets:
        for preprocess in preprocess_variants:
            combos.append((
                dataset,
                preprocess,
                resolve_config_path(dataset, preprocess, project_root),
            ))
    return combos
