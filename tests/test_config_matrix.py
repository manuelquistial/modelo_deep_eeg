"""Tests for dataset × preprocess config resolution."""

from pathlib import Path

import pytest

from physionet_mi.evaluation.config_matrix import (
    CONFIG_MATRIX,
    iter_benchmark_configs,
    resolve_config_path,
)


def test_config_matrix_covers_both_datasets_and_preprocess():
    for dataset in ("physionet", "bnci2014_001"):
        assert set(CONFIG_MATRIX[dataset]) == {"ea", "no_ea"}


def test_resolve_config_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = resolve_config_path("physionet", "ea", root)
    assert path.name == "preprocess_ea.yaml"
    assert path.exists()


def test_iter_benchmark_configs_count():
    root = Path(__file__).resolve().parents[2]
    combos = iter_benchmark_configs(
        ["physionet", "bnci2014_001"],
        ["ea", "no_ea"],
        root,
    )
    assert len(combos) == 4


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_config_path("unknown", "ea")
