import numpy as np

from physionet_mi.evaluation.random_seeds import (
    RepeatSeed,
    generate_repeat_seeds,
    resolve_repeat_seeds,
    save_repeat_seeds,
)


def test_same_master_seed_is_deterministic():
    a = generate_repeat_seeds(master_seed=42, n_repeats=10)
    b = generate_repeat_seeds(master_seed=42, n_repeats=10)
    assert a == b


def test_different_master_seed_differs():
    a = generate_repeat_seeds(master_seed=42, n_repeats=10)
    b = generate_repeat_seeds(master_seed=43, n_repeats=10)
    assert a != b


def test_n_repeats_count():
    records = generate_repeat_seeds(master_seed=42, n_repeats=10)
    assert len(records) == 10


def test_repeat_id_range():
    records = generate_repeat_seeds(master_seed=42, n_repeats=10)
    assert [r.repeat_id for r in records] == list(range(10))


def test_seeds_are_python_ints():
    records = generate_repeat_seeds(master_seed=42, n_repeats=3)
    for rec in records:
        assert isinstance(rec.split_seed, int)
        assert isinstance(rec.model_seed, int)


def test_shared_split_and_model_seed_by_default():
    records = generate_repeat_seeds(master_seed=42, n_repeats=5, separate_model_seeds=False)
    for rec in records:
        assert rec.split_seed == rec.model_seed


def test_separate_model_seeds():
    records = generate_repeat_seeds(master_seed=42, n_repeats=5, separate_model_seeds=True)
    assert any(r.split_seed != r.model_seed for r in records)


def test_uint32_range():
    records = generate_repeat_seeds(master_seed=42, n_repeats=20)
    for rec in records:
        assert 0 <= rec.split_seed < np.iinfo(np.uint32).max
        assert 0 <= rec.model_seed < np.iinfo(np.uint32).max


def test_resolve_default_master_mode():
    records, ms, nr, legacy = resolve_repeat_seeds(
        explicit_seeds=None, master_seed=None, n_repeats=None
    )
    assert ms == 42
    assert nr == 10
    assert legacy is False
    assert len(records) == 10


def test_resolve_explicit_seeds_legacy():
    records, ms, nr, legacy = resolve_repeat_seeds(
        explicit_seeds=[0, 1, 2], master_seed=None, n_repeats=None
    )
    assert ms is None
    assert nr == 3
    assert legacy is True
    assert records[1].split_seed == 1


def test_resolve_mutually_exclusive_error():
    try:
        resolve_repeat_seeds(explicit_seeds=[0], master_seed=42, n_repeats=1)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_save_repeat_seeds(tmp_path):
    records = generate_repeat_seeds(42, 2)
    save_repeat_seeds(records, tmp_path / "repeat_seeds.json", master_seed=42, n_repeats=2)
    assert (tmp_path / "repeat_seeds.csv").exists()
    assert (tmp_path / "repeat_seeds.json").exists()
