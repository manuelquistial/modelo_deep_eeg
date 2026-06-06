"""Master-seed strategy for repeated subject-disjoint hold-out."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_UINT32_MAX = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class RepeatSeed:
    repeat_id: int
    split_seed: int
    model_seed: int


def generate_repeat_seeds(
    master_seed: int = 42,
    n_repeats: int = 10,
    separate_model_seeds: bool = False,
) -> list[RepeatSeed]:
    """
    Generate deterministic repeated-experiment seeds from a master seed.

    Parameters
    ----------
    master_seed:
        Seed used to initialize NumPy's Generator.
    n_repeats:
        Number of repeated experiment partitions.
    separate_model_seeds:
        If True, generate independent split and model seeds from the same RNG.
        If False, use the same generated seed for split and model randomness.

    Returns
    -------
    list[RepeatSeed]
        One seed object per repetition.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    rng = np.random.default_rng(int(master_seed))
    split_seeds = rng.integers(0, _UINT32_MAX, size=n_repeats, dtype=np.uint32)

    if separate_model_seeds:
        model_seeds = rng.integers(0, _UINT32_MAX, size=n_repeats, dtype=np.uint32)
    else:
        model_seeds = split_seeds

    return [
        RepeatSeed(
            repeat_id=i,
            split_seed=int(split_seeds[i]),
            model_seed=int(model_seeds[i]),
        )
        for i in range(n_repeats)
    ]


def repeat_seeds_from_explicit_list(seeds: list[int]) -> list[RepeatSeed]:
    """Legacy mode: explicit split seed list (split_seed == model_seed)."""
    return [
        RepeatSeed(repeat_id=i, split_seed=int(s), model_seed=int(s))
        for i, s in enumerate(seeds)
    ]


def resolve_repeat_seeds(
    *,
    explicit_seeds: list[int] | None,
    master_seed: int | None,
    n_repeats: int | None,
    separate_model_seeds: bool = False,
) -> tuple[list[RepeatSeed], int | None, int | None, bool]:
    """
    Resolve repeat seeds from CLI-style arguments.

    Returns (records, master_seed, n_repeats, legacy_explicit_seeds).
    """
    if explicit_seeds is not None and master_seed is not None:
        raise ValueError("Use either --seeds or --master-seed/--n-repeats, not both.")

    if explicit_seeds is not None:
        return repeat_seeds_from_explicit_list(explicit_seeds), None, len(explicit_seeds), True

    ms = 42 if master_seed is None else int(master_seed)
    nr = 10 if n_repeats is None else int(n_repeats)
    records = generate_repeat_seeds(ms, nr, separate_model_seeds=separate_model_seeds)
    return records, ms, nr, False


def save_repeat_seeds(
    seed_records: list[RepeatSeed],
    output_path: str | Path,
    *,
    master_seed: int | None = None,
    n_repeats: int | None = None,
    legacy_explicit_seeds: bool = False,
) -> None:
    """Save generated repeat seeds as JSON and CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in seed_records:
        row = asdict(rec)
        if master_seed is not None:
            row["master_seed"] = int(master_seed)
        if n_repeats is not None:
            row["n_repeats"] = int(n_repeats)
        row["legacy_explicit_seeds"] = bool(legacy_explicit_seeds)
        rows.append(row)

    stem = path.with_suffix("")
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
