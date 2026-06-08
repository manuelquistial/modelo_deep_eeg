#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.distribution_figure import (
    generate_repeated_holdout_distributions,
    write_distribution_notes,
)
from physionet_mi.paths import artifacts_root, publishable_runs_dir

out = artifacts_root(ROOT) / "paper" / "figures"
generate_repeated_holdout_distributions(publishable_runs_dir(ROOT), out / "fig_repeated_holdout_distributions")
write_distribution_notes(out / "fig_repeated_holdout_distributions_notes.md")
print(f"Wrote distributions to {out}")
