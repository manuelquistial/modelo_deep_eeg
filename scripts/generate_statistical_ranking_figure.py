#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.ranking_figure import generate_statistical_ranking_figures, write_ranking_notes
from physionet_mi.paths import artifacts_root, publishable_runs_dir

out = artifacts_root(ROOT) / "paper" / "figures"
result = generate_statistical_ranking_figures(publishable_runs_dir(ROOT), out)
write_ranking_notes(out / "fig_critical_difference_notes.md", result)
print(result)
