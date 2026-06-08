#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.pipeline_figure import generate_pipeline_architecture_figure, write_pipeline_notes
from physionet_mi.paths import artifacts_root

out = artifacts_root(ROOT) / "paper" / "figures"
generate_pipeline_architecture_figure(out / "fig_pipeline_architecture")
write_pipeline_notes(out / "fig_pipeline_architecture_notes.md")
print(f"Wrote pipeline figure to {out}")
