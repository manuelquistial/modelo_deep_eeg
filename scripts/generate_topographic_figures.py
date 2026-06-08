#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.topographic_figure import generate_topographic_figures, write_topoplot_notes
from physionet_mi.paths import artifacts_root

art = artifacts_root(ROOT)
out = art / "paper" / "figures"
topo = generate_topographic_figures(art, out / "fig_topoplots_motor_imagery", ea_stem=out / "fig_topoplots_before_after_ea")
write_topoplot_notes(out / "fig_topoplots_notes.md", bnci_sparse=not list((art / "cache").glob("bnci_*")))
print(topo)
