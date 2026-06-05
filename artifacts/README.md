# Artifacts layout

All generated data and experiment outputs live under `artifacts/`.

```
artifacts/
├── raw/                      # MNE/MOABB downloads
├── cache/                    # Preprocessed EEG caches (*.joblib, holdout arrays)
├── runs/
│   ├── baseline/             # Fixed hold-out benchmark (compare_datasets, train_holdout)
│   └── publishable/          # Repeated hold-out, GroupKFold, stats, analysis
│       ├── repeated_holdout/
│       ├── groupkfold/
│       ├── riemannian/
│       ├── stats/
│       ├── subject_level/
│       ├── neurophysiology/
│       └── ea_diagnostics/
├── paper/
│   ├── tables/               # Generated LaTeX/CSV tables
│   └── figures/              # Generated paper figures
├── reports/                  # Logs, reproducibility report, implementation notes
├── failed_runs/              # Failed experiment log (CSV)
└── smoke/                    # Quick smoke-test outputs
```

LaTeX **source** for the manuscript stays in `paper/ieee/` (versioned).

## Legacy paths

Older clones may still have `data/`, `outputs/`, or `outputs_publishable/`.
Run once after pulling:

```bash
python scripts/setup_artifacts_layout.py
```

That creates symlinks from legacy folders into `artifacts/` when needed.

## Override

```bash
export PHYSIONET_MI_ARTIFACTS_ROOT=/path/to/custom/artifacts
```
