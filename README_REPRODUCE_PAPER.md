# Reproduce Paper Experiments

All generated data and experiment outputs are centralized under `artifacts/`.
See `artifacts/README.md` for the full layout.

**Note:** Leave-One-Subject-Out is intentionally excluded.

## 0. Setup artifact folders

```bash
python scripts/setup_artifacts_layout.py
```

## 1. Environment

```bash
pip install -e ".[ml,dev,riemann]"
```

## 2. Data preparation

```bash
python scripts/prepare_data.py --config configs/preprocess_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_no_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_bnci_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_bnci_no_ea.yaml
```

Cache: `artifacts/cache/`

## 3. Fixed hold-out benchmark (preserve baseline)

```bash
python scripts/compare_datasets.py --full
```

Results: `artifacts/runs/baseline/pipeline_comparison.csv` (seed=42).

## 4. Repeated subject-disjoint hold-out

```bash
python scripts/run_repeated_holdout.py \
  --dataset physionet \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr \
  --ea both \
  --output-dir artifacts/runs/publishable/repeated_holdout/physionet
```

## 5. GroupKFold (classical models)

```bash
python scripts/run_groupkfold.py \
  --dataset physionet \
  --n-splits 5 \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr \
  --ea both \
  --skip-deep \
  --output-dir artifacts/runs/publishable/groupkfold/physionet
```

## 6. Statistical analysis

```bash
python scripts/run_statistical_analysis.py \
  --results artifacts/runs/publishable/repeated_holdout/physionet/repeated_holdout_results.csv \
  --output-dir artifacts/runs/publishable/stats/physionet
```

## 7. Subject-level & neurophysiology

```bash
python scripts/run_subject_level_analysis.py \
  --predictions-root artifacts/runs/publishable/repeated_holdout/physionet \
  --output-dir artifacts/runs/publishable/subject_level/physionet

python scripts/run_neurophysiology_analysis.py \
  --dataset physionet --ea both \
  --output-dir artifacts/runs/publishable/neurophysiology/physionet
```

## 8. Paper tables & figures

```bash
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
python scripts/generate_reproducibility_report.py
python scripts/generate_paper_text_snippets.py
```

Outputs: `artifacts/paper/tables/`, `artifacts/paper/figures/`, `artifacts/reports/`

## 9. Paperspace / full pipeline (resumable)

```bash
./run_paperspace_publishable_experiments.sh
```

Smoke test:

```bash
python scripts/run_all_publishable_experiments.py \
  --stage repeated_holdout \
  --datasets bnci \
  --models fbcsp_lda \
  --seeds 0 \
  --ea false \
  --skip-existing \
  --output-root artifacts/smoke
```

Resume after interruption: re-run with `--skip-existing`.

- Log: `artifacts/reports/paperspace_execution_log.txt`
- Failed runs: `artifacts/failed_runs/failed_runs.csv`

LaTeX **source** stays in `paper/ieee/` (versioned). Generated tables/figures go to `artifacts/paper/`.
