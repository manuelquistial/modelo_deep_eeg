# Reproduce Paper Experiments

This guide extends the existing fixed hold-out benchmark (`outputs/`) with publishable repeated evaluation under `outputs_publishable/`.

**Note:** Leave-One-Subject-Out is intentionally excluded.

## 1. Environment

```bash
pip install -e ".[ml,dev,riemann]"
```

## 2. Data preparation (unchanged)

```bash
python scripts/prepare_data.py --config configs/preprocess_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_no_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_bnci_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_bnci_no_ea.yaml
```

## 3. Fixed hold-out benchmark (preserve baseline)

```bash
python scripts/compare_datasets.py --full
```

Results: `outputs/pipeline_comparison.csv` (seed=42, unchanged).

## 4. Repeated subject-disjoint hold-out

```bash
python scripts/run_repeated_holdout.py \
  --dataset physionet \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr \
  --ea both \
  --output-dir outputs_publishable/repeated_holdout/physionet
```

## 5. GroupKFold (classical models)

```bash
python scripts/run_groupkfold.py \
  --dataset physionet \
  --n-splits 5 \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr \
  --ea both \
  --skip-deep \
  --output-dir outputs_publishable/groupkfold/physionet
```

## 6. Statistical analysis

```bash
python scripts/run_statistical_analysis.py \
  --results outputs_publishable/repeated_holdout/physionet/repeated_holdout_results.csv \
  --output-dir outputs_publishable/stats/physionet
```

## 7. Subject-level & neurophysiology

```bash
python scripts/run_subject_level_analysis.py \
  --predictions-root outputs_publishable/repeated_holdout/physionet \
  --output-dir outputs_publishable/subject_level/physionet

python scripts/run_neurophysiology_analysis.py \
  --dataset physionet --ea both \
  --output-dir outputs_publishable/neurophysiology/physionet
```

## 8. Paper tables & figures

```bash
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
python scripts/generate_reproducibility_report.py
python scripts/generate_paper_text_snippets.py
```

## 9. Paperspace / full pipeline (resumable)

Main command on Paperspace GPU:

```bash
./run_paperspace_publishable_experiments.sh
```

Equivalent Python command (GPU-aware, excludes EEGMeModel initially):

```bash
python scripts/run_all_publishable_experiments.py \
  --stage all \
  --datasets physionet bnci \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr eegnet \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-splits 5 \
  --skip-existing \
  --output-root outputs_publishable
```

Smoke test (one seed, one classical model):

```bash
python scripts/run_all_publishable_experiments.py \
  --stage repeated_holdout \
  --datasets bnci \
  --models fbcsp_lda \
  --seeds 0 \
  --ea false \
  --skip-existing \
  --output-root outputs_publishable_smoke_test
```

Add EEGMeModel later:

```bash
python scripts/run_all_publishable_experiments.py \
  --stage repeated_holdout \
  --datasets physionet bnci \
  --models eegme \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --skip-existing \
  --output-root outputs_publishable
```

Resume after interruption: re-run the same command with `--skip-existing`.
Execution log: `outputs_publishable/reports/paperspace_execution_log.txt`
Failed runs: `outputs_publishable/failed_runs/failed_runs.csv`

## Expected outputs

See `outputs_publishable/reports/implementation_plan.md` and `outputs_publishable/reports/reproducibility_report.md`.
