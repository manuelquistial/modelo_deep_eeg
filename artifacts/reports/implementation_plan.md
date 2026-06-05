# Implementation Plan — Publishable EEG MI Benchmark

**Date:** 2026-06-05  
**Goal:** Extend existing `modelo_deep_eeg` repo with IEEE-style repeated evaluation, Riemannian baselines, statistics, neurophysiology, and paper artifacts — **without breaking** fixed hold-out results (`random_state=42`).

**Excluded by design:** Leave-One-Subject-Out (LOSO) — not implemented or referenced in new code.

---

## 1. Existing scripts (`scripts/`)

| Script | Wraps | Status |
|--------|-------|--------|
| `prepare_data.py` | `cli.prepare_data` | Keep |
| `train_holdout.py` | `training.holdout` | Keep — fixed benchmark |
| `train_loso.py` | `training.loso` | Keep (legacy, not used in paper pipeline) |
| `run_baseline_lda.py` | FBCSP+LDA hold-out | Keep |
| `run_baseline_csp_svm.py` | CSP+SVM hold-out | Keep |
| `compare_datasets.py` | `evaluation.compare` | Keep |
| `run_ablation.py` | ablation summary | Keep |
| `run_all.sh` | full legacy pipeline | Keep |

## 2. Existing model modules (`src/physionet_mi/`)

| Module | Contents |
|--------|----------|
| `models/eegme.py` | EEGMeModel |
| `models/eegnet.py` | EEGNet |
| `models/registry.py` | `build_model()` |
| `baseline/lda_fbcsp.py` | FBCSP+LDA |
| `baseline/csp_svm_classifier.py` | CSP+SVM |
| `training/holdout.py` | `run_holdout()` |
| `training/trainer.py` | `fit()` |
| `evaluation/metrics.py` | `compute_metrics()` |
| `evaluation/reporting.py` | `save_run_artifacts()` |
| `data/subject_dict.py` | `split_subjects_holdout`, `split_val_subjects_stratified` |
| `data/cache.py` | `build_and_cache_holdout`, `load_holdout_arrays` |
| `data/preprocessing.py` | EA, filters, harmonization |

## 3. Preprocessing / cache utilities

- **Cache key:** `{dataset}_lr_{ea|no_ea}_{N}sub_{all|hash}` under `data/processed/`
- **Fixed split:** `split.random_state=42` baked into cache at build time
- **Raw dict:** `subject_dict_raw.joblib` saved in cache — **reused for multi-seed splits**
- **EA:** per-subject, DEV/TEST separate in `preprocess_train_eval_subject_dicts`

## 4. Existing output structure

- `outputs/` (gitignored): fixed hold-out runs, `pipeline_comparison.csv`
- `literature_comparison/`: paper literature assets
- **New:** `outputs_publishable/` for repeated/groupkfold/stats/paper artifacts

## 5. Dependencies

**Installed:** numpy, scipy, sklearn, pandas, mne, moabb, matplotlib, pyyaml, joblib, torch (optional `[ml]`)

**Adding (optional):** `pyriemann>=0.5` in `[project.optional-dependencies].riemann` with log-Euclidean fallback

## 6. Reusable functions (extend, do not replace)

| Symbol | Location | Reuse for |
|--------|----------|-----------|
| `split_subjects_holdout` | `data/subject_dict.py` | Repeated hold-out per seed |
| `split_val_subjects_stratified` | `data/subject_dict.py` | DL val split |
| `preprocess_train_eval_subject_dicts` | `data/preprocessing.py` | Per-split preprocessing |
| `to_model_arrays` | `data/subject_dict.py` | Tensor conversion |
| `extract_fbcsp_features` + LDA | `baseline/lda_fbcsp.py` | Classical runner |
| `CSPSVMClassifier` | `baseline/csp_svm_classifier.py` | Classical runner |
| `run_holdout` | `training/holdout.py` | Deep models (via array injection) |
| `compute_metrics` | `evaluation/metrics.py` | All evaluators |
| `save_run_artifacts` | `evaluation/reporting.py` | Per-run outputs |
| `resolve_config_path` | `evaluation/config_matrix.py` | Dataset×EA configs |

## 7. New files to create

### Evaluation
- `evaluation/subject_splits.py` — wrappers + GroupKFold + split metadata
- `evaluation/repeated_holdout.py` — seed loop orchestration
- `evaluation/groupkfold.py` — GroupKFold orchestration
- `evaluation/bootstrap.py` — bootstrap CI
- `evaluation/statistical_tests.py` — Wilcoxon, Friedman, McNemar
- `evaluation/runners.py` — train/eval dispatch on arbitrary arrays

### Baselines
- `baselines/riemannian.py` — MDM, tangent-space LR (+ fallback)

### Analysis
- `analysis/subject_level.py`
- `analysis/erd_ers.py`
- `analysis/lateralization.py`
- `analysis/ea_diagnostics.py`
- `analysis/csp_patterns.py` (stub/minimal)

### Paper
- `paper/latex_tables.py`
- `paper/figures.py`
- `paper/report_generator.py`

### Scripts (12 new)
- `run_repeated_holdout.py`, `run_groupkfold.py`, `run_riemannian_baselines.py`
- `run_statistical_analysis.py`, `run_subject_level_analysis.py`
- `run_neurophysiology_analysis.py`, `run_ea_diagnostics.py`
- `generate_paper_tables.py`, `generate_paper_figures.py`
- `generate_reproducibility_report.py`, `generate_paper_text_snippets.py`
- `run_all_publishable_experiments.py`

### Configs (`configs/experiments/`)
- 8 YAML experiment configs (repeated, groupkfold, riemannian, neuro, paper)

### Tests
- `test_subject_splits.py`, `test_no_subject_leakage.py`, `test_metrics.py`
- `test_riemannian_baseline_shapes.py`, `test_lateralization_computation.py`
- `test_output_table_generation.py`

### Docs
- `README_REPRODUCE_PAPER.md`
- `outputs_publishable/reports/implementation_summary.md` (after smoke tests)

## 8. Minimal extensions to existing files

| File | Change |
|------|--------|
| `data/cache.py` | Add `load_raw_subject_dict()`, `build_arrays_for_subject_split()` |
| `evaluation/reporting.py` | Optional `groups` in predictions.csv (backward compatible) |
| `pyproject.toml` | Optional `riemann` extra; new script entry points |
| `.gitignore` | Add `/outputs_publishable/` only if large binaries — keep trackable CSVs |

**No changes** to `run_holdout` signature, `pipeline_comparison.csv` format, or fixed `outputs/` paths.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Seed 42 repeated run ≠ fixed cache | `build_arrays_for_subject_split` must use same `split_subjects_holdout` + preprocess; verify dev/test IDs match `meta.json` |
| EA leakage across seeds | Preprocess per split; EA reference from DEV subjects only for alignment applied consistently |
| Breaking `save_run_artifacts` | Add optional kwargs only |
| pyriemann missing | Fallback log-Euclidean + LogisticRegression |
| Deep repeated hold-out slow | `--skip-deep`, `--only-classical` flags |
| Overwriting fixed outputs | All new artifacts under `outputs_publishable/` |

## 10. Execution order

1. ✅ This plan
2. `subject_splits.py` + cache split helpers + leakage tests
3. `runners.py` + `riemannian.py`
4. `repeated_holdout.py` + `run_repeated_holdout.py` (classical first)
5. `groupkfold.py` + script
6. `bootstrap.py` + `statistical_tests.py` + script
7. Analysis modules + scripts
8. Paper generators + `run_all_publishable_experiments.py`
9. Smoke test: PhysioNet, seed 0, `fbcsp_lda`, EA off
10. `implementation_summary.md`

## 11. Smoke test command (after implementation)

```bash
python scripts/run_repeated_holdout.py \
  --dataset physionet \
  --seeds 0 \
  --models fbcsp_lda \
  --ea false \
  --output-dir outputs_publishable/repeated_holdout/physionet_smoke
```

Preserve baseline verification:

```bash
# seed=42 should match cached meta dev/test subjects
python scripts/run_repeated_holdout.py \
  --dataset physionet --seeds 42 --models fbcsp_lda --ea true \
  --output-dir outputs_publishable/repeated_holdout/physionet_verify
```
