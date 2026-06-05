#!/usr/bin/env bash
set -e

echo "Starting publishable EEG-MI experiments on Paperspace"

python --version
nvidia-smi || true

python scripts/setup_artifacts_layout.py

mkdir -p artifacts/reports
mkdir -p artifacts/failed_runs

python scripts/run_all_publishable_experiments.py \
  --stage all \
  --datasets physionet bnci \
  --models fbcsp_lda csp_svm riemann_mdm riemann_ts_lr eegnet \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-splits 5 \
  --skip-existing \
  --output-root artifacts \
  2>&1 | tee artifacts/reports/paperspace_execution_log.txt

echo "Finished publishable EEG-MI experiments"
