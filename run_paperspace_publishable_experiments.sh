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
  --master-seed 42 \
  --n-repeats 10 \
  --n-splits 5 \
  --skip-existing \
  --parallel-jobs 4 \
  --output-root artifacts \
  2>&1 | tee artifacts/reports/paperspace_execution_log.txt

echo "Finished publishable EEG-MI experiments"

if [ -f artifacts/reports/EXECUTION_COMPLETE.txt ]; then
  echo ""
  echo "SUCCESS: full pipeline completed."
  echo "Marker file: artifacts/reports/EXECUTION_COMPLETE.txt"
  cat artifacts/reports/EXECUTION_COMPLETE.txt
elif [ -f artifacts/reports/EXECUTION_FAILED.txt ]; then
  echo ""
  echo "WARNING: pipeline did not complete successfully."
  echo "Marker file: artifacts/reports/EXECUTION_FAILED.txt"
  cat artifacts/reports/EXECUTION_FAILED.txt
else
  echo ""
  echo "WARNING: no execution marker found (run may have been interrupted)."
fi
