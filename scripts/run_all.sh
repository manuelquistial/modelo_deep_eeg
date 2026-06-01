#!/usr/bin/env bash
# Full benchmark: prepare all caches + compare all models x datasets x preprocess.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

echo "=== Step 1/2: prepare_data (4 cache variants) ==="
for cfg in \
  preprocess_ea \
  preprocess_no_ea \
  preprocess_bnci_ea \
  preprocess_bnci_no_ea
do
  echo "--- configs/${cfg}.yaml ---"
  python scripts/prepare_data.py --config "configs/${cfg}.yaml"
done

echo "=== Step 2/2: compare_datasets (full LOSO, all models) ==="
python scripts/compare_datasets.py \
  --datasets physionet,bnci2014_001 \
  --preprocess ea,no_ea \
  --full-loso

echo "Done. Results: outputs/pipeline_comparison.csv"
