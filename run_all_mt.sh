#!/bin/zsh
# Full MT pass for every dataset after the running vox pass, then LaBSE agreement scoring.
# Usage: ./run_all_mt.sh            (logs to batch_translate.log)
cd "$(dirname "$0")"
while pgrep -f 'vox_editor/batch_translate.py$' >/dev/null; do sleep 20; done   # let an already-running vox pass finish
for ds in vox demo-d1 demo-d2 zmovie-d1 zmovie-d2; do
  .venv/bin/python vox_editor/batch_translate.py --dataset $ds || echo "!! $ds pass failed"
done
for ds in vox demo-d1 demo-d2 zmovie-d1 zmovie-d2; do
  HF_HUB_OFFLINE=1 .venv/bin/python vox_editor/batch_translate.py --dataset $ds --score 2>&1 | grep -v -E "Warning|Loading weights|Batches" || true
done
echo "ALL DONE"
