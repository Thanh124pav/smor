#!/usr/bin/env bash
# Fast end-to-end flow-through check: runs the test suite plus a tiny run of every
# experiment driver so you can confirm the whole pipeline works (e.g. on a GTX 1650).
# Finishes in ~1-2 minutes. Usage:
#   bash scripts/smoke.sh            # device=auto (cuda if available)
#   DEVICE=cpu bash scripts/smoke.sh
source "$(dirname "$0")/_env.sh"

OUT="${OUTDIR%/}/smoke"
CFG=configs/curvature_reweight.yaml

echo; echo "==================== [1/6] unit tests ===================="
python -m pytest -q

echo; echo "==================== [2/6] validate_hvp ===================="
python -m experiments.validate_hvp

echo; echo "==================== [3/6] collect_demos ===================="
python -m experiments.collect_demos --out "$OUT/data" --n-expert 20 --n-noisy 20 --horizon 25

echo; echo "==================== [4/6] train_reweighting (tiny) ===================="
python -m experiments.train_reweighting --config "$CFG" --n 8 --K 2 --steps 15 \
    --device "$DEVICE" --outdir "$OUT" --eval-episodes 64

echo; echo "==================== [5/6] compare_K (tiny) ===================="
python -m experiments.compare_K --config "$CFG" --n 8 --Ks 1 2 4 \
    --base-steps 40 --oracle-steps 60 --hg-batches 8 --device "$DEVICE" --outdir "$OUT"

echo; echo "==================== [6/6] compare_n + grid (tiny) ===================="
python -m experiments.compare_n --config "$CFG" --ns 8 32 --steps 10 \
    --device "$DEVICE" --outdir "$OUT"
python -m experiments.run_reweighting_grid --config "$CFG" --ns 8 --Ks 1 2 --steps 10 \
    --device "$DEVICE" --outdir "$OUT"

echo; echo "==================== SMOKE OK ===================="
echo "artifacts in: $OUT/"
