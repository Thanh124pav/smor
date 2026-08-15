#!/usr/bin/env bash
# Full experiment suite for the Module-1 report (reproduces RESULTS_REWEIGHTING.md).
# Usage:
#   bash scripts/run_experiments.sh                 # device=auto, outdir=results
#   DEVICE=cpu OUTDIR=results_cpu bash scripts/run_experiments.sh
#
# NOTE: compare_n / the grid at n=1 create one beta cell PER trajectory (many groups ->
# many HVPs) and are the slow cells. Override the granularities with env vars if needed:
#   NS_TRAIN, KS, NS_COMPARE, NS_GRID, KS_GRID, STEPS
source "$(dirname "$0")/_env.sh"

CFG="${CFG:-configs/curvature_reweight.yaml}"
STEPS="${STEPS:-40}"
KS="${KS:-1 2 4}"
NS_COMPARE="${NS_COMPARE:-1 8 32}"
NS_GRID="${NS_GRID:-8 32}"
KS_GRID="${KS_GRID:-1 2 4}"

log() { echo; echo "########## $* ##########"; }

log "TRAIN sweep K=[$KS] at n=8 (steps=$STEPS)"
for K in $KS; do
  python -m experiments.train_reweighting --config "$CFG" --n 8 --K "$K" --steps "$STEPS" \
      --device "$DEVICE" --outdir "$OUTDIR"
done

log "COMPARE_K (long-horizon utility, §8)"
python -m experiments.compare_K --config "$CFG" --n 8 --Ks $KS \
    --base-steps 100 --oracle-steps 150 --hg-batches 16 --device "$DEVICE" --outdir "$OUTDIR"

log "COMPARE_N (granularity trade-off, §9) — ns=[$NS_COMPARE]"
python -m experiments.compare_n --config "$CFG" --ns $NS_COMPARE --steps 20 \
    --device "$DEVICE" --outdir "$OUTDIR"

log "GRID n=[$NS_GRID] x K=[$KS_GRID]"
python -m experiments.run_reweighting_grid --config "$CFG" --ns $NS_GRID --Ks $KS_GRID \
    --steps "$STEPS" --device "$DEVICE" --outdir "$OUTDIR"

log "DONE — results in $OUTDIR/"
echo "  summaries: $OUTDIR/n*_K*_seed*/summary.json"
echo "  compare_K: $OUTDIR/compare_K.json"
echo "  compare_n: $OUTDIR/compare_n.json"
echo "  grid:      $OUTDIR/grid/grid.json"
