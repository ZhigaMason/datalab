#!/bin/bash
# Train one run inside a PBS job: stage data to WORK, train under a time budget, copy the run dir back atomically.
# Env: W2V_LANG MODE JEPA(0/1) DIM SEED [WINDOW] [EPOCHS] [TRAIN_BUDGET] WORK (per-run scratch dir); jobs/config.sh sourced by caller.
set -euo pipefail
: "${W2V_LANG:?}" "${MODE:?}" "${DIM:?}" "${SEED:?}" "${WORK:?}"
JEPA_FLAG=()
SUFFIX=""
if [[ "${JEPA:-0}" == 1 ]]; then JEPA_FLAG=(--jepa); SUFFIX="-jepa"; fi
WINDOW=${WINDOW:-5}
NAME="${W2V_LANG}-${MODE}${SUFFIX}-d${DIM}-w${WINDOW}-s${SEED}"  # mirrors w2v.train.run_name

D="$WORK/data"
mkdir -p "$D/$W2V_LANG" "$D/benchmarks" "$WORK/runs"

STAGED=0
copy_back() {
    if [[ "$STAGED" != 1 ]]; then return 0; fi
    mkdir -p "$W2V_RUNS"
    rm -rf "$W2V_RUNS/.$NAME.tmp"
    if cp -r "$WORK/runs/$NAME" "$W2V_RUNS/.$NAME.tmp" \
        && rm -rf "$W2V_RUNS/$NAME" && mv "$W2V_RUNS/.$NAME.tmp" "$W2V_RUNS/$NAME"; then
        rm -rf "$WORK"
    else
        echo "copy back failed for $NAME; results remain in $WORK on $(hostname)" >&2
    fi
}
trap copy_back EXIT

cp "$W2V_DATA/$W2V_LANG"/{tokens.bin,vocab.json,stats.json} "$D/$W2V_LANG/"
if [[ -d "$W2V_DATA/benchmarks/$W2V_LANG" ]]; then cp -r "$W2V_DATA/benchmarks/$W2V_LANG" "$D/benchmarks/"; fi
if [[ -d "$W2V_RUNS/$NAME" ]]; then cp -r "$W2V_RUNS/$NAME" "$WORK/runs/"; fi
STAGED=1

echo "[$(date +%T)] start $NAME on GPU ${CUDA_VISIBLE_DEVICES:-?}"
TRAIN_BUDGET=${TRAIN_BUDGET:-50m}
rc=0
timeout --signal=TERM --kill-after=2m "$TRAIN_BUDGET" "$UV_PROJECT_ENVIRONMENT/bin/python" -m w2v.train \
    --lang "$W2V_LANG" --mode "$MODE" "${JEPA_FLAG[@]}" --dim "$DIM" --window "$WINDOW" --seed "$SEED" \
    --epochs "${EPOCHS:-1}" --data-dir "$D" --run-dir "$WORK/runs/$NAME" || rc=$?
if (( rc != 0 )); then
    echo "[$(date +%T)] $NAME stopped (exit $rc): budget $TRAIN_BUDGET or error; resubmit to resume from ckpt.pt" >&2
    exit "$rc"
fi
echo "[$(date +%T)] done $NAME"
