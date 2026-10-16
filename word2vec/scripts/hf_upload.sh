#!/bin/bash
# Publish a grid's ONNX tables to a Hugging Face model repo. Run on a MetaCentrum node, in a shell where
# `hf auth whoami` shows your account:
#   PYTHONUNBUFFERED=1 bash scripts/hf_upload.sh user/repo [RUNS_DIR [SUBDIR]] 2>&1 | tee "$W2V_STORAGE/hf_upload.log"
# RUNS_DIR (default $W2V_RUNS) goes to the repo root, or to SUBDIR/ when given. Each run directory gets
# model_int8.onnx, model_fp16.onnx when the run has one (runs converted by scripts.quantize_onnx), config.json,
# final_metrics*.json and metrics.jsonl. To replace what is already at the root, stage with STAGE_ONLY=1 and
# apply with scripts/hf_sync.py.
# vocab/<lang>.json is shared by all grids: every run of a language must have the same vocabulary.
# scripts/hf_model_card.md becomes README.md. Files are staged as hard links (no extra space) in
# $W2V_STORAGE/hf_stage[_SUBDIR], mirroring the repo layout; the upload only adds or updates files.
# STAGE_ONLY=1 stops after staging. Re-running resumes the upload. If it sends nothing for minutes, retry
# with HF_HUB_DISABLE_XET=1 (classic LFS upload instead of the Xet backend).
set -euo pipefail
cd "$(dirname "$0")/.."
# config.sh points HF_HOME at storage (dataset cache), where no login token lives; keep the caller's login.
caller_hf_home=${HF_HOME-__unset__}
source jobs/config.sh
if [[ $caller_hf_home == __unset__ ]]; then unset HF_HOME; else export HF_HOME=$caller_hf_home; fi
REPO=${1:?usage: scripts/hf_upload.sh user/repo [RUNS_DIR [SUBDIR]]}
RUNS=${2:-$W2V_RUNS}
SUBDIR=${3:-}
STAGE=$W2V_STORAGE/hf_stage${SUBDIR:+_$SUBDIR}
DEST=$STAGE${SUBDIR:+/$SUBDIR}

rm -rf "$STAGE/vocab"; find "$DEST" -mindepth 1 -maxdepth 1 -name '*-s*' -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$STAGE/vocab" "$DEST"  # keeps $STAGE/.cache/huggingface: the upload state of an interrupted run
for r in "$RUNS"/*-s[0-9]*; do
    n=$(basename "$r")
    [[ -f $r/model.onnx && -f $r/final_metrics.json ]] || { echo "skip $n: not finished"; continue; }
    mkdir -p "$DEST/$n"
    ln "$r/model.onnx" "$DEST/$n/model_int8.onnx"
    for f in model_fp16.onnx config.json final_metrics.json final_metrics_int8.json; do
        if [[ -f $r/$f ]]; then ln "$r/$f" "$DEST/$n/$f"; fi
    done
    if [[ -f $r/metrics.jsonl ]]; then ln "$r/metrics.jsonl" "$DEST/$n/metrics.jsonl"; fi
    lang=${n%%-*}
    if [[ ! -f $STAGE/vocab/$lang.json ]]; then
        ln "$r/vocab.json" "$STAGE/vocab/$lang.json"
    elif ! cmp -s "$r/vocab.json" "$STAGE/vocab/$lang.json"; then
        echo "vocab of $n differs from vocab/$lang.json" >&2; exit 1
    fi
done
cp scripts/hf_model_card.md "$STAGE/README.md"
echo "staged $(find "$DEST" -name '*.onnx' | wc -l) onnx files in ${SUBDIR:-the repo root}, $(du -sh "$STAGE" | cut -f1)"

if [[ ${STAGE_ONLY:-0} == 1 ]]; then exit 0; fi
hf repo create "$REPO" --repo-type model --exist-ok
hf upload-large-folder "$REPO" "$STAGE" --repo-type model
echo UPLOAD_DONE
