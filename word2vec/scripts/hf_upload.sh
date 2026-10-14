#!/bin/bash
# Publish the grid's ONNX tables (int8 + fp16) to a Hugging Face model repo. Run on a MetaCentrum node,
# in a shell where `hf auth whoami` shows your account, after scripts.quantize_onnx has converted every run:
#   PYTHONUNBUFFERED=1 bash scripts/hf_upload.sh user/repo 2>&1 | tee "$W2V_STORAGE/hf_upload.log"
# Files are staged as hard links in $W2V_STORAGE/hf_stage (no extra space): one directory per run with
# model_int8.onnx, model_fp16.onnx, config.json and final_metrics*.json, plus vocab/<lang>.json (one per
# language; every run of a language must share it) and scripts/hf_model_card.md as README.md.
# STAGE_ONLY=1 stops after staging. Re-running resumes the upload. If it sends nothing for minutes, retry
# with HF_HUB_DISABLE_XET=1 (classic LFS upload instead of the Xet backend).
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
REPO=${1:?usage: scripts/hf_upload.sh user/repo}
STAGE=$W2V_STORAGE/hf_stage

rm -rf "$STAGE/vocab"; find "$STAGE" -mindepth 1 -maxdepth 1 -name '*-s*' -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$STAGE/vocab"  # keeps $STAGE/.cache/huggingface: the upload state of an interrupted run
for r in "$W2V_RUNS"/*-s[0-9]*; do
    n=$(basename "$r")
    [[ -f $r/model.onnx && -f $r/model_fp16.onnx ]] || { echo "skip $n: needs int8 model.onnx and model_fp16.onnx"; continue; }
    mkdir -p "$STAGE/$n"
    ln "$r/model.onnx" "$STAGE/$n/model_int8.onnx"
    ln "$r/model_fp16.onnx" "$STAGE/$n/model_fp16.onnx"
    for f in config.json final_metrics.json final_metrics_int8.json; do
        if [[ -f $r/$f ]]; then ln "$r/$f" "$STAGE/$n/$f"; fi
    done
    lang=${n%%-*}
    if [[ ! -f $STAGE/vocab/$lang.json ]]; then
        ln "$r/vocab.json" "$STAGE/vocab/$lang.json"
    elif ! cmp -s "$r/vocab.json" "$STAGE/vocab/$lang.json"; then
        echo "vocab of $n differs from vocab/$lang.json" >&2; exit 1
    fi
done
cp scripts/hf_model_card.md "$STAGE/README.md"
echo "staged $(find "$STAGE" -name '*.onnx' | wc -l) onnx files, $(du -sh "$STAGE" | cut -f1)"

if [[ ${STAGE_ONLY:-0} == 1 ]]; then exit 0; fi
hf repo create "$REPO" --repo-type model --exist-ok
hf upload-large-folder "$REPO" "$STAGE" --repo-type model
echo UPLOAD_DONE
