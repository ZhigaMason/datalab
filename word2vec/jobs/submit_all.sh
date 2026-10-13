#!/bin/bash
# Submit data preparation and the training grid. Run from anywhere on a MetaCentrum frontend.
#   LANGS="en cs uk" DIMS="100 300" SEEDS="0 1" EPOCHS=1 SKIP_PREP=0 jobs/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
LANGS=${LANGS:-"en cs uk"}
DIMS=${DIMS:-300}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-1}
VARIANTS=${VARIANTS:-"bidir causal bidir-jepa causal-jepa"}
SKIP_PREP=${SKIP_PREP:-0}

declare -A VOCAB_JOB=()
prep_lang() {  # $1 = lang, $2 = extra dependency string ("" or ":id:id")
    local L=$1 N=${W2V_SHARDS[$1]} dl prep
    dl=$(qsub -v W2V_LANG="$L" jobs/download.pbs)
    prep=$(qsub -W depend=afterok:"$dl" -J 0-$((N - 1)) -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/prep.pbs)
    VOCAB_JOB[$L]=$(qsub -W depend=afterok:"$prep$2" -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/vocab.pbs)
    echo "$L: download=$dl prep=$prep vocab=${VOCAB_JOB[$L]}"
}

if [[ "$SKIP_PREP" != 1 ]]; then
    for L in cs uk; do [[ " $LANGS " == *" $L "* ]] && prep_lang "$L" ""; done
    if [[ " $LANGS " == *" en "* ]]; then
        EN_DEPS=""
        for L in cs uk; do
            if [[ -n "${VOCAB_JOB[$L]:-}" ]]; then
                EN_DEPS+=":${VOCAB_JOB[$L]}"
            elif [[ ! -f "$W2V_DATA/$L/stats.json" ]]; then
                echo "en needs $L stats for its token cap: include $L in LANGS or prepare it first" >&2
                exit 1
            fi
        done
        prep_lang en "$EN_DEPS"
    fi
fi

GPU_QUEUE=${GPU_QUEUE:-gpu_dgx}  # gpu_dgx: one 2-GPU job per language; gpu: one 1-GPU job per run
for L in $LANGS; do
    DEP=()
    if [[ -n "${VOCAB_JOB[$L]:-}" ]]; then DEP=(-W depend=afterok:"${VOCAB_JOB[$L]}"); fi
    RUNS=()
    for V in $VARIANTS; do
        MODE=${V%-jepa}
        JEPA=0
        SUFFIX=""
        if [[ "$V" == *-jepa ]]; then JEPA=1; SUFFIX="-jepa"; fi
        for D in $DIMS; do
            for S in $SEEDS; do
                NAME="${L}-${MODE}${SUFFIX}-d${D}-s${S}"  # mirrors train_run.sh
                if [[ -f "$W2V_RUNS/$NAME/final_metrics.json" ]]; then
                    echo "skip $NAME: finished"
                    continue
                fi
                if [[ "$GPU_QUEUE" == gpu_dgx ]]; then
                    RUNS+=("$L/$V/$D/$S")
                else
                    id=$(qsub "${DEP[@]}" -N "w2v-$L-$V-d$D-s$S" \
                        -v W2V_LANG="$L",MODE="$MODE",JEPA="$JEPA",DIM="$D",SEED="$S",EPOCHS="$EPOCHS" jobs/train.pbs)
                    echo "train $L $V d$D s$S: $id"
                fi
            done
        done
    done
    if [[ "$GPU_QUEUE" == gpu_dgx && ${#RUNS[@]} -gt 0 ]]; then
        id=$(qsub "${DEP[@]}" -N "w2v-$L-dgx" \
            -v RUNS="$(IFS=+; echo "${RUNS[*]}")",EPOCHS="$EPOCHS" jobs/train_dgx.pbs)
        echo "train $L (${#RUNS[@]} runs on gpu_dgx): $id"
    fi
done
