#!/bin/bash
# Submit data preparation and the training grid. Run from anywhere on a MetaCentrum frontend.
#   LANGS="en cs uk" DIMS="100 300" WINDOWS="5 10" SEEDS="0 1" EPOCHS=1 SKIP_PREP=0 jobs/submit_all.sh
# Longer training into a separate runs dir (run names do not encode epochs; TRAIN_BUDGET caps each run):
#   W2V_RUNS=$W2V_STORAGE/runs_e5 EPOCHS=5 TRAIN_BUDGET=2h RUNS_PER_JOB=8 SKIP_PREP=1 jobs/submit_all.sh
# Prep modes: default = download -> preprocess -> vocab/encode; VOCAB_ONLY=1 = only vocab/encode (re-encode
# existing text, e.g. after changing W2V_PAD); SKIP_PREP=1 = training only.
# CPU_SELECT="select=1:ncpus=8:mem=16gb:cl_haldan=True" overrides the resources of every CPU job (pin clusters).
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
LANGS=${LANGS:-"en cs uk"}
DIMS=${DIMS:-300}
WINDOWS=${WINDOWS:-5}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-1}
TRAIN_BUDGET=${TRAIN_BUDGET:-50m}  # per run; resubmitting resumes runs that hit it
VARIANTS=${VARIANTS:-"bidir causal bidir-jepa causal-jepa"}
SKIP_PREP=${SKIP_PREP:-0}
VOCAB_ONLY=${VOCAB_ONLY:-0}
RUNS_PER_JOB=${RUNS_PER_JOB:-20}  # gpu_dgx: runs per 2-GPU job (two lanes)
CPU_L=()
if [[ -n "${CPU_SELECT:-}" ]]; then CPU_L=(-l "$CPU_SELECT"); fi

for W in $WINDOWS; do
    if (( W > W2V_PAD )); then
        echo "window $W exceeds W2V_PAD=$W2V_PAD in jobs/config.sh; re-encode with a larger pad first" >&2
        exit 1
    fi
done

declare -A VOCAB_JOB=()
prep_lang() {  # $1 = lang, $2 = extra dependency string ("" or ":id:id")
    local L=$1 N=${W2V_SHARDS[$1]} dl prep dep=""
    if [[ "$VOCAB_ONLY" != 1 ]]; then
        dl=$(qsub "${CPU_L[@]}" -v W2V_LANG="$L" jobs/download.pbs)
        prep=$(qsub "${CPU_L[@]}" -W depend=afterok:"$dl" -J 0-$((N - 1)) -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/prep.pbs)
        dep="afterok:$prep$2"
        echo "$L: download=$dl prep=$prep"
    elif [[ -n "$2" ]]; then
        dep="afterok:${2#:}"
    fi
    VOCAB_JOB[$L]=$(qsub "${CPU_L[@]}" ${dep:+-W depend=$dep} -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/vocab.pbs)
    echo "$L: vocab=${VOCAB_JOB[$L]}"
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

submit_dgx() {  # $1 = lang, rest = runs
    local L=$1; shift
    local id
    id=$(qsub "${DEP[@]}" -N "w2v-$L-dgx" -v RUNS="$(IFS=+; echo "$*")",EPOCHS="$EPOCHS",W2V_RUNS="$W2V_RUNS",TRAIN_BUDGET="$TRAIN_BUDGET" \
        jobs/train_dgx.pbs)
    echo "train $L ($# runs on gpu_dgx): $id"
}

GPU_QUEUE=${GPU_QUEUE:-gpu_dgx}  # gpu_dgx: 2-GPU jobs of up to RUNS_PER_JOB runs; gpu: one 1-GPU job per run
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
            for W in $WINDOWS; do
                for S in $SEEDS; do
                    NAME="${L}-${MODE}${SUFFIX}-d${D}-w${W}-s${S}"  # mirrors train_run.sh
                    if [[ -f "$W2V_RUNS/$NAME/final_metrics.json" ]]; then
                        echo "skip $NAME: finished"
                        continue
                    fi
                    if [[ "$GPU_QUEUE" == gpu_dgx ]]; then
                        RUNS+=("$L/$V/$D/$W/$S")
                    else
                        id=$(qsub "${DEP[@]}" -N "w2v-$L-$V-d$D-w$W-s$S" \
                            -v W2V_LANG="$L",MODE="$MODE",JEPA="$JEPA",DIM="$D",WINDOW="$W",SEED="$S",EPOCHS="$EPOCHS",W2V_RUNS="$W2V_RUNS",TRAIN_BUDGET="$TRAIN_BUDGET" \
                            jobs/train.pbs)
                        echo "train $L $V d$D w$W s$S: $id"
                    fi
                done
            done
        done
    done
    if [[ "$GPU_QUEUE" == gpu_dgx ]]; then
        for (( i = 0; i < ${#RUNS[@]}; i += RUNS_PER_JOB )); do
            submit_dgx "$L" "${RUNS[@]:i:RUNS_PER_JOB}"
        done
    fi
done
