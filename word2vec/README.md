# word2vec variants: bidirectional / causal × negative sampling / JEPA

Word2vec on Wikipedia (en, cs, uk — cs/uk lemmatized with UDPipe) with a fixed
convolutional context average, trained four ways:

| variant | context | objective |
|---|---|---|
| `bidir` | k left + k right | negative sampling |
| `causal` | k left | negative sampling |
| `bidir-jepa` | k left + k right | predict EMA target embedding (cosine + VICReg) |
| `causal-jepa` | k left | predict EMA target embedding (cosine + VICReg) |

Each run writes `runs/<lang>-<variant>-d<dim>-s<seed>/model.onnx`
(`ids (batch, seq) int64 -> embeddings (batch, seq, dim)`), `vocab.json`, `config.json`,
`metrics.jsonl`, TensorBoard logs and `final_metrics.json`.
Design: `docs/superpowers/specs/2026-10-13-word2vec-variants-design.md`.

## Local (CPU)

```bash
uv sync --extra prep
uv run pytest
# tiny real-data pipeline (first 1000 Czech articles)
uv run python -m scripts.download --lang cs --tiny --data-dir data-tiny
uv run python -m scripts.preprocess --lang cs --shard 0 --num-shards 1 --workers 4 --data-dir data-tiny
uv run python -m scripts.build_vocab --lang cs --num-shards 1 --min-count 2 --data-dir data-tiny
uv run python -m scripts.encode --lang cs --pad 5 --data-dir data-tiny
uv run python -m scripts.fetch_benchmarks --lang cs --data-dir data-tiny
uv run python -m w2v.train --lang cs --mode causal --jepa --dim 64 --batch 8 --seq-len 256 \
    --max-steps 50 --data-dir data-tiny --runs-dir runs-tiny
```

## MetaCentrum

```bash
# on a frontend, inside your storage home (default /storage/brno12-cerit/home/$USER/w2v)
git clone <this repo> "$W2V_STORAGE/datalab" && cd "$W2V_STORAGE/datalab/word2vec"
$EDITOR jobs/config.sh             # set W2V_STORAGE
jobs/setup_env.sh                  # once: uv + locked environment on storage
jobs/submit_all.sh                 # download -> preprocess (array) -> vocab/encode -> 12 training jobs
DIMS="100 300 500" SKIP_PREP=1 jobs/submit_all.sh   # more sizes once data exists
qstat -u "$USER"
uv run --frozen --no-sync python -m scripts.compare --runs-dir "$W2V_RUNS" --out "$W2V_STORAGE/results"
```

If an upstream job fails, its dependents stay held (`qstat -u $USER` shows them as H); `qdel` them and resubmit.
A training job stops after `TRAIN_BUDGET` (default 50m) and resumes when resubmitted; `qsub -v ...,TRAIN_BUDGET=...` can change it alongside a longer walltime.
Training jobs resume from `ckpt.pt` when resubmitted (same variables). English is capped
to the larger of the Czech/Ukrainian token counts, so its vocab job waits for both.
Czech/Ukrainian analogy benchmarks keep only semantic categories (lemmatization removes
the morphological ones); there is no Ukrainian similarity set.
