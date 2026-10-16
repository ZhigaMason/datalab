# word2vec variants: bidirectional / causal × negative sampling / JEPA

Word2vec on Wikipedia (en, cs, uk — cs/uk lemmatized with UDPipe) with a fixed
convolutional context average, trained four ways:

| variant | context | objective |
|---|---|---|
| `bidir` | k left + k right | negative sampling |
| `causal` | k left | negative sampling |
| `bidir-jepa` | k left + k right | predict EMA target embedding (cosine + VICReg) |
| `causal-jepa` | k left | predict EMA target embedding (cosine + VICReg) |

JEPA applies the VICReg terms to the predictor output and, with weight `--lambda-emb` (default 0.3), to the
embedding table itself. Without the second part (`--lambda-emb 0`, the objective of the first grids) the table
drifts toward one shared direction (mean cosine 0.83 between random words) and analogy accuracy stalls; with it,
5-epoch en `bidir-jepa` analogy accuracy roughly doubles (0.098 → 0.211 mean over the grid). Removing the linear
predictor (`--predictor identity`) makes the table collapse completely.

Each run writes `runs/<lang>-<variant>-d<dim>-w<window>-s<seed>/model.onnx`
(`ids (batch, seq) int64 -> embeddings (batch, seq, dim) float32`), `vocab.json`, `config.json`,
`metrics.jsonl`, TensorBoard logs and `final_metrics.json`.
Design: `docs/superpowers/specs/2026-10-13-word2vec-variants-design.md`.

## ONNX artifact

`model.onnx` holds only the input embedding table and a lookup. The table is stored as
int8 (initializer `embeddings_int8`, `[V, dim]`). Each row also has a float32 scale (`row_scales`, `[V, 1]`), which is the row's
absolute maximum divided by 127. The graph dequantizes the looked-up rows, so the output is float32. `final_metrics.json` is computed on
this stored table. The file is a little over a quarter of the float32 size (sizes for one table):

| dim (V = 200k) | int8 | fp16 | fp32 |
|---|---|---|---|
| 16 | 4.0 MB | 6.4 MB | 12.8 MB |
| 64 | 13.6 MB | 25.6 MB | 51.2 MB |
| 256 | 52.0 MB | 102.4 MB | 204.8 MB |
| 300 | 60.8 MB | 120.0 MB | 240.0 MB |

The rows keep a mean cosine of 0.99997 with float16, and benchmark scores move by at most about 0.001.
General-purpose compressors gain little on top: zstd −9 %, xz −13 %, and byte-shuffle + zstd −20 % of fp16.

Runs exported before int8 (the 2026-10 grid, stored as float16) keep their float16 `final_metrics.json` and are
converted in place:

```bash
# keeps the original as model_fp16.onnx; --eval writes final_metrics_int8.json and prints the largest score change
python -m scripts.quantize_onnx --runs-dir "$W2V_RUNS" --glob '*-w5-*' --eval --data-dir "$W2V_DATA"
python -m scripts.quantize_onnx --runs-dir "$W2V_RUNS"          # the rest, without re-evaluation
```

`scripts.quantize_onnx` skips runs that are already int8. It verifies each new file against the dequantized table
(atol 1e-5) before it replaces `model.onnx`.

## Hugging Face

The grid is published at <https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk>. The repo has one directory per run
with `model_int8.onnx`, `model_fp16.onnx`, `config.json` and `final_metrics*.json`. It also has one `vocab/<lang>.json` per language,
with words in row order. The model card is `scripts/hf_model_card.md`. To publish, log in with `hf auth login` on a
MetaCentrum node, then run:

```bash
PYTHONUNBUFFERED=1 bash scripts/hf_upload.sh mkkhavil/word2vec-wiki-cs-en-uk 2>&1 | tee "$W2V_STORAGE/hf_upload.log"
```

The script stages hard links in `$W2V_STORAGE/hf_stage` and runs `hf upload-large-folder`, which resumes when re-run.
`PYTHONUNBUFFERED=1` is needed because the status reports otherwise stay buffered behind `tee`. If the upload sends nothing for minutes,
re-run it with `HF_HUB_DISABLE_XET=1`. To check progress, count the ONNX files on the Hub:
`curl -s https://huggingface.co/api/models/<repo> | grep -o '\.onnx"' | wc -l`.

The repo is public, so a static page such as GitHub Pages can load a table with onnxruntime-web straight from
`https://huggingface.co/<repo>/resolve/main/<run>/model_int8.onnx`; the Hub sends CORS headers. GitHub Release
assets don't send CORS headers, and Pages doesn't serve Git LFS files, so neither works for this.

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
jobs/submit_all.sh                 # download -> preprocess (array) -> vocab/encode -> one 2-GPU gpu_dgx training job per language
DIMS="16 64 256 300" WINDOWS="5 10 20" SKIP_PREP=1 jobs/submit_all.sh   # the 2026-10 grid, once data exists
VOCAB_ONLY=1 jobs/submit_all.sh    # only re-encode existing text, e.g. after raising W2V_PAD
qstat -u "$USER"
uv run --frozen --no-sync python -m scripts.compare --runs-dir "$W2V_RUNS" --out "$W2V_STORAGE/results"
```

If an upstream job fails, its dependents stay held (`qstat -u $USER` shows them as H); `qdel` them and resubmit.
Training goes to `gpu_dgx` by default (`jobs/train_dgx.pbs`: the queue requires >= 2 GPUs, so each language's runs share one 2-GPU job, one run per GPU at a time); `GPU_QUEUE=gpu jobs/submit_all.sh` submits one 1-GPU `jobs/train.pbs` job per run instead. Both use `jobs/train_run.sh` per run. Each run stops after `TRAIN_BUDGET` (default 50m) and resumes when resubmitted; `qsub -v ...,TRAIN_BUDGET=...` can change it alongside a longer walltime.
Training jobs resume from `ckpt.pt` when resubmitted (same variables). English is capped
to the larger of the Czech/Ukrainian token counts, so its vocab job waits for both.
Czech/Ukrainian analogy benchmarks keep only semantic categories (lemmatization removes
the morphological ones); there is no Ukrainian similarity set.
Analogy accuracy uses 3CosAdd over the full vocabulary (not the common top-30k restriction), so numbers are not
directly comparable to published Google-analogy figures. After a resume, `metrics.jsonl` may contain repeated
steps; keep the last record per step.

## Web explorer

`web/` is a static page (<https://zhigamason.github.io/datalab/>) that downloads one int8 table
from the Hugging Face repo on demand and runs everything in the browser: word arithmetic with
k nearest neighbours and a PCA map, and a 2D projection onto two word/expression axes. It reads
`embeddings_int8` and `row_scales` straight from `model_int8.onnx` with a small protobuf reader,
so it needs no ONNX runtime. It has no runtime dependencies.

```bash
cd web
npm install
npm run dev                          # http://localhost:5173/datalab/
npm test                             # unit tests; LIVE=1 npm test also checks the published en d300 model
uv run python -m scripts.make_web_fixture   # from word2vec/, regenerates test/fixtures/tiny.onnx
```

`.github/workflows/pages.yml` builds and deploys it on every push to `main` that touches `word2vec/web/`.
One-time setup: in the GitHub repo settings, under Pages → Build and deployment → Source, choose **GitHub Actions**.
