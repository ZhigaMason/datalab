# Word2vec variants: bidirectional / causal × softmax-NS / JEPA — design

Date: 2026-10-13
Status: approved in conversation, pending written-spec review

## Goal

Run controlled, modernized word2vec experiments in **English, Czech, Ukrainian**
comparing four training objectives:

| Variant         | Context         | Objective                                   |
|-----------------|-----------------|---------------------------------------------|
| `bidir`         | k left + k right | negative sampling over vocab               |
| `causal`        | k left only      | negative sampling over vocab               |
| `bidir-jepa`    | k left + k right | latent prediction of EMA target embedding  |
| `causal-jepa`   | k left only      | latent prediction of EMA target embedding  |

Each run produces an **ONNX artifact** (embedding lookup) plus metadata.
Training runs on **MetaCentrum GPUs** via PBS; development and testing happen on a
local CPU-only machine.

Success: all 12 (3 languages × 4 variants) runs train end-to-end from a single
submit script, export a validated ONNX file, and produce a comparison table of
intrinsic evaluation metrics. Embedding size is selectable (`--dim`).

## Non-goals

- Subword models, contextual (transformer) encoders, downstream task evaluation.
- Learned convolution kernels (the convolution is a fixed averaging kernel).
- Full-softmax or sampled-softmax objectives.

## 1. Data pipeline

**Source:** HF `wikimedia/wikipedia`, snapshot `20231101`, configs `en`, `cs`, `uk`.

**Steps** (separate, resumable scripts writing into `data/<lang>/`):

1. `scripts/download.py` — stream articles to parquet shards.
2. `scripts/preprocess.py` — per shard (CPU PBS array job):
   - clean (drop very short lines / leftover markup), sentence split;
   - **en:** Unicode word regex tokenizer, lowercase (no lemmatization);
   - **cs / uk:** UDPipe 1 (`ufal.udpipe`) with UD 2.x models (`czech-pdt`,
     `ukrainian-iu`) for tokenization + lemmatization; lowercase lemmas;
   - drop punctuation; numbers → `<num>`;
   - output one sentence per line (plain text, inspectable).
3. `scripts/build_vocab.py` — count tokens; keep `min_count ≥ 5`, cap at top
   200k (both configurable); write `vocab.json` with counts.
   Reserved ids: `0 = <pad>`, `1 = <unk>`.
4. `scripts/encode.py` — encode to a flat `uint32` memmap `tokens.bin`, inserting
   **k `<pad>` tokens between sentences** (k = window radius) so no context window
   can span a sentence boundary.

**Token budget:** English is capped so its encoded token count equals
`max(tokens_cs, tokens_uk)`, measured after preprocessing. Actual counts are
recorded in `data/<lang>/stats.json`.

**Loader:** `TokenStream` samples random contiguous chunks of length `T` from the
memmap. Frequent-word subsampling (word2vec `p_drop = 1 − sqrt(t/f)`,
`t = 1e-4`) is applied on GPU per batch by **masking** dropped tokens (excluded as
both context and target) rather than removing them, keeping tensor shapes static.

**Tiny mode:** `--tiny` limits each language to ~1k articles so the full pipeline
runs on a laptop CPU in minutes.

## 2. Model and objectives

Single module `W2V(V, dim, k, mode ∈ {bidir, causal}, jepa: bool)`.

### Shared context encoder (fixed convolutional averaging)

- `E = nn.Embedding(V, dim, padding_idx=0)`, init `U(−0.5/dim, 0.5/dim)`.
- Non-trainable normalized kernel buffer of length `2k+1`:
  - bidir: `([1]*k + [0] + [1]*k) / 2k`
  - causal: `([1]*k + [0] + [0]*k) / k` (position t sees only t−k … t−1)
- With mask `m` (1 = real, not pad, not subsampled):
  ```
  ctx = conv1d((E(x) * m).transpose(1,2), kernel.expand(dim,1,-1), padding=k, groups=dim)
  ```
- The normalizer is fixed (closed form), not the count of valid context tokens.
  Deliberate departure from CBOW: near sentence boundaries and where subsampling
  masked words, `ctx` is scaled down rather than renormalized.
- Loss positions: `m[t] == 1` (target is a real, non-subsampled token) **and** the
  position has at least one valid context token, computed as
  `conv1d(m, (kernel > 0).float(), padding=k) > 0`. Masked targets (incl. sentence
  `<pad>`s) and context-less positions contribute nothing to the loss.

### Non-JEPA variants (negative sampling)

- `out = nn.Linear(dim, V, bias=False)`, zero-initialized (word2vec style).
- Per step draw K = 1024 negatives from unigram^0.75, shared across the batch.
- `loss_t = −logσ(ctx_t · w_{x_t}) − (5/K) · Σ_{n∈K} logσ(−ctx_t · w_n)`
  (scaled to match word2vec's expectation with 5 negatives); mean over loss positions.

### JEPA variants

- Predictor: `P = nn.Linear(dim, dim)`; `pred_t = P(ctx_t)`.
- Target encoder: `E_ema`, an EMA copy of the context encoder `E`, no gradient.
  After each optimizer step: `E_ema ← τ·E_ema + (1−τ)·E`, τ cosine-scheduled
  0.996 → 1.0 over training.
- Loss:
  `(1 − cos(pred_t, E_ema(x_t))).mean() + λ_var·var_loss(pred) + λ_cov·cov_loss(pred)`
  (VICReg terms on a random subsample of loss positions; defaults
  `λ_var = 1.0`, `λ_cov = 0.04`).
  Since 2026-10-16 the loss also adds `λ_emb·(λ_var·var_loss(E(x_t)) + λ_cov·cov_loss(E(x_t)))` on the
  online embeddings of the same positions, default `λ_emb = 0.3` (`--lambda-emb`). Without it the table
  collapsed toward one direction (mean row cosine 0.83) and analogy accuracy plateaued; grids trained before
  this change used `λ_emb = 0`.
- Collapse monitor logged each log interval: mean per-dimension std of `E` rows
  in the batch and of `pred`. If std < 1e-3, a `collapse_warning` is written to
  metrics; the run continues and still exports.

### Exported embedding

For all variants the exported table is the context encoder `E`.

## 3. Training and evaluation

### Training — `python -m w2v.train --lang cs --mode causal [--jepa] --dim 300 --seed 0`

- Batch: B = 64 × T = 1024 tokens; same seed ⇒ identical data order across variants.
- bf16 autocast on CUDA; `torch.compile` when available (flag to disable).
- AdamW, LR 2e-3, 2% linear warmup, cosine decay to 0; weight decay 0 on `E` and
  `out`, 0.01 on predictor.
- Default 1 epoch over the encoded corpus (`--epochs` configurable).
- Hyperparameters exposed as CLI flags: `--dim` (default 300), `--window` (k, default
  5), `--neg` (K), `--subsample-t`, `--lr`, `--batch`, `--seq-len`, `--epochs`,
  `--ema-start`, `--lambda-var`, `--lambda-cov`, `--eval-every`.
- Run directory: `runs/<lang>-<mode>[-jepa]-d<dim>-w<window>-s<seed>/`
  - `metrics.jsonl` (loss, lr, tokens/s, collapse monitors, periodic eval)
  - TensorBoard events
  - `ckpt.pt` saved periodically; automatic resume if present.
- On completion: `model.onnx`, `vocab.json`, `config.json`, `final_metrics.json`,
  optional `vectors.vec`.

### ONNX export — `w2v/export.py`

- Graph: `ids (B, T) int64 → embeddings (B, T, dim) float32`, dynamic B and T.
- Table storage: int8 with per-row symmetric scales, `embeddings_int8 [V, dim]` and
  `row_scales [V, 1] float32` with scale = row absmax / 127. All-zero rows (`<pad>`) get scale 0. The
  graph is `gather(q).float() * gather(scale)`. It is a single file with no external data, so
  `V·dim + 4V` must stay under the 2 GB protobuf limit.
- Post-export check: onnxruntime output vs the dequantized table, `atol = 1e-5`; NaN or
  a mismatch aborts. `final_metrics.json` is evaluated on the stored (quantized) table.
- Revision history: float32 (initial), then float16 (`e8d683e`), then int8 (`3a8f3f5`). The int8 rows keep a mean
  cosine of 0.99997 with fp16, and benchmark deltas are ≤ 0.001. `scripts/quantize_onnx.py` converts earlier
  float16 runs in place and keeps `model_fp16.onnx`.

### Evaluation — `w2v/evaluate.py` (operates on the ONNX artifact)

- **en:** SimLex-999, WS-353, MEN (Spearman ρ); Google analogy (3CosAdd acc@1,
  search restricted to the vocab).
- **cs:** Czech analogy corpus (Svoboda & Brychcín 2016), **semantic categories
  only**, items lemmatized with the same UDPipe model; Czech word-similarity set
  if a maintained one is found.
- **uk:** lang-uk analogy set, semantic categories only, lemmatized items;
  Ukrainian similarity set if available.
- Morphological analogy categories are excluded for cs/uk because lemmatization
  erases the distinctions they test.
- Benchmark OOV rate is reported alongside every score.
- `scripts/fetch_benchmarks.py` downloads benchmarks into `data/benchmarks/`;
  exact sources are verified during implementation.
- `scripts/compare.py` aggregates all `final_metrics.json` into
  `results/summary.{csv,md}` with columns lang × variant × dim × seed × metric.

### Testing (local CPU, pytest)

- Conv averaging matches a naive Python loop computing the closed-form
  (fixed-normalizer) window sum (both kernels, edges, masks).
- Causal leakage: perturbing tokens ≥ t does not change `ctx_t`.
- Loss-position mask: masked targets and context-less positions (chunk edges,
  sentence pads) are excluded.
- Negative-sampling loss matches a hand-computed reference.
- EMA update correctness and no gradient into `E_ema`.
- ONNX export parity.
- End-to-end smoke test: `--tiny` data → train a few steps per variant → export →
  evaluate, all on CPU.

## 4. MetaCentrum execution

- Environment: managed by **uv** from `pyproject.toml` + committed `uv.lock`; no
  container. `torch` is pinned to the PyTorch **cu126** wheel index (PyPI's Linux
  wheels target CUDA 13 and need driver ≥ 580; cu126 runs on older MetaCentrum
  drivers). The same lock works on the local CPU machine and on GPU nodes.
  - Core deps: `torch`, `numpy`, `onnx`, `onnxruntime`, `tensorboard`.
  - Optional extra `prep`: `ufal.udpipe`, `datasets`, `pyarrow` (CPU prep jobs only).
  - Dependency group `dev`: `pytest`.
- `jobs/setup_env.sh` (run once on a frontend): installs uv into the storage home
  if missing, sets `UV_CACHE_DIR` on storage, and runs `uv sync --frozen --extra prep`
  creating `.venv` on storage.
- PBS jobs invoke `uv run --frozen python -m …` (or the module/script directly).
- `jobs/config.sh` — the only user-edited file: storage root
  (e.g. `/storage/brno2/home/$USER/w2v`), uv cache location, defaults.
- Jobs:
  - `download.pbs` — CPU, one per lang: Wikipedia parquet + UDPipe model + benchmarks.
  - `prep.pbs` — CPU array job, one index per shard of a language: preprocess.
  - `vocab.pbs` — CPU, one per lang: build vocab + encode.
  - `train.pbs` — 1 GPU (`gpu_mem≥20gb`), ~4 h walltime; stages `tokens.bin` and
    vocab to `$SCRATCHDIR`, runs training with env vars LANG/MODE/JEPA/DIM/SEED,
    copies the run directory back to storage (also on failure via `trap`).
- `jobs/submit_all.sh` — submits the grid {en, cs, uk} × {4 variants} ×
  `DIMS` (default `"300"`) × `WINDOWS` (default `"5"`) × `SEEDS` (default `"0"`), chaining with
  `qsub -W depend=afterok:` so training waits for data preparation. Training runs on
  `gpu_dgx` (`train_dgx.pbs`, two lanes on 2 GPUs, `RUNS_PER_JOB` runs per job) by default.
  The 2026-10 grid is dims {16, 64, 256, 300} × windows {5, 10, 20}, 144 runs, encoded
  with `W2V_PAD = 20`.
- Robustness: scripts validate inputs and fail loudly; prep steps are idempotent
  (skip finished shards via marker files); training resumes from `ckpt.pt`.

## 5. Publishing

- `scripts/hf_upload.sh` stages every run of the grid as hard links (`model_int8.onnx`,
  `model_fp16.onnx`, `config.json`, `final_metrics*.json`), plus one `vocab/<lang>.json` per
  language and the model card `scripts/hf_model_card.md`. It uploads them with
  `hf upload-large-folder` to the public model repo `mkkhavil/word2vec-wiki-cs-en-uk`.
- Intended consumer: a static GitHub Pages demo running onnxruntime-web. It fetches the
  selected table from the Hub at runtime, since the Hub serves CORS headers. The current
  graph is lookup-only; nearest-neighbour and analogy search would need a second exported
  graph (query · tableᵀ → top-k), which is not built yet.

## Repository layout

```
word2vec/
  w2v/              data.py, model.py, losses.py, train.py, export.py, evaluate.py
  scripts/          download.py, preprocess.py, build_vocab.py, encode.py,
                    fetch_benchmarks.py, compare.py, quantize_onnx.py,
                    hf_upload.sh, hf_model_card.md
  jobs/             config.sh, setup_env.sh, download.pbs, prep.pbs, vocab.pbs, train.pbs,
                    train_dgx.pbs, train_run.sh, submit_all.sh
  tests/            unit tests + tiny end-to-end smoke test
  pyproject.toml    managed by uv
  README.md
```

## Open items to verify during implementation

- Resolved 2026-10-13: Czech similarity = WordSim353-cs (LINDAT 11234/1-1713);
  no Ukrainian similarity set found, so uk is evaluated on analogies only. UDPipe
  models = UD 2.5 `czech-pdt-ud-2.5-191206`, `ukrainian-iu-ud-2.5-191206`
  (LINDAT 11234/1-3131, new DSpace bitstream URLs).
- cu126 `torch` wheel compatibility with MetaCentrum GPU node drivers (check `nvidia-smi` in the first train job).
- Actual per-language token counts after preprocessing (sets the English cap).
