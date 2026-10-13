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
- Non-trainable kernel buffer of length `2k+1`:
  - bidir: `[1]*k + [0] + [1]*k`
  - causal: `[1]*k + [0] + [0]*k` (position t sees only t−k … t−1)
- With mask `m` (1 = real, not pad, not subsampled):
  ```
  ctx_sum = conv1d((E(x) * m).transpose(1,2), kernel.expand(dim,1,-1), padding=k, groups=dim)
  Z       = conv1d(m, kernel, padding=k)
  ctx     = ctx_sum / Z.clamp(min=1)
  ```
  i.e. CBOW averaging with normalizer `Z` = number of valid context tokens.
- Loss positions: `m[t] == 1` and `Z[t] > 0`.

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
- Run directory: `runs/<lang>-<mode>[-jepa]-d<dim>-s<seed>/`
  - `metrics.jsonl` (loss, lr, tokens/s, collapse monitors, periodic eval)
  - TensorBoard events
  - `ckpt.pt` saved periodically; automatic resume if present.
- On completion: `model.onnx`, `vocab.json`, `config.json`, `final_metrics.json`,
  optional `vectors.vec`.

### ONNX export — `w2v/export.py`

- Graph: `ids (B, T) int64 → embeddings (B, T, dim) float32`, dynamic B and T.
- Post-export check: onnxruntime output vs PyTorch, `atol = 1e-5`; failure aborts.

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

- Conv averaging matches a naive Python window loop (both kernels, edges, masks).
- Causal leakage: perturbing tokens ≥ t does not change `ctx_t`.
- `Z` / mask handling at chunk edges and sentence pads.
- Negative-sampling loss matches a hand-computed reference.
- EMA update correctness and no gradient into `E_ema`.
- ONNX export parity.
- End-to-end smoke test: `--tiny` data → train a few steps per variant → export →
  evaluate, all on CPU.

## 4. MetaCentrum execution

- Environment: NGC PyTorch Singularity image from
  `/cvmfs/singularity.metacentrum.cz/NGC/`; extra packages
  (`ufal.udpipe`, `onnx`, `onnxruntime`, `datasets`, `tensorboard`) installed once
  into a `PYTHONUSERBASE` on storage by `jobs/setup_env.sh`.
- `jobs/config.sh` — the only user-edited file: storage root
  (e.g. `/storage/brno2/home/$USER/w2v`), image path, defaults.
- Jobs:
  - `prep.pbs` — CPU array job, one index per (lang, shard): download + preprocess.
  - `vocab.pbs` — CPU, one per lang: build vocab + encode.
  - `train.pbs` — 1 GPU (`gpu_mem≥20gb`), ~4 h walltime; stages `tokens.bin` and
    vocab to `$SCRATCHDIR`, runs training with env vars LANG/MODE/JEPA/DIM/SEED,
    copies the run directory back to storage (also on failure via `trap`).
- `jobs/submit_all.sh` — submits the grid {en, cs, uk} × {4 variants} ×
  `DIMS` (default `"300"`) × `SEEDS` (default `"0"`), chaining with
  `qsub -W depend=afterok:` so training waits for data preparation.
- Robustness: scripts validate inputs and fail loudly; prep steps are idempotent
  (skip finished shards via marker files); training resumes from `ckpt.pt`.

## Repository layout

```
word2vec/
  w2v/        data.py, model.py, losses.py, train.py, export.py, evaluate.py
  scripts/    download.py, preprocess.py, build_vocab.py, encode.py,
              fetch_benchmarks.py, compare.py
  jobs/       config.sh, setup_env.sh, prep.pbs, vocab.pbs, train.pbs, submit_all.sh
  tests/      unit tests + tiny end-to-end smoke test
  requirements.txt
  README.md
```

## Open items to verify during implementation

- Availability/licensing of Czech and Ukrainian word-similarity datasets.
- Exact UDPipe 1 UD 2.x model filenames/download URL.
- Latest NGC PyTorch image name on MetaCentrum CVMFS.
- Actual per-language token counts after preprocessing (sets the English cap).
