---
language: [en, cs, uk]
license: cc-by-sa-4.0
library_name: onnx
tags: [word2vec, word-embeddings, onnx, jepa]
datasets: [wikimedia/wikipedia]
---

# word2vec variants on Wikipedia (English, Czech, Ukrainian)

Word embedding tables from a modernized word2vec trained in PyTorch for five epochs on Wikipedia 20231101.
Czech and Ukrainian text is lemmatized (UDPipe 1, plus pymorphy3 for Ukrainian); English is not.

The context of a word is aggregated with a fixed averaging convolution over the window, either
**bidirectional** (k words left and right) or **causal** (k words left only). Each context aggregation is trained with one of two objectives:

- **NS:** negative sampling, 1024 negatives shared across the batch.
- **JEPA:** predicts an EMA target encoder's embedding, with a cosine + VICReg loss; VICReg covers both the
  predictor output and the embedding table (`--lambda-emb 0.3`).

Combining them gives four variants: `bidir`, `causal`, `bidir-jepa`, `causal-jepa`.

## Layout

```
<lang>-<variant>-d<dim>-w<window>-s0/
    model_int8.onnx         int8 table with per-row float32 scales
    config.json             training configuration
    final_metrics.json      benchmark scores of the int8 table
    metrics.jsonl           training log, with benchmark scores every 1000 steps
e5-jepa-lemb0/<run>/        the 72 JEPA runs with the original objective (collapsed; for reference)
vocab/<lang>.json           {"words": [...]}: the row index of a word is its position in the list (200k words)
```

Earlier versions of this repo held 1-epoch tables (with float16 copies) at the root and the 5-epoch ones under
`e5/`; they remain in the git history.

The grid covers:

- `lang`: en, cs, uk
- `dim`: 16, 64, 256, 300
- `window`: 5, 10, 20

That is 144 runs. int8 quantization changes benchmark scores by at most about 0.001.

## Which tables to use

Five epochs instead of one help negative sampling a lot: mean analogy accuracy for en `bidir` rises from 0.085 to
0.400, and the best English table, `en-bidir-d300-w5-s0`, reaches 0.616 on the Google analogies (3CosAdd over the full
200k vocabulary, so not comparable to figures restricted to the top 30k words).

The original JEPA variants collapse: the embedding rows drift toward one shared direction (mean cosine 0.83
between random words), because VICReg only regularized the predictor output; they are kept in `e5-jepa-lemb0/`.
The JEPA runs at the root apply the VICReg terms to the embedding table as well (`--lambda-emb 0.3`). This fixes the collapse and roughly doubles en
`bidir-jepa` analogy accuracy (0.098 to 0.211), but JEPA remains well below negative sampling, and its accuracy
stops improving after the first epoch.

| best analogy accuracy | en | cs | uk |
|---|---|---|---|
| NS | 0.616 | 0.289 | 0.442 |
| JEPA | 0.307 | 0.200 | 0.149 |
| JEPA, original objective, `e5-jepa-lemb0/` | 0.201 | 0.175 | 0.104 |

## Usage

All ONNX files have the same interface: input `ids` (int64, `[batch, n]`) and output
`embeddings` (float32, `[batch, n, dim]`).

```python
import json, numpy as np, onnxruntime as ort
words = json.load(open("vocab/en.json"))["words"]
sess = ort.InferenceSession("en-bidir-d256-w5-s0/model_int8.onnx")
ids = np.array([[words.index("king"), words.index("queen")]], dtype=np.int64)
vec = sess.run(None, {"ids": ids})[0][0]
```

In the browser (onnxruntime-web), the files can be fetched directly:

```js
const url = "https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk/resolve/main/en-bidir-d64-w5-s0/model_int8.onnx";
const session = await ort.InferenceSession.create(url);
```
