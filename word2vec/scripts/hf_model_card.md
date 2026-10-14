---
language: [en, cs, uk]
license: cc-by-sa-4.0
library_name: onnx
tags: [word2vec, word-embeddings, onnx, jepa]
datasets: [wikimedia/wikipedia]
---

# word2vec variants on Wikipedia (English, Czech, Ukrainian)

Word embedding tables from a modernized word2vec trained in PyTorch for one epoch on Wikipedia 20231101.
Czech and Ukrainian text is lemmatized (UDPipe 1, plus pymorphy3 for Ukrainian); English is not.

The context of a word is aggregated with a fixed averaging convolution over the window, either
**bidirectional** (k words left and right) or **causal** (k words left only). Each context aggregation is trained with one of two objectives:

- **NS:** negative sampling, 1024 negatives shared across the batch.
- **JEPA:** predicts an EMA target encoder's embedding, with a cosine + VICReg loss.

Combining them gives four variants: `bidir`, `causal`, `bidir-jepa`, `causal-jepa`.

## Layout

```
<lang>-<variant>-d<dim>-w<window>-s0/
    model_int8.onnx         int8 table with per-row float32 scales (about half the size of fp16)
    model_fp16.onnx         float16 table
    config.json             training configuration
    final_metrics.json      benchmark scores of the fp16 table
    final_metrics_int8.json benchmark scores of the int8 table, only for the w5 runs
vocab/<lang>.json           {"words": [...]}: the row index of a word is its position in the list (200k words)
```

The grid covers:

- `lang`: en, cs, uk
- `dim`: 16, 64, 256, 300
- `window`: 5, 10, 20

That is 144 runs in total. int8 quantization changes benchmark scores by at most about 0.001.

## Usage

Both ONNX files have the same interface: input `ids` (int64, `[batch, n]`) and output
`embeddings` (float32, `[batch, n, dim]`).

```python
import json, numpy as np, onnxruntime as ort
words = json.load(open("vocab/en.json"))["words"]
sess = ort.InferenceSession("en-bidir-jepa-d256-w5-s0/model_int8.onnx")
ids = np.array([[words.index("king"), words.index("queen")]], dtype=np.int64)
vec = sess.run(None, {"ids": ids})[0][0]
```

In the browser (onnxruntime-web), the files can be fetched directly:

```js
const url = "https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk/resolve/main/en-bidir-jepa-d64-w5-s0/model_int8.onnx";
const session = await ort.InferenceSession.create(url);
```
