# Word2vec web explorer — design

Date: 2026-10-14
Status: approved in conversation (approach A, "very lightweight"), pending written-spec review

## Goal

A public, static GitHub Pages playground for the published word2vec grid
(<https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk>). A visitor picks a model, which is
downloaded on demand from the Hub, and then:

1. **Arithmetic editor** — types an expression like `king - man + woman`, sees the k nearest
   neighbours of the result and a small 2D PCA map of the terms, result and neighbours.
2. **2D projection** — defines an x and a y axis (each a word or an expression such as
   `woman - man`), lists words, and sees them placed in that plane.

Success: the page loads in a browser from `https://zhigamason.github.io/datalab/`, the default
model (en, d300) is usable within a few seconds of download finishing, a k-NN query over all
200k words returns in well under a second, and the bundle has **zero runtime dependencies**.

## Non-goals (lightweight on purpose)

- No UI framework, no chart library, no onnxruntime-web, no Web Worker.
- No autocomplete, no URL/state sharing, no explicit Cache Storage (HTTP cache only).
- No side-by-side model comparison, no metrics display.
- No background "frequent words" cloud in the projection.

Any of these can be added later; none is designed for now.

## Data source

Files are fetched from `https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk/resolve/main/`:

- `vocab/<lang>.json` — `{"words": [...]}` in row order (≈3.2 MB for en). Row index = frequency rank.
- `<run>/model_int8.onnx` — 4 MB (d16) … 61 MB (d300).

Run name: `<lang>-<mode>[-jepa]-d<dim>-w<window>-s0`, with lang ∈ {en, cs, uk},
mode ∈ {bidir, causal}, dim ∈ {16, 64, 256, 300}, window ∈ {5, 10, 20}. All 144 combinations exist.
The Hub sends `Access-Control-Allow-Origin` for the Pages origin (verified).

The ONNX file holds two initializers stored as `raw_data`:
`embeddings_int8` (INT8, `[V, d]`) and `row_scales` (FLOAT, `[V, 1]`, little-endian).
Dequantized vector of row i = `embeddings_int8[i] * row_scales[i]`.

## Architecture

`word2vec/web/` — Vite + TypeScript, dev dependencies only (`vite`, `typescript`, `vitest`).

| File | Responsibility |
|---|---|
| `index.html` | Static layout: model picker, editor panel, projection panel. |
| `src/onnx.ts` | `readInitializers(buf: ArrayBuffer): Map<string, {dims, dataType, raw: Uint8Array}>` — minimal protobuf walker (ModelProto → graph → initializer: name, dims, data_type, raw_data). Ignores everything else. |
| `src/model.ts` | `Model` class: `words`, `index: Map<string, number>`, `dim`, `vector(i): Float32Array` (dequantized), `invNorm: Float32Array` precomputed on load. `loadModel(run, lang, onProgress)` fetches vocab + ONNX with a streamed progress callback. |
| `src/expr.ts` | Parser for `[coef*]word (± [coef*]word)*`; returns `{terms: {word, coef}[]}` or an error naming the unknown/bad token. Evaluation: `Σ coef·unit(vec(word))` (terms are L2-normalized before summing, as in standard analogy evaluation). |
| `src/knn.ts` | `knn(model, query, k, exclude: Set<number>)` — cosine over all rows, int8 dot product × scale × invNorm, top-k via a small bounded min-heap. |
| `src/geom.ts` | `pca2(vectors)` (power iteration on the small centred set, 2 components), `projectAxes(vecs, x, y, orthogonalize)` (y Gram–Schmidt against x when on; then dot products with unit axes). |
| `src/plot.ts` | Hand-written SVG scatter: points with labels, optional arrows, auto-fit with padding, hover title. Shared by both panels. |
| `src/main.ts` | Wires DOM events to the above. |

`.github/workflows/pages.yml` (repo root) — on push to `main` touching `word2vec/web/**`:
`npm ci && npm test && npm run build` in `word2vec/web`, upload `dist/` with
`actions/upload-pages-artifact`, deploy with `actions/deploy-pages`. Vite `base: '/datalab/'`.

## UI behaviour

**Model picker.** Four `<select>`s: language (en/cs/uk), objective (NS/JEPA), direction
(bidir/causal), dim (16/64/256/300, sizes shown), window (5/10/20). Default
`en-bidir-d300-w5`. "Load" button; progress bar showing MB downloaded. Vocab is reused when only
the run changes within a language. Panels are disabled until a model is loaded.

**Arithmetic editor.** A text input, `k` number input (default 10), "Go" (also on Enter).
Unknown words are reported in an inline error message. Output:
- Neighbour list: word + cosine (3 decimals). Input words are excluded. Clicking a neighbour
  appends `+ word` to the expression.
- PCA map: points for each term word, the result (marked distinctly) and the k neighbours,
  PCA fitted on all of them; arrows from origin-term chaining through the signed terms to the
  result (`king → king−man → king−man+woman`), drawn from the summed unit vectors.

**2D projection.** Inputs: x-axis expression, y-axis expression (each a single word or an
expression, evaluated with `expr.ts` and normalized), a textarea of words (whitespace/comma
separated), "orthogonalize y against x" checkbox (default on), "Plot". Unknown words are listed
and skipped. Each word is plotted at `(⟨unit(v), x̂⟩, ⟨unit(v), ŷ⟩)` with its label; axis titles
show the expressions. Defaults for en: x = `woman - man`, y = `rich - poor`, words =
`king queen prince princess doctor nurse engineer teacher beggar billionaire`.
For cs/uk the defaults are equivalent lemmas (cs: `žena - muž`, `bohatý - chudý`; uk:
`жінка - чоловік`, `багатий - бідний`) with a short word list.

## Error handling

- Fetch failure / non-200: message in the status area with the URL; picker stays enabled.
- Parse errors and unknown words: inline message under the relevant input; nothing else changes.
- Zero-norm axis (e.g. `man - man`): inline error.
- All vocabs are lowercase (verified for en), so input words are lowercased before lookup.
- Special tokens (`<pad>`, `<unk>`, `<num>`, any `<…>`) are never returned as neighbours.

## Testing

Vitest, no browser needed:
- `onnx.ts`: parse a tiny ONNX fixture (a few-row int8 table generated by a short Python
  script in `scripts/`, committed as `web/test/fixtures/tiny.onnx`) and check names, dims and
  bytes; truncated buffer throws.
- `expr.ts`: coefficients, signs, whitespace, leading minus, errors on bad tokens / unknown words.
- `knn.ts`: on a hand-built model, matches a brute-force float reference; exclusion works.
- `geom.ts`: PCA recovers the principal axis of a synthetic set; orthogonalization makes axes
  orthogonal; projection of an axis word onto itself is 1.

Manual check before merging: `npm run dev`, load en-d16 and en-d300, run
`king - man + woman` (expect `queen` near the top for d300) and the default projection.
