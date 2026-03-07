# who-are-you

Local reproduction scaffold for:

Logan Blue et al., "Who Are You (I Really Wanna Know)? Detecting Audio DeepFakes Through Vocal Tract Reconstruction", USENIX Security 2022.

The repository works directly with the prepared corpora in this workspace:

- `datasets/TIMIT`
- `datasets/generated_TIMIT`

Both corpora are expected to contain:

- `.wav`
- `.TXT`
- `.WRD`
- `.PHN`

The pipeline uses those existing alignments directly and does not call the paper authors' code.

## Install

```bash
uv sync
```

## Entry Point

```bash
uv run python main.py -h
```

## Available Commands

Show the default paper configuration:

```bash
uv run python main.py show-config
```

Show a simple tract reconstruction sanity check:

```bash
uv run python main.py demo-transfer
```

Show how in-word phoneme bigrams are constructed:

```bash
uv run python main.py demo-bigrams
```

Summarize the prepared local datasets:

```bash
uv run python main.py dataset-summary
```

Build a detector model:

```bash
uv run python main.py build-detector --model-path artifacts/detector_model.json
```

`build-detector` will also write one CSV per utterance under `artifacts/feature_values/` by default.
For example:

```bash
artifacts/feature_values/FAEM0/SA1__organic.csv
artifacts/feature_values/FAEM0/SA1__deepfake.csv
```

Change the output directory with:

```bash
uv run python main.py build-detector \
  --feature-output-dir artifacts/per_utterance_features \
  --model-path artifacts/detector_model.json
```

Evaluate a saved detector model:

```bash
uv run python main.py evaluate --model-path artifacts/detector_model.json --mode ideal
uv run python main.py evaluate --model-path artifacts/detector_model.json --mode range
```

## Fast Smoke Test

```bash
uv run python main.py build-detector \
  --sample-size 4 \
  --feature-speakers 2 \
  --max-sentence-pairs 1 \
  --coordinate-search-max-iterations 10 \
  --fft-bin-stride 4 \
  --min-precision 0.6 \
  --min-recall 0.6 \
  --model-path artifacts/smoke_model.json

uv run python main.py evaluate \
  --model-path artifacts/smoke_model.json \
  --mode ideal \
  --max-sentence-pairs 1
```

## Paper-Like Run

```bash
uv run python main.py build-detector \
  --sample-size 300 \
  --feature-speakers 51 \
  --model-path artifacts/paper_model.json

uv run python main.py evaluate \
  --model-path artifacts/paper_model.json \
  --mode ideal
```

This remains expensive because the vocal-tract estimator dominates runtime.

## Implementation Notes

- The numeric core is `numba` on top of `numpy`.
- The tract estimator follows the paper's concatenated-tube transfer-function setup and coordinate-search loop.
- `build-detector` writes one CSV per utterance, with one row per bigram window and flattened reflection / tract-area columns.
- The detector supports both:
  - `range` mode: compare against organic min/max ranges
  - `ideal` mode: compare against thresholded ideal features

## Practical Note

Exact paper metrics still depend on:

- the exact deepfake generation recipe
- the sampled speaker split
- the numerical details of the tract estimator
- alignment quality

This codebase reproduces the method end to end, but the final precision and recall will vary with those details.
