# who-are-you

Local reproduction scaffold for:

Logan Blue et al., "Who Are You (I Really Wanna Know)? Detecting Audio DeepFakes Through Vocal Tract Reconstruction", USENIX Security 2022.

The repository is set up to work directly with the prepared corpora already present in this workspace:

- `datasets/TIMIT`
- `datasets/generated_TIMIT`

Both corpora contain:

- `.wav`
- `.TXT`
- `.WRD`
- `.PHN`

This means the pipeline can use the provided phoneme and word alignments directly instead of running Gentle.

## Entry Point

Run everything through:

```bash
uv run python main.py -h
```

Check which runtime backend will be used:

```bash
uv run python main.py runtime-info
uv run python main.py runtime-info --device cpu
uv run python main.py runtime-info --device cuda
uv run python main.py runtime-info --device mps
```

## Runtime Backend

The numeric core now uses `PyTorch` only.

- `cpu`
- `cuda`
- `mps`

`--device auto` resolves in this order:

- `cuda` when CUDA is available
- otherwise `mps` when MPS is available
- otherwise `cpu`

Install dependencies into the `uv` environment:

```bash
uv sync
```

## Available Commands

Show the default paper configuration:

```bash
uv run python main.py show-config
```

Summarize the locally prepared datasets:

```bash
uv run python main.py dataset-summary
```

Build a detector model:

```bash
uv run python main.py build-detector --model-path artifacts/detector_model.json
```

Evaluate a saved detector model:

```bash
uv run python main.py evaluate --model-path artifacts/detector_model.json --mode ideal
uv run python main.py evaluate --model-path artifacts/detector_model.json --mode range
```

Explicitly choose CPU, CUDA, or MPS:

```bash
uv run python main.py build-detector --device cpu --model-path artifacts/detector_model.json
uv run python main.py build-detector --device cuda --model-path artifacts/detector_model.json
uv run python main.py build-detector --device mps --model-path artifacts/detector_model.json
uv run python main.py evaluate --device cuda --model-path artifacts/detector_model.json --mode ideal
```

## Fast Smoke Test

This is the quickest end-to-end run on the local datasets:

```bash
uv run python main.py build-detector \
  --device auto \
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

The paper uses a much more expensive setup. Start with:

```bash
uv run python main.py build-detector \
  --device auto \
  --sample-size 300 \
  --feature-speakers 51 \
  --model-path artifacts/paper_model.json

uv run python main.py evaluate \
  --model-path artifacts/paper_model.json \
  --mode ideal
```

This will be slow because the vocal tract estimator dominates runtime.

## Implementation Notes

- The pipeline works from local prepared alignments and does not call the paper authors' code.
- The tract estimator uses the paper's concatenated-tube transfer function and a coordinate-search optimizer.
- The numerical core uses `PyTorch` and supports `cpu`, `cuda`, and `mps`.
- The detector supports both:
  - `range` mode: compare against organic min/max ranges
  - `ideal` mode: compare against thresholded ideal features

## Important Practical Note

Exact reproduction of the paper's reported metrics depends on:

- the exact deepfake generation recipe
- the speaker split
- numerical details of the tract estimator
- alignment quality

This codebase reproduces the method end-to-end, but the final precision and recall will depend on those details.
