# Public inference and evaluation scripts

Run commands from the repository root. There are no public training or distillation entry points.

## Released-model generation

```bash
python3 scripts/generate.py "What is the capital of France?" --quality 0.5
python3 scripts/generate.py --help
```

`generate.py` loads the released checkpoint locally or from Hugging Face, rebuilds its recorded
architecture, and strictly loads model weights. The default backend is selected from the available
CUDA, MLX, MPS, or CPU path.

## MLX sampling and parity

```bash
pip install -e ".[mlx]"
python3 scripts/sample_mlx.py --help
python3 scripts/verify_mlx_model.py
```

See `docs/BACKENDS.md` for supported checkpoint formats, precision caveats, and measured results.

## Evaluation

```bash
python3 scripts/evaluate.py --help
python3 scripts/eval_vs_smollm.py --help
python3 scripts/perplexity_eval.py --help
```

DIMBA denoising-reconstruction perplexity is not autoregressive perplexity. Preserve the caveats
printed and documented by the evaluation scripts.

## Dependency-light CPU benchmark

```bash
python3 scripts/benchmark.py
```

This constructs a tiny model and is a local regression/smoke benchmark, not release-model or CUDA
performance evidence.

## Explicit release upload

```bash
python3 scripts/upload_to_hf.py --help
```

The uploader requires an explicit artifacts directory, repository id, and Hugging Face token. It
is never invoked by inference or tests. Do not commit tokens or generated artifacts.
