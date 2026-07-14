# Running DIMBA on a Mac — CPU, Apple GPU (MPS), and MLX

`dimbapeare1-30m` was trained on a CUDA GPU with the `mamba_ssm` Mamba-2 kernel. That kernel
is CUDA-only, so for a long time the checkpoint could not run on a Mac at all. This document
describes the three backends that now run it locally — and how fast each is.

**TL;DR:** the whole sampler runs on the Apple GPU via MLX. A historical continuous-model
benchmark measured **~1.5 s/sample**, ~17× faster than PyTorch-MPS and ~44× faster than CPU.
The current masked production model is a different workload; its latest paired optimization
measurement is reported separately below.

## The three backends

| Backend | Device | What runs there | How it's selected |
|---|---|---|---|
| **PyTorch + `TorchMamba2`** | CPU | Whole model | Automatic when CUDA/`mamba_ssm` is absent |
| **PyTorch + `TorchMamba2`** | Apple GPU (MPS / Metal) | Whole model | `model.to("mps")` |
| **MLX `MLXDIMBA`** | Apple GPU (Metal) | Whole model | `scripts/sample_mlx.py` |

All three load the *same* checkpoint weights. The deterministic parity benchmark below produced
the same argmax tokens; other samplers, temperatures, or precision paths must be checked rather
than assumed identical. The key enabler is
[`src/dimba/models/torch_mamba2.py`](../src/dimba/models/torch_mamba2.py) — a pure-PyTorch
Mamba-2 (SSD) mixer whose parameter tree is **identical** to `mamba_ssm.Mamba2`, so a
CUDA-trained checkpoint loads with `strict=True` and runs unchanged on CPU/MPS. MLX then ports
that same SSD mixer (and the rest of the diffusion model) to Metal via the MLX framework.

> Note on "NPU/ANE": MLX targets the Apple **GPU**, not the Neural Engine (ANE). The ANE is not
> programmable for arbitrary training/inference graphs like this one; the GPU is the right
> Apple-Silicon accelerator here, and it's what delivers the speedups below.

## Benchmarks (Apple M-series, fp32)

**SSD mixer forward** (1×256×384), `src/dimba/backends/mlx/mamba2.py`:

| | torch-CPU | torch-MPS | **MLX-GPU** |
|---|---|---|---|
| time | 40.6 ms | 17.2 ms | **1.57 ms** |
| speedup vs CPU | 1× | 2.4× | **26×** |

**Full Shakespeare sample** (256 tokens, 64 DDIM steps), `scripts/verify_mlx_model.py`:

| | torch-CPU | torch-MPS | **MLX-GPU** |
|---|---|---|---|
| time | 64.7 s | 25.5 s | **1.47 s** |
| speedup vs CPU | 1× | 2.5× | **44×** |

**Correctness:** feeding identical initial noise through PyTorch and MLX with deterministic DDIM
(eta=0) gives **100.00% argmax-token agreement** (final-logit max\|Δ\| ≈ 1e-2, from 64-step ×
12-layer × 2-direction fp32 accumulation — the *tokens* are identical). Verify with:

```bash
python scripts/verify_mlx_model.py
```

**Current masked diffusion optimization (hr-diffuse-1-nano, 288M)** — paired on this
16-GB M1 Pro, one candidate, 128 unmasking steps, CFG 2.0, 40-token answer:

| MLX implementation | time | relative |
|---|---:|---:|
| before this pass | 7.8311 s | 1.000× |
| device-resident trajectory + selected projection + batched CFG | **6.8176 s** | **1.149×** |

This is a measured **12.9% latency reduction** with identical final argmax tokens. It is one
paired local run, not a multi-run CUDA comparison. A current one-candidate CLI call is roughly
6.9 s on this machine; two candidates plus reranking is roughly 15.7 s. The older model-card
13.3 s/H100 number used a different code revision and must not be used as a current backend
comparison.

`MLXDIMBA.predict_token_logits` is parity-tested against torch in
`tests/test_mlx_masked_parity.py` (max |Δ| ≈ 3e-5, identical argmax). On MLX 0.29.3 (the last
version for Python 3.9), batches ≥3 are internally chunked into two-row calls to avoid a fused-
graph NaN bug; remove the workaround only after testing Python ≥3.10 with a current MLX release.

## How to run

### Fastest — MLX on the Apple GPU
```bash
pip install mlx
# hr-diffuse-1-nano (masked diffusion; auto-downloads the HF checkpoint, MLX auto-picked):
python scripts/generate.py "What is the capital of France?" --quality 0.5
HF_TOKEN=hf_... python scripts/sample_mlx.py --num-samples 3 --temperature 0.8
# or a local checkpoint, with a prompt:
python scripts/sample_mlx.py --ckpt shakespeare1.ckpt --prompt "ROMEO:" --steps 64
```

### PyTorch (CPU or MPS), programmatically
```python
import torch
from dimba.models.diffusion import DIMBA
from dimba.diffusion.sampling import sample_from_model

model = DIMBA(**model_config).eval()
model.load_state_dict(state_dict, strict=True)   # TorchMamba2 mixers load the CUDA weights as-is
model.to("mps")                                  # or "cpu"
ids = sample_from_model(model, prompt_ids=None, seq_len=256, num_steps=64, temperature=0.8)
```

## Implementation notes

- **`TorchMamba2`** (`src/dimba/models/torch_mamba2.py`) — fp32 SSD with a chunked scan; the
  scan masks the log-decay to `-inf` above the diagonal *before* `exp` (segment-sum trick) to
  avoid `inf*0=NaN`. Selected automatically by `denoiser._make_mixer` when `mamba_ssm` is absent.
- **`MLXMamba2Mixer`** (`src/dimba/backends/mlx/mamba2.py`) — the same SSD math in MLX. Handles
  MLX's NLC `Conv1d` layout (depthwise, `conv1d.weight` `moveaxis(2,1)` on load), `mx.split`
  cut-points, and gated RMSNorm via `mx.fast.rms_norm(y*silu(z), w, eps)`.
- **`MLXDIMBA`** (`src/dimba/backends/mlx/model.py`) — the full inference path (embeddings,
  latent projector, FiLM conditioning, 12 bidirectional blocks, timestep embedding, schedule,
  x0-DDIM, and masked diffusion). Masked trajectories and CFG stay on device, vocabulary
  projection is limited to unresolved positions, and the host sees only the final result.
  `MLXDIMBA.from_torch(torch_model)` copies weights; `.sample(...)` returns token ids.

## Why not Rust?

A CPU Rust kernel was prototyped and removed: it is CPU-only and so cannot beat the Apple GPU,
and against PyTorch's already-vectorized (BLAS, multithreaded) CPU scan the realistic gain was
only ~1.2–1.5×. MLX is the right Apple-Silicon acceleration path, and it wins decisively.
