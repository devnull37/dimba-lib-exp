#!/usr/bin/env python3
"""DIMBA generation with an accuracy-vs-speed slider.

A single ``--quality`` knob (Q in [0, 1]) trades wall-clock time for output
quality by scaling BOTH the number of iterative unmasking steps AND the number
of best-of-N candidates that get ranked by a guidance-gap verifier.

Production sampler: MaskGIT-style iterative unmasking with a cosine remask
schedule, classifier-free guidance (2.0), an exempt-first frequency penalty
(0.7), temperature (0.7), and top-k (20) multinomial sampling. The production
implementation includes the validated sampler, guidance-gap verifier, and
optional critic tie-break.

CLI:
    python scripts/generate.py "What is the capital of France?" --quality 0.5

Python API:
    from generate import slider_generate
    out = slider_generate(model, tokenizer, mask_id, "...", quality=0.5)
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import shutil
import sys
import time
import warnings

# System-python 3.9 ships LibreSSL; urllib3 warns about it on every run. Harmless.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import torch
import torch.nn.functional as F

# Allow ``PYTHONPATH=src`` runs as well as direct ``python scripts/generate.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dimba import DIMBA  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants (validated defaults; overridable via CLI expert flags).
# --------------------------------------------------------------------------- #
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# bf16 matmuls are CUDA-tuned; fp32 is the safe/fast default elsewhere (MPS included).
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()
T_MIN = 0.03                # floor for the fraction-masked t passed to the model
DEFAULT_GEN_LEN = 40
DEFAULT_TEMPERATURE = 0.7
DEFAULT_GUIDANCE = 2.0
DEFAULT_TOP_K = 20
DEFAULT_FREQ_PEN = 0.7
DEFAULT_CKPT = os.path.join(
    _REPO_ROOT, "checkpoints", "mdm_sft_cfg2", "mdm_sft_final.pt"
)
TOKENIZER_NAME = "HuggingFaceTB/SmolLM-135M"
CRITIC_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "critic_head.pt")
CRITIC_TIE_BAND = 0.02      # gap within this band -> break tie with critic


# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #
def load_model(path: str = DEFAULT_CKPT):
    """Load the flagship DIMBA checkpoint. Returns (model, mask_id).

    Falls back to the HuggingFace release (devnull37/hr-diffuse-1-nano) when
    the local training checkpoint is absent."""
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("devnull37/hr-diffuse-1-nano", "hr_diffuse_1_nano.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, mask_id = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(DEVICE).to(DTYPE).eval()
    return model, mask_id


class _MLXModel:
    """Thin Torch-facing adapter around the device-resident MLX sampler."""

    def __init__(self, torch_model, dtype="fp32"):
        from dimba.backends.mlx.model import MLXDIMBA
        import numpy as np
        self._np = np
        mx_dtype = None
        if dtype != "fp32":
            import mlx.core as mx
            mx_dtype = {"fp16": mx.float16, "bf16": mx.bfloat16}[dtype]
        self._m = MLXDIMBA.from_torch(torch_model, dtype=mx_dtype)

    def predict_token_logits(self, ids, t):
        out = self._m.predict_token_logits(ids.cpu().numpy(), float(t))
        return torch.from_numpy(self._np.array(out))

    def guided_logits(self, ids, mask_id, prompt_len, t, guidance, positions=None):
        import mlx.core as mx
        pos = None if positions is None else positions.cpu().numpy()
        out = self._m.guided_token_logits(
            ids.cpu().numpy(), mask_id, prompt_len, float(t), guidance, pos
        )
        mx.eval(out)
        return torch.from_numpy(self._np.array(out))

    def masked_generate(
        self,
        prompt_ids,
        mask_id,
        *,
        gen_len,
        steps,
        temperature,
        top_k,
        freq_pen,
        guidance,
        seed,
        on_step,
        commit_threshold=None,
    ):
        callback = None
        if on_step is not None:

            def callback(ids, still, step, total):
                on_step(torch.from_numpy(ids), torch.from_numpy(still), step, total)

        out = self._m.sample_masked(
            prompt_ids.cpu().numpy(),
            mask_id,
            gen_len=gen_len,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            freq_pen=freq_pen,
            guidance=guidance,
            seed=seed,
            on_step=callback,
            commit_threshold=commit_threshold,
        )
        return torch.from_numpy(out)

    def gap_score(self, ids, mask_id, eos, prompt_len, groups):
        return torch.from_numpy(
            self._m.masked_gap_score(
                ids.cpu().numpy(), mask_id, eos, prompt_len, groups
            )
        )


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def load_critic(use_compile=True):
    """Load the trained critic head MLP: Linear(dim,256)+GELU+Linear(256,1)."""
    ck = torch.load(CRITIC_CKPT, map_location="cpu")
    dim = ck["dim"]
    critic = torch.nn.Sequential(
        torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
    )
    state = {k.replace("net.", ""): v for k, v in ck["critic_state_dict"].items()}
    critic.load_state_dict(state)
    critic = critic.to(DEVICE).to(DTYPE).eval()
    from dimba.utils.compile import maybe_compile_fn
    return maybe_compile_fn(critic, enable=use_compile,
                            dynamic=True if DEVICE == "cuda" else None)


# --------------------------------------------------------------------------- #
# Sampler.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def guided_logits(model, mask_id, ids, prompt_len, t, guidance, positions=None):
    """Classifier-free guidance: lu + guidance * (lc - lu).

    The cond and uncond passes are batched into one forward (identical math,
    one model dispatch instead of two)."""
    if hasattr(model, "guided_logits"):  # MLX: CFG combined on-GPU
        return model.guided_logits(ids, mask_id, prompt_len, t, guidance, positions)
    u = ids.clone()
    u[:, :prompt_len] = mask_id
    if guidance in (0.0, 1.0):
        chosen = u if guidance == 0.0 else ids
        if hasattr(model, "predict_token_features"):
            features = model.predict_token_features(chosen, t)
            return model.project_token_features(features, positions=positions).float()
        return model.predict_token_logits(chosen, t, positions=positions).float()
    both_ids = torch.cat([ids, u], dim=0)
    if hasattr(model, "predict_token_features"):
        # The final head projection is linear. Combine post-normalization/head-
        # attention features first, then project once: exactly the same CFG logits
        # with half the vocabulary GEMMs and only unresolved positions materialized.
        features = model.predict_token_features(both_ids, t)
        fc, fu = features.chunk(2, dim=0)
        return model.project_token_features(
            fu + guidance * (fc - fu), positions=positions
        ).float()
    both_positions = torch.cat([positions, positions], dim=0) if positions is not None else None
    both = model.predict_token_logits(
        both_ids, t, positions=both_positions
    ).float()
    lc, lu = both.chunk(2, dim=0)
    return lu + guidance * (lc - lu)


def _make_cfg_features(model, mask_id, prompt_len, guidance, use_graph=False):
    """Fixed-shape per-step feature pass: CFG combined in feature space.

    Returns ``fn(ids, t) -> [B, L, d_model]`` with ``t`` a 0-d float tensor.
    Same math as :func:`guided_logits` minus the (shape-varying) vocabulary
    projection, which the caller keeps eager. Every input shape is constant
    across the whole trajectory, so the callable is CUDA-graph capturable.
    ``guidance`` is closed over as a constant (one graph per guidance value)."""

    graph_handle = None
    if use_graph and hasattr(model, "_predict_token_features"):
        from dimba.utils.cuda_graphs import GraphedFn

        denoise_fn = lambda z, t_idx, cond: model._denoiser_raw(z, t_idx, cond, None)
        graph_handle = GraphedFn(denoise_fn)
        predict = lambda ids, t: model._predict_token_features(ids, t, graph_handle)
    else:
        predict = model.predict_token_features

    def cfg_features(ids, t):
        if guidance in (0.0, 1.0):
            chosen = ids
            if guidance == 0.0:
                chosen = ids.clone()
                chosen[:, :prompt_len] = mask_id
            return predict(chosen, t)
        u = ids.clone()
        u[:, :prompt_len] = mask_id
        both_ids = torch.cat([ids, u], dim=0)
        features = predict(both_ids, t)
        fc, fu = features.chunk(2, dim=0)
        return fu + guidance * (fc - fu)

    cfg_features.graph_handle = graph_handle
    return cfg_features


def _denoise_step(logits, ids, still, positions, prompt_len, mask_id,
                  temperature, top_k, freq_pen, n_keep):
    """One fused denoising step: frequency penalty, temperature, top-k,
    sampling, confidence-ranked remasking. Pure tensor ops so torch.compile
    (inductor) can fuse the masking/schedule/penalty math into few kernels.

    ``logits`` contains only the currently unresolved ``positions``; the model
    still processed the complete bidirectional sequence, but avoids the dominant
    ``hidden @ vocab`` projection for prompt and committed tokens.

    Returns ``(ids, still, positions)`` for the next step."""
    ids, conf = _sample_tokens(logits, ids, still, positions, prompt_len,
                               mask_id, temperature, top_k, freq_pen)
    return _commit_tokens(ids, still, positions, conf, mask_id, n_keep)


def _sample_tokens(logits, ids, still, positions, prompt_len, mask_id,
                   temperature, top_k, freq_pen):
    """Penalty/temperature/top-k sampling half of the step: writes the sampled
    tokens into ``ids`` and returns their confidence for the commit decision."""
    B, _, V = logits.shape
    # Exempt-first frequency penalty over already-committed generated tokens
    # (vectorized equivalent of the old per-row unique() loop).
    committed = ~still
    committed[:, :prompt_len] = False
    counts = torch.zeros(B, V, device=logits.device, dtype=logits.dtype)
    counts.scatter_add_(1, ids.masked_fill(~committed, 0),
                        committed.to(logits.dtype))
    logits = logits - freq_pen * (counts - 1).clamp(min=0).unsqueeze(1)
    # Temperature AFTER penalty/guidance, then top-k multinomial sampling.
    logits = logits / temperature
    top_values, top_indices = logits.topk(top_k, dim=-1)
    top_probs = F.softmax(top_values, dim=-1)
    choice = torch.multinomial(top_probs.reshape(-1, top_k), 1).view(B, -1, 1)
    sampled = top_indices.gather(-1, choice).squeeze(-1)
    conf = top_probs.gather(-1, choice).squeeze(-1)
    ids = ids.scatter(1, positions, sampled)
    return ids, conf


def _commit_tokens(ids, still, positions, conf, mask_id, n_keep):
    """Remask the ``n_keep`` lowest-confidence positions; commit the rest."""
    if n_keep > 0:
        keep_local = conf.topk(n_keep, dim=1, largest=False).indices
        positions = positions.gather(1, keep_local)
        still = torch.zeros_like(still).scatter(1, positions, True)
        ids = ids.scatter(1, positions, torch.full_like(positions, mask_id))
    else:
        positions = positions[:, :0]
        still = torch.zeros_like(still)
    return ids, still, positions


# Compiled lazily on first use so import stays cheap and --no-compile can veto.
_compiled_step = None
_compiled_split = None


def _get_step(use_compile):
    global _compiled_step
    if not use_compile:
        return _denoise_step
    if _compiled_step is None:
        from dimba.utils.compile import maybe_compile_fn
        # dynamic=True lets inductor fuse across the varying still-masked
        # fraction without a recompile per n_keep; the MPS inductor backend
        # chokes on explicit dynamic=True (torch 2.8), so let dynamo
        # auto-detect there. maybe_compile_fn falls back to eager on any
        # compile or runtime failure, so this is safe on every device.
        _compiled_step = maybe_compile_fn(
            _denoise_step, dynamic=True if DEVICE == "cuda" else None
        )
    return _compiled_step


def _get_split_step(use_compile):
    """Sample/commit halves compiled separately: the confidence-threshold
    commit count is decided on the host between them (one scalar sync)."""
    global _compiled_split
    if not use_compile:
        return _sample_tokens, _commit_tokens
    if _compiled_split is None:
        from dimba.utils.compile import maybe_compile_fn
        dyn = True if DEVICE == "cuda" else None
        _compiled_split = (
            maybe_compile_fn(_sample_tokens, dynamic=dyn),
            maybe_compile_fn(_commit_tokens, dynamic=dyn),
        )
    return _compiled_split


@torch.inference_mode()
def generate(
    model,
    mask_id,
    prompt_ids,
    gen_len=DEFAULT_GEN_LEN,
    steps=128,
    temperature=DEFAULT_TEMPERATURE,
    top_k=DEFAULT_TOP_K,
    freq_pen=DEFAULT_FREQ_PEN,
    guidance=DEFAULT_GUIDANCE,
    on_step=None,
    use_compile=True,
    seed=None,
    commit_threshold=None,
    cfg_drop=None,
    use_graph=None,
):
    """MaskGIT-style iterative unmasking with a cosine remask schedule.

    The backbone forward has a constant input shape across the trajectory, so
    on CUDA it is captured as a CUDA graph (``use_graph``, default on) and each
    step replays one graph instead of ~1000 eager kernel launches; the
    shape-varying vocabulary projection stays eager on the selected-positions
    fast path. The per-step tensor math (penalty, top-k, sampling, remask)
    runs through ``torch.compile`` when available (CUDA/MPS/CPU inductor),
    falling back to eager transparently. ``on_step(ids, still, s, steps)`` is
    called after every unmasking step (used by --watch to render the denoising
    live).

    ``commit_threshold`` (quality-affecting, off by default): additionally
    commit any sampled token whose confidence exceeds the threshold even when
    the cosine schedule would remask it, and stop as soon as every position is
    committed. Costs one scalar GPU->host sync per step.

    ``cfg_drop`` (quality-affecting, off by default): once this fraction of
    response tokens is committed, drop the unconditional CFG row and run
    single-row (guidance-free) forwards for the remaining steps."""
    if steps < 1 or gen_len < 1:
        raise ValueError("steps and gen_len must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if seed is not None:
        torch.manual_seed(int(seed))
    if hasattr(model, "masked_generate"):
        return model.masked_generate(
            prompt_ids,
            mask_id,
            gen_len=gen_len,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            freq_pen=freq_pen,
            guidance=guidance,
            seed=seed,
            on_step=on_step,
            commit_threshold=commit_threshold,
        )

    step_fn = _get_step(use_compile)
    if commit_threshold is not None:
        sample_fn, commit_fn = _get_split_step(use_compile)
    B, P = prompt_ids.shape
    device = prompt_ids.device
    ids = torch.cat(
        [
            prompt_ids,
            torch.full((B, gen_len), mask_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    still = torch.zeros(B, P + gen_len, dtype=torch.bool, device=device)
    still[:, P:] = True
    positions = torch.arange(P, P + gen_len, device=device).unsqueeze(0).expand(B, -1)

    # Fast path: models exposing the split feature/projection API get CFG
    # combined in feature space with a fixed-shape backbone call per step,
    # optionally captured as a CUDA graph. Other models (and the MLX adapter's
    # guided_logits) keep the generic per-step dispatch below.
    feature_cfg = hasattr(model, "predict_token_features") and not hasattr(
        model, "guided_logits"
    )
    if use_graph is None:
        use_graph = device.type == "cuda"
    feats_main = feats_cond = None
    if feature_cfg:
        feats_main = _make_cfg_features(model, mask_id, P, guidance, use_graph=use_graph)
        if cfg_drop is not None and guidance not in (0.0, 1.0):
            feats_cond = _make_cfg_features(model, mask_id, P, 1.0, use_graph=use_graph)

    n_active = gen_len
    for s in range(steps):
        # Python already knows the schedule count; avoid a GPU -> host ``.item()``
        # synchronization just to recover the masked fraction.
        frac = n_active / (P + gen_len)
        t = max(frac, T_MIN)
        if feature_cfg:
            drop_uncond = (
                feats_cond is not None
                and (gen_len - n_active) / gen_len >= cfg_drop
            )
            # 0-d float tensor: under graph capture/compile a Python float
            # would be baked in (or trigger a recompile) per distinct t.
            t_dev = torch.tensor(t, device=device)
            features = (feats_cond if drop_uncond else feats_main)(ids, t_dev)
            logits = model.project_token_features(
                features, positions=positions
            ).float()
        else:
            logits = guided_logits(model, mask_id, ids, P, t, guidance, positions)
        n_keep = int(gen_len * math.cos(math.pi / 2 * (s + 1) / steps))
        if commit_threshold is None:
            ids, still, positions = step_fn(
                logits, ids, still, positions, P, mask_id,
                temperature, top_k, freq_pen, n_keep,
            )
        else:
            ids, conf = sample_fn(
                logits, ids, still, positions, P, mask_id,
                temperature, top_k, freq_pen,
            )
            # Keep masked only what is BOTH scheduled to stay masked and still
            # low-confidence; the max over rows never commits a token the
            # schedule+threshold combination wouldn't. One scalar sync.
            n_low = int((conf <= commit_threshold).sum(dim=1).max())
            n_keep = min(n_keep, n_low)
            ids, still, positions = commit_fn(
                ids, still, positions, conf, mask_id, n_keep
            )
        n_active = n_keep
        if on_step:
            on_step(ids, still, s, steps)
        if n_keep <= 0:
            break
    return ids


# --------------------------------------------------------------------------- #
# Verifiers.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def gap_score(model, mask_id, eos, ids, prompt_len, K=4):
    """Guidance-gap verifier: mean over generated positions of
    [logp_cond - logp_uncond] for the committed token, with the scored
    positions MASKED via K interleaved leave-k-out passes at t=0.25.
    Higher = better."""
    if K < 1:
        raise ValueError("K must be positive")
    if hasattr(model, "gap_score"):
        return model.gap_score(ids, mask_id, eos, prompt_len, K)

    B, L = ids.shape
    if prompt_len >= L:
        return torch.zeros(B, device=ids.device)
    gen = torch.zeros_like(ids, dtype=torch.bool)
    gen[:, prompt_len:] = (ids[:, prompt_len:] != eos) & (ids[:, prompt_len:] != mask_id)
    response_positions = torch.arange(prompt_len, L, device=ids.device)
    width = (response_positions.numel() + K - 1) // K
    variants, selected_positions, valid_positions, selected_tokens = [], [], [], []
    for r in range(K):
        positions_r = response_positions[response_positions % K == r]
        valid_width = torch.arange(width, device=ids.device) < positions_r.numel()
        if positions_r.numel() < width:
            positions_r = F.pad(positions_r, (0, width - positions_r.numel()))
        positions_r = positions_r.unsqueeze(0).expand(B, -1)
        valid = gen.gather(1, positions_r) & valid_width.unsqueeze(0)
        m = torch.zeros_like(gen).scatter(1, positions_r, valid)
        x = ids.masked_fill(m, mask_id)
        variants.append(x)
        selected_positions.append(positions_r)
        valid_positions.append(valid)
        selected_tokens.append(ids.gather(1, positions_r))

    conditional = torch.cat(variants, dim=0)
    positions = torch.cat(selected_positions, dim=0)
    valid = torch.cat(valid_positions, dim=0)
    targets = torch.cat(selected_tokens, dim=0)
    unconditional = conditional.clone()
    unconditional[:, :prompt_len] = mask_id
    both_ids = torch.cat((conditional, unconditional), dim=0)
    both_positions = torch.cat((positions, positions), dim=0)

    if hasattr(model, "predict_token_features"):
        features = model.predict_token_features(both_ids, 0.25)
        logits = model.project_token_features(features, positions=both_positions)
    else:
        logits = model.predict_token_logits(both_ids, 0.25, positions=both_positions)
    lc, lu = logits.float().chunk(2, dim=0)

    def chosen_logp(logits):
        return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
            logits, dim=-1
        )

    gaps = (chosen_logp(lc) - chosen_logp(lu)) * valid
    return gaps.view(K, B, width).sum((0, 2)) / valid.view(K, B, width).sum(
        (0, 2)
    ).clamp(min=1)


@torch.no_grad()
def _xdec(model, ids, t):
    """Decoded x0 latent features ("x_dec") used by the critic head."""
    B = ids.shape[0]
    z = model.encode_latent(model.token_embed(ids))
    cond = model._build_conditioning(None, B, ids.device)
    ti = model._to_timestep_index(t, B, ids.device)
    raw = model._denoiser_raw(z, ti, cond, None)
    return model.decode_latent(model._to_x0_latent(z, raw, ti))


@torch.no_grad()
def critic_score(model, critic, mask_id, eos, ids, prompt_len):
    """Mean predicted wrongness over generated tokens (lower = better)."""
    gen = torch.zeros_like(ids, dtype=torch.bool)
    gen[:, prompt_len:] = (ids[:, prompt_len:] != mask_id) & (ids[:, prompt_len:] != eos)
    s = torch.sigmoid(critic(_xdec(model, ids, 0.15)).squeeze(-1).float())
    return (s * gen).sum(1) / gen.sum(1).clamp(min=1)


# --------------------------------------------------------------------------- #
# Quality slider mapping.
# --------------------------------------------------------------------------- #
def _geom_interp(q, lo, mid, hi):
    """Geometric interpolation lo (q=0) -> mid (q=0.5) -> hi (q=1.0)."""
    q = max(0.0, min(1.0, q))
    if q <= 0.5:
        f = q / 0.5
        return lo * (mid / lo) ** f
    f = (q - 0.5) / 0.5
    return mid * (hi / mid) ** f


def steps_for_quality(q):
    return int(round(_geom_interp(q, 16.0, 64.0, 256.0)))


def n_for_quality(q):
    if q < 0.35:
        return 1
    if q < 0.6:
        return 2
    if q < 0.85:
        return 4
    return 8


# --------------------------------------------------------------------------- #
# Live watch (--watch): redraw the unmasking in place each step.
# --------------------------------------------------------------------------- #
def make_watcher(tokenizer, mask_id, prompt_len):
    """Return an ``on_step`` callback that renders candidate 0's response live,
    masked positions shown as ░."""
    state = {"lines": 0}

    def on_step(ids, still, s, steps):
        toks = ids[0, prompt_len:].tolist()
        text = "".join(
            "░" if t >= mask_id else tokenizer.decode([t]) for t in toks
        ).replace("\n", "¶")
        out = f"[{s + 1:>3}/{steps}] {text}"
        width = max(shutil.get_terminal_size().columns, 20)
        lines = [out[i:i + width] for i in range(0, len(out), width)] or [""]
        sys.stdout.write("\x1b[1A\x1b[2K" * state["lines"])
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        state["lines"] = len(lines)

    return on_step


# --------------------------------------------------------------------------- #
# Decoding.
# --------------------------------------------------------------------------- #
def decode(tokenizer, ids_row, prompt_len, mask_id, eos):
    """Drop tokens >= mask_id and truncate at the first EOS."""
    toks = [x for x in ids_row[prompt_len:].tolist() if x < mask_id]
    if eos in toks:
        toks = toks[: toks.index(eos)]
    return tokenizer.decode(toks)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def slider_generate(model, tokenizer, mask_id, question, quality=0.5,
                    critic=None, **overrides):
    """Generate an answer with the accuracy-vs-speed slider.

    Args:
        model, tokenizer, mask_id: as loaded by ``load_*``.
        question: the raw question string.
        quality: Q in [0, 1] mapping to (steps, N).
        critic: optional loaded critic head; used to break near-ties in gap.
        overrides: any of steps, n, temperature, guidance, seed, gen_len,
            top_k, freq_pen, commit_threshold, cfg_drop, use_graph.

    Returns:
        dict with keys: text, steps, n, seconds, scores (list, empty when N==1),
        plus best (index) and all_texts.
    """
    eos = tokenizer.eos_token_id
    # ponytail: `or` treats falsy overrides (e.g. --guidance 0) as unset; use None-check like freq_pen did.
    ov = lambda k, default: default if overrides.get(k) is None else overrides.get(k)
    steps = int(ov("steps", steps_for_quality(quality)))
    n = int(ov("n", n_for_quality(quality)))
    gen_len = int(ov("gen_len", DEFAULT_GEN_LEN))
    temperature = float(ov("temperature", DEFAULT_TEMPERATURE))
    guidance = float(ov("guidance", DEFAULT_GUIDANCE))
    top_k = int(ov("top_k", DEFAULT_TOP_K))
    freq_pen = float(ov("freq_pen", DEFAULT_FREQ_PEN))
    seed = overrides.get("seed")

    if seed is not None:
        torch.manual_seed(int(seed))

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)] * n, device=DEVICE
    )
    P = prompt_ids.shape[1]

    _sync()
    t0 = time.time()

    ids = generate(
        model, mask_id, prompt_ids, gen_len=gen_len, steps=steps,
        temperature=temperature, top_k=top_k, freq_pen=freq_pen, guidance=guidance,
        on_step=overrides.get("on_step"),
        use_compile=bool(ov("use_compile", True)),
        seed=seed,
        commit_threshold=overrides.get("commit_threshold"),
        cfg_drop=overrides.get("cfg_drop"),
        use_graph=overrides.get("use_graph"),
    )

    scores = []
    if n == 1:
        best = 0
    else:
        gap = gap_score(model, mask_id, eos, ids, P)
        scores = gap.tolist()
        order = sorted(range(n), key=lambda i: -scores[i])
        best = order[0]
        if critic is not None:
            # Break near-ties (gap within band) by lower critic wrongness.
            top_gap = scores[best]
            tied = [i for i in range(n) if top_gap - scores[i] <= CRITIC_TIE_BAND]
            if len(tied) > 1:
                cs = critic_score(model, critic, mask_id, eos, ids, P).tolist()
                best = min(tied, key=lambda i: cs[i])

    _sync()
    seconds = time.time() - t0

    text = decode(tokenizer, ids[best], P, mask_id, eos)
    all_texts = [decode(tokenizer, ids[i], P, mask_id, eos) for i in range(n)]
    return {
        "text": text,
        "steps": steps,
        "n": n,
        "seconds": seconds,
        "scores": scores,
        "best": best,
        "all_texts": all_texts,
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="DIMBA generation with an accuracy-vs-speed slider."
    )
    p.add_argument("question", help="the question to answer")
    p.add_argument("--quality", type=float, default=0.5,
                   help="Q in [0,1]: higher = slower & better (default 0.5)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT, help="model checkpoint")
    # Expert overrides.
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n", type=int, default=None, help="best-of-N candidates")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gen-len", type=int, default=None, dest="gen_len")
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"],
                   help="auto = MLX (Apple GPU) when installed, else torch")
    p.add_argument("--mlx-dtype", default="fp32", dest="mlx_dtype",
                   choices=["fp32", "fp16", "bf16"],
                   help="MLX weight/activation precision; fp16 roughly halves "
                        "weight bandwidth (quality-affecting, scan and "
                        "confidence math stay fp32)")
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile on the per-step sampler math")
    p.add_argument("--no-graph", action="store_true",
                   help="disable CUDA-graph capture of the backbone forward")
    p.add_argument("--commit-threshold", type=float, default=None,
                   dest="commit_threshold", metavar="TAU",
                   help="commit tokens above this confidence ahead of the "
                        "cosine schedule and stop early (quality-affecting; "
                        "try 0.9)")
    p.add_argument("--cfg-drop", type=float, default=None, dest="cfg_drop",
                   metavar="FRAC",
                   help="drop the unconditional CFG pass once this fraction "
                        "of tokens is committed (quality-affecting; try 0.75)")
    p.add_argument("--critic", action="store_true",
                   help="load the critic head to break near-ties in gap")
    p.add_argument("--show-all", action="store_true",
                   help="print all candidates with scores")
    p.add_argument("--watch", action="store_true",
                   help="render the diffusion live: masked positions (░) fill in "
                        "step by step (candidate 0)")
    return p


def main(argv=None):
    global DEVICE, DTYPE
    args = build_parser().parse_args(argv)

    backend = args.backend
    if backend == "auto":
        try:
            import mlx.core  # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "torch"
    if backend == "mlx":
        if args.critic:
            sys.exit("--critic uses torch model internals; use --backend torch")
        # MLX does the model math on the Apple GPU; keep the (cheap) sampler-loop
        # tensors on CPU and load the torch model in fp32 for weight conversion.
        DEVICE, DTYPE = "cpu", torch.float32

    model, mask_id = load_model(args.checkpoint)
    if backend == "mlx":
        model = _MLXModel(model, dtype=args.mlx_dtype)
    tokenizer = load_tokenizer()
    critic = load_critic(use_compile=not args.no_compile) if args.critic else None

    overrides = {}
    for k in ("steps", "n", "temperature", "guidance", "seed", "gen_len",
              "commit_threshold", "cfg_drop"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
    # MLX keeps only tiny glue tensors on CPU; compile latency isn't worth it.
    if args.no_compile or backend == "mlx":
        overrides["use_compile"] = False
    if args.no_graph:
        overrides["use_graph"] = False
    if args.watch:
        P = len(tokenizer.encode(f"Question: {args.question}\nAnswer:",
                                 add_special_tokens=False))
        overrides["on_step"] = make_watcher(tokenizer, mask_id, P)

    out = slider_generate(
        model, tokenizer, mask_id, args.question,
        quality=args.quality, critic=critic, **overrides,
    )

    if args.show_all and out["n"] > 1:
        print("Candidates:")
        for i in range(out["n"]):
            score = out["scores"][i] if out["scores"] else float("nan")
            mark = " *" if i == out["best"] else "  "
            print(f"{mark}[{i}] gap={score:+.3f}  {out['all_texts'][i]}")
        print()

    print(f"quality      : {args.quality}")
    print(f"steps        : {out['steps']}")
    print(f"N candidates : {out['n']}")
    print(f"wall-clock   : {out['seconds']:.2f} s")
    print(f"answer       : {out['text']}")


if __name__ == "__main__":
    main()
