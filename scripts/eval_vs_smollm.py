#!/usr/bin/env python3
"""Evaluation harness: DIMBA diffusion model vs SmolLM-135M-Instruct.

USAGE
-----
Basic (gen + speed tasks, base checkpoint):
    python scripts/eval_vs_smollm.py \\
        --checkpoint /path/to/base/final.pt \\
        --tasks gen,speed --steps 20

SFT/GRPO instruct checkpoint (block-CoT enabled):
    python scripts/eval_vs_smollm.py \\
        --checkpoint checkpoints/sft/final.pt \\
        --tasks gen,speed \\
        --block-cot --steps 20

All tasks including perplexity:
    python scripts/eval_vs_smollm.py \\
        --checkpoint checkpoints/sft/final.pt \\
        --tasks gen,speed,ppl \\
        --block-cot --steps 20

NOTES ON CORRECTNESS
---------------------
* The DIMBA checkpoint is detected as flow-matching (use_flow_matching=True in
  ckpt["config"]) and sampled with the Euler ODE integrator (sample_from_model_flow)
  rather than DDIM. Using DDIM on a flow-matching checkpoint produces degenerate
  output because the alpha_cumprod schedule does not correspond to the linear
  interpolation the model was trained with.

* SmolLM tokenizer (HuggingFaceTB/SmolLM-135M) is used for DIMBA token decode,
  since DIMBA is distilled from that teacher and shares its vocabulary.

* Speed comparison caveat: DIMBA runs a fixed number of diffusion steps (NFE)
  over the ENTIRE sequence in parallel; SmolLM generates tokens sequentially with
  a KV cache. The comparison is wall-clock honest but architecturally asymmetric.
  We report both NFE and wall-clock so the reader is not misled.

* Perplexity caveat: DIMBA's perplexity is a noise-level reconstruction NLL (an
  ELBO term, NOT standard AR perplexity). SmolLM's is standard AR perplexity.
  They cannot be compared head-to-head; each is a within-model trend metric.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Make the src/ directory importable regardless of where the script is called from.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

# ── evaluation prompts ─────────────────────────────────────────────────────────
# A curated set covering instruction-following, factual QA, arithmetic, and open
# completion so that a human reviewer can eyeball model quality across task types.
EVAL_PROMPTS = [
    # Instruction-following
    "Write a short poem about the night sky.",
    "Explain what photosynthesis is in two sentences.",
    # Factual QA
    "What is the capital of France?",
    "Who wrote the play Romeo and Juliet?",
    # Arithmetic / reasoning
    "What is 17 multiplied by 8?",
    "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    # Open completion
    "Once upon a time in a land far away,",
    "The most important invention of the 20th century was",
]

# Small held-out text for the perplexity task (PPL).
# Sourced from public domain; does not overlap with DIMBA's training data.
PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning God created the heavens and the earth. "
    "It was the best of times, it was the worst of times, it was the age of wisdom. "
    "All happy families are alike; each unhappy family is unhappy in its own way. "
    "Call me Ishmael. Some years ago, never mind how long precisely, having little "
    "money in my purse, and nothing particular to interest me on shore, I thought I "
    "would sail about a little and see the watery part of the world. "
    "To be, or not to be, that is the question. "
    "We hold these truths to be self-evident, that all men are created equal. "
    "In the beginning was the Word, and the Word was with God, and the Word was God. "
    "It is a truth universally acknowledged that a single man in possession of a "
    "good fortune must be in want of a wife. "
    "The sky above the port was the color of television, tuned to a dead channel. "
    "You are not the kind of guy who would be at a place like this at this time of "
    "the morning. But here you are, and you cannot say that the terrain is entirely unfamiliar."
)

# ── DIMBA checkpoint loading ───────────────────────────────────────────────────

def load_dimba(checkpoint_path: str, device: torch.device):
    """Load a DIMBA model from a .pt checkpoint.

    Uses DIMBA(**ckpt["config"]) + load_state_dict(strict=False), mirroring
    train_4090.py's _build_or_load_model. Handles the force_torch_mixer key
    to route to the pure-PyTorch TorchMamba2 backend when mamba_ssm is absent.
    """
    import inspect
    from dimba.models.diffusion import DIMBA

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"DIMBA checkpoint not found: {checkpoint_path}\n"
            "If the SFT/GRPO run hasn't finished yet, use the base checkpoint "
            "instead (e.g. --checkpoint /path/to/base/final.pt)."
        )

    print(f"  loading DIMBA checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            "Re-save it with model.config, or use a checkpoint produced by train_4090.py."
        )

    # Resolve the actual vocabulary size from the embedding tensor (handles post-SFT
    # resize where <think>/</think> tokens pushed vocab_size beyond the config value).
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    emb_keys = [k for k in sd if k.endswith("token_embed.embedding.weight")]
    if emb_keys:
        true_vocab = sd[emb_keys[0]].shape[0]
        cfg = {**cfg, "vocab_size": true_vocab}

    # Honour force_torch_mixer if the checkpoint was saved on a TorchMamba2 box.
    # On a CPU-only machine mamba_ssm is not installed, so TorchMamba2 will be used
    # automatically; the flag just makes the intent explicit.
    dimba_params = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    cfg_clean = {k: v for k, v in cfg.items() if k in dimba_params}

    model = DIMBA(**cfg_clean)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if "noise_schedule" not in m and "flow_schedule" not in m]
    if real_missing:
        print(f"  [warn] missing keys (non-schedule): {real_missing[:6]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:6]}")

    model = model.to(device).eval()

    use_flow = bool(cfg_clean.get("use_flow_matching", False))
    print(f"  DIMBA loaded: vocab={model.vocab_size} d_model={model.d_model} "
          f"d_latent={model.d_latent} use_flow_matching={use_flow}")
    return model, cfg_clean, use_flow


# ── SmolLM loading ─────────────────────────────────────────────────────────────

def load_smollm(baseline_id: str, device: torch.device):
    """Load SmolLM-135M-Instruct (or any HF causal LM) and its tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  loading SmolLM: {baseline_id}")
    tok = AutoTokenizer.from_pretrained(baseline_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(baseline_id, dtype=torch.float32 if device.type == "cpu" else torch.bfloat16)
    lm = lm.to(device).eval()
    print(f"  SmolLM loaded: vocab={lm.config.vocab_size} "
          f"params={sum(p.numel() for p in lm.parameters()):,}")
    return lm, tok


# ── tokenizer for DIMBA decode ────────────────────────────────────────────────

def load_dimba_tokenizer(baseline_id: str):
    """Use the SmolLM/teacher tokenizer to decode DIMBA tokens (shared vocabulary)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(baseline_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── DIMBA generation helpers ──────────────────────────────────────────────────

def _add_think_tokens(tokenizer, model):
    """Add <think>/</think> tokens and resize model embedding if not already present.

    Mirrors the run_sft / run_grpo pattern in train_4090.py exactly.
    Returns (think_start_id, think_end_id).
    """
    if "<think>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
        model.token_embed.resize(len(tokenizer))
        if hasattr(model.output_head, "embedding_weight"):
            model.output_head.embedding_weight = model.token_embed.get_weight()
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    return think_start_id, think_end_id


def dimba_generate(
    model,
    tokenizer,
    prompt: str,
    seq_len: int,
    num_steps: int,
    use_flow: bool,
    block_cot: bool,
    think_start_id: Optional[int],
    think_end_id: Optional[int],
    device: torch.device,
) -> str:
    """Generate text with DIMBA, using the correct sampler for the checkpoint type."""
    from dimba.diffusion.sampling import sample_from_model_flow, sample_from_model
    from dimba.inference.block_cot import block_sample_from_model

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if block_cot and think_start_id is not None:
        # Block-CoT path: <think> blocks then response.
        # Uses block_size=64, 2 think blocks (matching GRPO_CFG defaults).
        result = block_sample_from_model(
            model,
            ids,
            block_size=64,
            num_think_blocks=2,
            response_len=seq_len,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            eos_id=tokenizer.eos_token_id,
            adaptive_stop=True,
            num_steps=num_steps,
            sampler="euler" if use_flow else "ddim",
        )
        out_ids = result["response"][0].cpu().tolist()
    elif use_flow:
        out = sample_from_model_flow(
            model,
            ids,
            seq_len=seq_len,
            num_steps=num_steps,
            sampler="euler",
            device=device,
        )
        out_ids = out[0].cpu().tolist()
    else:
        out = sample_from_model(
            model,
            ids,
            seq_len=seq_len,
            num_steps=num_steps,
            sampler="ddim",
            device=device,
        )
        out_ids = out[0].cpu().tolist()

    # Decode and clean up: strip padding/EOS tokens at the end.
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        try:
            cut = out_ids.index(eos_id)
            out_ids = out_ids[:cut]
        except ValueError:
            pass
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def smollm_generate(lm, tokenizer, prompt: str, max_new: int, device: torch.device) -> str:
    """Generate with SmolLM-Instruct using chat template."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = lm.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,          # greedy for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Task 1: generation quality ────────────────────────────────────────────────

def run_gen(
    dimba_model,
    dimba_tok,
    smollm,
    smollm_tok,
    num_steps: int,
    max_new: int,
    use_flow: bool,
    block_cot: bool,
    think_start_id: Optional[int],
    think_end_id: Optional[int],
    device: torch.device,
):
    print("\n" + "=" * 70)
    print("TASK 1: GENERATION QUALITY (side-by-side)")
    print("=" * 70)
    sampler_label = (
        "flow/euler block-CoT" if block_cot
        else "flow/euler" if use_flow
        else "ddim"
    )
    print(f"DIMBA sampler: {sampler_label} | steps={num_steps} | response_len={max_new}")
    print(f"SmolLM: greedy | max_new_tokens={max_new}\n")

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"--- Prompt {i}/{len(EVAL_PROMPTS)} ---")
        print(f"  PROMPT: {prompt}")

        dimba_out = dimba_generate(
            dimba_model, dimba_tok, prompt, max_new, num_steps,
            use_flow, block_cot, think_start_id, think_end_id, device,
        )
        smollm_out = smollm_generate(smollm, smollm_tok, prompt, max_new, device)

        print(f"  DIMBA:   {dimba_out!r}")
        print(f"  SmolLM:  {smollm_out!r}")
        print()

    print("NOTE: DIMBA is flow-matching / bidirectional — output quality is")
    print("  expected to differ from AR SmolLM. A base checkpoint will be")
    print("  incoherent; SFT/GRPO checkpoints should show coherent responses.")


# ── Task 2: throughput / speed ─────────────────────────────────────────────────

def _time_dimba(model, tokenizer, prompt_ids, seq_len, num_steps, use_flow, device,
                n_runs=3, n_warmup=1):
    """Return (mean_wall_sec, tokens_per_sec) for DIMBA over n_runs timed runs."""
    from dimba.diffusion.sampling import sample_from_model_flow, sample_from_model

    def _once():
        if use_flow:
            return sample_from_model_flow(
                model, prompt_ids, seq_len=seq_len, num_steps=num_steps,
                sampler="euler", device=device,
            )
        else:
            return sample_from_model(
                model, prompt_ids, seq_len=seq_len, num_steps=num_steps,
                sampler="ddim", device=device,
            )

    # Warmup (untimed)
    for _ in range(n_warmup):
        _once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = sum(times) / len(times)
    tps = seq_len / mean_t
    return mean_t, tps


def _time_smollm(lm, tokenizer, prompt_ids, max_new, device, n_runs=3, n_warmup=1):
    """Return (mean_wall_sec, tokens_per_sec) for SmolLM.generate over n_runs timed runs."""
    attn_mask = torch.ones_like(prompt_ids)

    def _once():
        return lm.generate(
            input_ids=prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    for _ in range(n_warmup):
        _once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = sum(times) / len(times)
    tps = max_new / mean_t
    return mean_t, tps


def run_speed(
    dimba_model,
    dimba_tok,
    smollm,
    smollm_tok,
    num_steps: int,
    max_new: int,
    use_flow: bool,
    device: torch.device,
):
    print("\n" + "=" * 70)
    print("TASK 2: THROUGHPUT / SPEED")
    print("=" * 70)

    # Use a short fixed prompt so prompt tokenization doesn't dominate.
    prompt_text = "The"
    prompt_ids_dimba = dimba_tok.encode(prompt_text, return_tensors="pt").to(device)
    prompt_ids_smollm = smollm_tok.encode(prompt_text, return_tensors="pt").to(device)

    # Output lengths to benchmark. Clip to --max-new so short smoke-test runs quickly.
    lengths_candidates = [32, 128, 512]
    lengths = [l for l in lengths_candidates if l <= max_new]
    if not lengths:
        lengths = [max_new]

    print(f"\nDevice: {device}")
    print(f"DIMBA: {num_steps} NFE (Euler steps) per sequence, batch=1")
    print(f"SmolLM: L sequential AR steps with KV cache, batch=1")
    print(f"Warmup=1 run (untimed), timed=3 runs, mean reported.\n")
    print(
        "IMPORTANT: DIMBA generates the full response in one parallel diffusion pass "
        "(NFE = num_diffusion_steps). SmolLM generates one token at a time with KV "
        "cache. Wall-clock reflects very different compute patterns."
    )
    print()

    # Table header
    col_w = 14
    header = (
        f"{'Output len':>10}  "
        f"{'DIMBA (s)':>{col_w}}  {'DIMBA tok/s':>{col_w}}  "
        f"{'SmolLM (s)':>{col_w}}  {'SmolLM tok/s':>{col_w}}  "
        f"{'speedup':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in lengths:
        d_t, d_tps = _time_dimba(
            dimba_model, dimba_tok, prompt_ids_dimba, seq_len, num_steps, use_flow, device
        )
        s_t, s_tps = _time_smollm(smollm, smollm_tok, prompt_ids_smollm, seq_len, device)
        speedup = d_tps / s_tps

        print(
            f"{seq_len:>10}  "
            f"{d_t:>{col_w}.3f}  {d_tps:>{col_w}.1f}  "
            f"{s_t:>{col_w}.3f}  {s_tps:>{col_w}.1f}  "
            f"{speedup:>{col_w}.2f}x"
        )

    print()
    print("Notes:")
    print("  * DIMBA NFE = number of Euler/DDIM steps (--steps), not output length.")
    print("  * SmolLM AR steps = output length (each step produces one token).")
    print("  * speedup > 1 means DIMBA is faster in tokens/sec on this device.")
    print("  * At very short lengths DIMBA has fixed overhead; at long lengths")
    print("    its parallel denoising shines vs. SmolLM's sequential cost.")


# ── Task 3: perplexity proxy ──────────────────────────────────────────────────

@torch.no_grad()
def _dimba_denoising_nll(model, ids, device, n_noise=2, t_stride_frac=0.05, seed=42):
    """Return mean denoising NLL (nats/token) averaged uniformly over timesteps.

    This is the ELBO denoising proxy from perplexity_eval.py, adapted for the
    new-style train_4090 checkpoint format (uses model.num_diffusion_steps directly).
    """
    T = model.num_diffusion_steps
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t_stride = max(1, int(T * t_stride_frac))
    timesteps = list(range(0, T, t_stride))

    tot_nll, tot_tok = 0.0, 0
    N, L = ids.shape
    BS = 16  # conservative for CPU / low-VRAM

    for t_idx in timesteps:
        for _ in range(n_noise):
            for b0 in range(0, N, BS):
                batch = ids[b0: b0 + BS].to(device)
                t = torch.full((batch.shape[0],), t_idx, dtype=torch.long, device=device)
                noise = torch.randn(
                    *batch.shape, model.d_latent,
                    generator=gen
                ).to(device)
                x_pred, _, _ = model(batch, t, noise=noise)
                logits = model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.reshape(-1),
                    reduction="sum",
                )
                tot_nll += nll.item()
                tot_tok += batch.numel()

    return tot_nll / max(tot_tok, 1)


@torch.no_grad()
def _smollm_ar_ppl(lm, tokenizer, text: str, device: torch.device, stride: int = 512) -> float:
    """Standard stride-based AR perplexity for a causal LM."""
    enc = tokenizer.encode(text)
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    max_len = lm.config.max_position_embeddings if hasattr(lm.config, "max_position_embeddings") else 2048
    max_len = min(max_len, 2048)

    nll_sum, n_tokens = 0.0, 0
    for begin in range(0, len(ids), stride):
        chunk = ids[begin: begin + max_len]
        if len(chunk) < 2:
            break
        with torch.no_grad():
            out = lm(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        nll_sum += out.loss.item() * (len(chunk) - 1)
        n_tokens += len(chunk) - 1

    return math.exp(nll_sum / max(n_tokens, 1))


def run_ppl(
    dimba_model,
    dimba_tok,
    smollm,
    smollm_tok,
    device: torch.device,
    seed: int = 42,
):
    print("\n" + "=" * 70)
    print("TASK 3: PERPLEXITY PROXY")
    print("=" * 70)
    print()
    print("WARNING: DIMBA's 'perplexity' and SmolLM's perplexity are computed")
    print("  by fundamentally different methods and CANNOT be compared head-to-head.")
    print()
    print("  * DIMBA: noise-level reconstruction NLL averaged over timesteps.")
    print("    This is a denoising ELBO term — a proxy for how well the model")
    print("    can reconstruct tokens from their noised latents. It is lower when")
    print("    the model is better at denoising, but it is NOT the same quantity")
    print("    as AR perplexity. Think of it as an 'ELBO proxy ppl'.")
    print()
    print("  * SmolLM: standard autoregressive perplexity = exp(mean -log p(token|ctx)).")
    print("    Directly measures next-token prediction quality.")
    print()
    print("  Both are shown as within-model quality indicators, not a competition.")
    print()

    # Tokenize the held-out text.
    enc_dimba = dimba_tok.encode(PPL_TEXT, add_special_tokens=False)
    seq_len = 64
    n = (len(enc_dimba) // seq_len) * seq_len
    if n < seq_len:
        print("  [warn] PPL text too short after tokenization; skipping DIMBA ppl.")
        dimba_ppl = float("nan")
    else:
        ids = torch.tensor(enc_dimba[:n], dtype=torch.long).view(-1, seq_len)
        print(f"  DIMBA: evaluating on {ids.shape[0]} windows x {seq_len} tokens ...")
        avg_nll = _dimba_denoising_nll(dimba_model, ids, device, n_noise=1,
                                        t_stride_frac=0.1, seed=seed)
        dimba_ppl = math.exp(avg_nll)
        print(f"  DIMBA denoising ELBO proxy: nll={avg_nll:.3f}  ppl={dimba_ppl:.2f}")

    print(f"  SmolLM: computing AR perplexity ...")
    smollm_ppl = _smollm_ar_ppl(smollm, smollm_tok, PPL_TEXT, device)
    print(f"  SmolLM AR perplexity: {smollm_ppl:.2f}")

    print()
    print("Summary:")
    print(f"  DIMBA  ELBO-proxy ppl : {dimba_ppl:.2f}  (lower=better; NOT comparable to AR ppl)")
    print(f"  SmolLM AR ppl         : {smollm_ppl:.2f}  (lower=better; standard metric)")
    print()
    print("  These numbers measure different things. Do not subtract or ratio them.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/sft/final.pt",
        help="Path to DIMBA .pt checkpoint (default: checkpoints/sft/final.pt)",
    )
    p.add_argument(
        "--baseline",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="HuggingFace model ID for the AR baseline (default: HuggingFaceTB/SmolLM-135M-Instruct)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device string (default: auto — cuda if available else cpu)",
    )
    p.add_argument(
        "--tasks",
        default="gen,speed",
        help="Comma-separated list of tasks to run: gen, speed, ppl (default: gen,speed)",
    )
    p.add_argument(
        "--block-cot",
        action="store_true",
        help=(
            "Use block_sample_from_model with <think>/</think> delimiters. "
            "Use this for SFT/GRPO checkpoints (instruct-tuned). "
            "Omit for base checkpoints."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of diffusion/ODE steps (default: 20). 15-30 is typical for flow matching.",
    )
    p.add_argument(
        "--max-new",
        type=int,
        default=64,
        help=(
            "Max response tokens to generate for gen/speed tasks (default: 64). "
            "Use 128+ for quality evaluation; use 16-32 for quick smoke tests."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tasks = {t.strip() for t in args.tasks.split(",")}
    valid_tasks = {"gen", "speed", "ppl"}
    unknown = tasks - valid_tasks
    if unknown:
        print(f"[error] Unknown tasks: {unknown}. Valid options: {valid_tasks}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("DIMBA vs SmolLM Evaluation Harness")
    print("=" * 70)
    print(f"  DIMBA checkpoint : {args.checkpoint}")
    print(f"  Baseline         : {args.baseline}")
    print(f"  Device           : {device}")
    print(f"  Tasks            : {', '.join(sorted(tasks))}")
    print(f"  Diffusion steps  : {args.steps}")
    print(f"  Max new tokens   : {args.max_new}")
    print(f"  Block-CoT        : {args.block_cot}")
    print(f"  Seed             : {args.seed}")
    print()

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    dimba_model, dimba_cfg, use_flow = load_dimba(args.checkpoint, device)
    dimba_tok = load_dimba_tokenizer(args.baseline)  # SmolLM tokenizer for DIMBA decode
    smollm, smollm_tok = load_smollm(args.baseline, device)
    print()

    # ── think token setup (for block-CoT) ─────────────────────────────────────
    think_start_id: Optional[int] = None
    think_end_id: Optional[int] = None
    if args.block_cot:
        think_start_id, think_end_id = _add_think_tokens(dimba_tok, dimba_model)
        print(f"  Block-CoT: think_start_id={think_start_id} think_end_id={think_end_id}")
        print()

    # ── run tasks ─────────────────────────────────────────────────────────────
    if "gen" in tasks:
        run_gen(
            dimba_model, dimba_tok, smollm, smollm_tok,
            num_steps=args.steps,
            max_new=args.max_new,
            use_flow=use_flow,
            block_cot=args.block_cot,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            device=device,
        )

    if "speed" in tasks:
        run_speed(
            dimba_model, dimba_tok, smollm, smollm_tok,
            num_steps=args.steps,
            max_new=args.max_new,
            use_flow=use_flow,
            device=device,
        )

    if "ppl" in tasks:
        run_ppl(
            dimba_model, dimba_tok, smollm, smollm_tok,
            device=device,
            seed=args.seed,
        )

    # ── final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  DIMBA checkpoint : {args.checkpoint}")
    print(f"  Sampler          : {'flow/euler' if use_flow else 'ddim'} ({args.steps} steps)")
    print(f"  Block-CoT        : {args.block_cot}")
    print(f"  Baseline         : {args.baseline}")
    print()
    print("RECOMMENDED COMMANDS FOR THE GPU BOX")
    print()
    print("  # After SFT finishes (instruct checkpoint, with block-CoT):")
    print("  python scripts/eval_vs_smollm.py \\")
    print("      --checkpoint checkpoints/sft/final.pt \\")
    print("      --tasks gen,speed,ppl \\")
    print("      --block-cot --steps 20 --max-new 128")
    print()
    print("  # After GRPO finishes (policy-optimised checkpoint):")
    print("  python scripts/eval_vs_smollm.py \\")
    print("      --checkpoint checkpoints/grpo/final.pt \\")
    print("      --tasks gen,speed,ppl \\")
    print("      --block-cot --steps 15 --max-new 128")


if __name__ == "__main__":
    main()
