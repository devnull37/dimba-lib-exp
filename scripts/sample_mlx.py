"""Sample from a DIMBA checkpoint on the Apple GPU via MLX.

Runs the *entire* DIMBA diffusion sampler on the Apple-Silicon GPU through MLX — ~17x faster
than PyTorch-MPS and ~44x faster than CPU for the dimbapeare1-30m Shakespeare model, with
token-identical output (see scripts/verify_mlx_model.py). Requires Apple Silicon + ``mlx``.

Supports both DDIM (cosine schedule) and flow-matching (Euler/Heun ODE) checkpoints.
The sampler is auto-selected from the checkpoint config: ``use_flow_matching=True`` ->
flow (``--steps`` = Euler/Heun steps, default 20); else DDIM (``--steps`` = DDIM steps).

Checkpoint formats handled:
  * Lightning ``.ckpt``: ``hyper_parameters.model_config`` + ``state_dict`` (``model.*``).
  * Plain ``.pt``: top-level ``config`` + ``model_state_dict`` (the d1-135m base format).

Tokenizers:
  * ``--tokenizer path.json`` — char or BPE tokenizer.json (auto-detected by content).
  * ``--hf-tokenizer NAME``   — a HuggingFace tokenizer via ``AutoTokenizer`` (default
    ``HuggingFaceTB/SmolLM-135M`` for the 49152-vocab BPE models).
  * neither                   — a char tokenizer sized to the model's vocab.

Examples:
    # Flow + block_ffn base checkpoint (SmolLM BPE, Euler 20 steps):
    python scripts/sample_mlx.py \
        --ckpt ~/.cache/huggingface/hub/models--devnull37--d1-135m-base/snapshots/3137559530d534726845ce28bb36002102fe0ae4/final.pt \
        --hf-tokenizer HuggingFaceTB/SmolLM-135M --prompt "The meaning of life is" \
        --seq-len 64 --steps 20 --sampler euler --temperature 0.9

    # Old Shakespeare DDIM checkpoint (char tokenizer):
    HF_TOKEN=hf_... python scripts/sample_mlx.py --num-samples 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from dimba.models.diffusion import DIMBA
from dimba.backends.mlx.model import MLXDIMBA
from dimba.tokenizers.simple import SimpleCharacterTokenizer


def _load_tokenizer_auto(path):
    """Load a char or BPE tokenizer from a tokenizer.json, auto-detected by content."""
    import json
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "char_to_id" in data:
        from dimba.tokenizers.simple import SimpleCharacterTokenizer
        tok = SimpleCharacterTokenizer()
        tok.load(path)
        return tok
    from dimba.tokenizers.bpe import BPETokenizer
    tok = BPETokenizer()
    tok.load(path)
    return tok


class _HFTokenizerAdapter:
    """Wrap a HuggingFace ``AutoTokenizer`` with the encode/decode list API the
    sampler expects (token-id lists, no special tokens added)."""

    def __init__(self, name):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)


def load_checkpoint(args):
    """Load a DIMBA from a Lightning ``.ckpt`` or a plain ``.pt`` checkpoint.

    Returns the constructed (eval) torch ``DIMBA``.
    """
    if args.ckpt and os.path.exists(args.ckpt):
        path = args.ckpt
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.repo, args.file, token=args.token)
    c = torch.load(path, map_location="cpu", weights_only=False)

    # Format A: plain .pt with top-level `config` + `model_state_dict` (d1-135m base).
    if "config" in c and ("model_state_dict" in c or "state_dict" in c):
        cfg = dict(c["config"])
        sd = c.get("model_state_dict", c.get("state_dict"))
        # Strip a leading "model." prefix if present (some exporters add one).
        if sd and all(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items()}
    # Format B: Lightning .ckpt with `hyper_parameters.model_config` + `state_dict`.
    elif "hyper_parameters" in c:
        cfg = dict(c["hyper_parameters"]["model_config"])
        cfg["vocab_size"] = c["hyper_parameters"]["vocab_size"]
        sd = {k[len("model."):]: v for k, v in c["state_dict"].items() if k.startswith("model.")}
    else:
        raise ValueError(
            "Unrecognized checkpoint format: expected top-level 'config'+'model_state_dict' "
            "or 'hyper_parameters'+'state_dict'."
        )

    # `latent_scale` is restored from the persisted buffer; drop the config copy so the
    # constructor doesn't fight the buffer load (it's not a plain constructor kwarg path).
    cfg.pop("latent_scale", None)
    m = DIMBA(**cfg).eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected state-dict keys (showing 5): {unexpected[:5]}")
    # Ignore the non-persistent tied-embedding buffer in `missing` (it is re-derived).
    missing = [k for k in missing if not k.endswith("embedding_weight")]
    if missing:
        print(f"  [warn] {len(missing)} missing state-dict keys (showing 5): {missing[:5]}")
    return m


def _build_tokenizer(args, model):
    if args.tokenizer:
        return _load_tokenizer_auto(args.tokenizer)
    if args.hf_tokenizer:
        return _HFTokenizerAdapter(args.hf_tokenizer)
    return SimpleCharacterTokenizer(vocab_size=model.vocab_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="Local checkpoint path (.ckpt or .pt); else download from HF.")
    ap.add_argument("--repo", default="devnull37/dimbapeare1-30m")
    ap.add_argument("--file", default="shakespeare1.ckpt")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=None,
                    help="Sampler steps. DDIM: defaults to model T; flow: defaults to 20.")
    ap.add_argument("--sampler", default=None, choices=["euler", "heun", "ddim"],
                    help="Override the sampler (default: auto from config — flow uses euler).")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax.")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="Optional conditioning prompt text.")
    ap.add_argument("--tokenizer", default=None,
                    help="Path to tokenizer.json (char or BPE; auto-detected by content).")
    ap.add_argument("--hf-tokenizer", default=None,
                    help="HuggingFace tokenizer name, e.g. 'HuggingFaceTB/SmolLM-135M'.")
    args = ap.parse_args()

    print(f"Loading {args.ckpt or args.repo + '/' + args.file} ...")
    m = load_checkpoint(args)
    mlx_model = MLXDIMBA.from_torch(m)
    tok = _build_tokenizer(args, m)

    is_flow = bool(mlx_model.use_flow_matching)
    # Flow sampler chooses euler/heun; the DDIM path ignores `sampler`.
    flow_sampler = "euler"
    if args.sampler in ("euler", "heun"):
        flow_sampler = args.sampler
    mode = (f"flow-{flow_sampler}" if is_flow else "ddim")

    prompt_ids = None
    if args.prompt:
        prompt_ids = np.asarray([tok.encode(args.prompt)], dtype=np.int64)

    print(f"MLX-GPU sampling [{mode}]: {args.num_samples} x {args.seq_len} tokens,"
          f" steps={args.steps if args.steps is not None else ('20' if is_flow else 'T')},"
          f" temperature={args.temperature}, block_ffn={mlx_model.block_ffn}\n")
    import time
    for k in range(args.num_samples):
        t = time.time()
        ids = mlx_model.sample(
            args.seq_len, num_steps=args.steps, temperature=args.temperature,
            prompt_ids=prompt_ids, seed=args.seed + k,
            sampler=(flow_sampler if is_flow else None),
        )[0]
        dt = time.time() - t
        text = tok.decode(ids.tolist())
        if args.prompt:
            text = args.prompt + text
        print(f"--- sample {k + 1}/{args.num_samples}  ({dt:.2f}s) " + "-" * 30 + f"\n{text}\n")


if __name__ == "__main__":
    main()
