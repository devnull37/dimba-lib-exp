"""Sample from a DIMBA checkpoint on the Apple GPU via MLX.

Runs the *entire* DIMBA diffusion sampler on the Apple-Silicon GPU through MLX — ~17x faster
than PyTorch-MPS and ~44x faster than CPU for the dimbapeare1-30m Shakespeare model, with
token-identical output (see scripts/verify_mlx_model.py). Requires Apple Silicon + ``mlx``.

Examples:
    HF_TOKEN=hf_... python scripts/sample_mlx.py --num-samples 3
    python scripts/sample_mlx.py --ckpt path/to/shakespeare1.ckpt --prompt "ROMEO:" --temperature 0.9
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
    import json
    with open(path) as f: data = json.load(f)
    if isinstance(data, dict) and "char_to_id" in data:
        from dimba.tokenizers.simple import SimpleCharacterTokenizer
        tok = SimpleCharacterTokenizer(); tok.load(path); return tok
    from dimba.tokenizers.bpe import BPETokenizer
    tok = BPETokenizer(); tok.load(path); return tok


def load_checkpoint(args):
    if args.ckpt and os.path.exists(args.ckpt):
        path = args.ckpt
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.repo, args.file, token=args.token)
    c = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(c["hyper_parameters"]["model_config"])
    cfg["vocab_size"] = c["hyper_parameters"]["vocab_size"]
    sd = {k[len("model."):]: v for k, v in c["state_dict"].items() if k.startswith("model.")}
    m = DIMBA(**cfg).eval()
    m.load_state_dict(sd, strict=False)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="Local checkpoint path (else download from HF).")
    ap.add_argument("--repo", default="devnull37/dimbapeare1-30m")
    ap.add_argument("--file", default="shakespeare1.ckpt")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64, help="DDIM steps (<= model T).")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax.")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="Optional conditioning prompt (chars).")
    ap.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (char or BPE); auto-detected by file content.")
    args = ap.parse_args()

    print(f"Loading {args.ckpt or args.repo + '/' + args.file} ...")
    m = load_checkpoint(args)
    mlx_model = MLXDIMBA.from_torch(m)
    if args.tokenizer:
        tok = _load_tokenizer_auto(args.tokenizer)
    else:
        tok = SimpleCharacterTokenizer(vocab_size=m.vocab_size)

    prompt_ids = None
    if args.prompt:
        prompt_ids = np.asarray([tok.encode(args.prompt)], dtype=np.int64)

    print(f"MLX-GPU sampling: {args.num_samples} x {args.seq_len} tokens, {args.steps} steps,"
          f" temperature={args.temperature}\n")
    import time
    for k in range(args.num_samples):
        t = time.time()
        ids = mlx_model.sample(args.seq_len, args.steps, temperature=args.temperature,
                               prompt_ids=prompt_ids, seed=args.seed + k)[0]
        dt = time.time() - t
        text = tok.decode(ids.tolist())
        if args.prompt:
            text = args.prompt + text
        print(f"--- sample {k + 1}/{args.num_samples}  ({dt:.2f}s) " + "-" * 30 + f"\n{text}\n")


if __name__ == "__main__":
    main()
