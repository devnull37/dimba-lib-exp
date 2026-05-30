"""Verify + benchmark the full-model MLX (Apple-GPU) DIMBA sampler against PyTorch.

Reconstructs the PyTorch ``dimbapeare1-30m`` model, copies its weights into ``MLXDIMBA``,
runs the *same* deterministic DDIM trajectory (identical initial noise, eta=0) through both,
and checks the final response logits match. Then times a full sample on torch-CPU,
torch-MPS, and MLX-GPU, and prints an MLX-generated Shakespeare sample.

Usage: HF_TOKEN=... python scripts/verify_mlx_model.py   (token also read from the constant below)
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from dimba.models.diffusion import DIMBA
from dimba.diffusion.sampling import _make_timesteps, _ddim_step
from dimba.backends.mlx.model import MLXDIMBA
from dimba.tokenizers.simple import SimpleCharacterTokenizer
import mlx.core as mx

TOKEN = os.environ.get("HF_TOKEN")
SEQ_LEN, NUM_STEPS = 256, 64


def load_torch_model():
    ckpt = hf_hub_download("devnull37/dimbapeare1-30m", "shakespeare1.ckpt", token=TOKEN)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(c["hyper_parameters"]["model_config"])
    cfg["vocab_size"] = c["hyper_parameters"]["vocab_size"]
    sd = {k[len("model."):]: v for k, v in c["state_dict"].items() if k.startswith("model.")}
    m = DIMBA(**cfg).eval()
    res = m.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys[:5]}"
    # missing should only be non-persistent buffers, if any
    miss = [k for k in res.missing_keys if "alphas_cumprod" not in k]
    assert not miss, f"missing: {miss[:8]}"
    return m


@torch.no_grad()
def torch_sample_logits(m, noise, num_steps, device):
    m = m.to(device)
    x_t = torch.from_numpy(noise).to(device)
    cond = m.conditioning_from_prompt(None, noise.shape[0], torch.device(device))
    acp = m.get_alphas_cumprod().to(device)
    ts = _make_timesteps(m.num_diffusion_steps, num_steps, torch.device(device))
    for i in range(len(ts)):
        t = torch.full((noise.shape[0],), int(ts[i].item()), dtype=torch.long, device=device)
        x0 = m.denoise_to_x0_latent(x_t, t, cond, None)
        acp_t = acp[ts[i]]
        acp_prev = acp[ts[i + 1]] if i < len(ts) - 1 else torch.ones((), device=device)
        x_t = _ddim_step(x_t, x0, acp_t, acp_prev, 0.0)
    x_dec = m.decode_latent(x_t)
    return m.output_head(x_dec)


def main():
    print("loading dimbapeare1-30m (reconstruct + strict load) ...")
    m = load_torch_model()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  torch model loaded: {n_params:,} params, d_latent={m.d_latent}, T={m.num_diffusion_steps}")
    print("building MLXDIMBA.from_torch ...")
    mlx_model = MLXDIMBA.from_torch(m)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1, SEQ_LEN, m.d_latent)).astype(np.float32)

    # ---- numerical parity (identical noise, deterministic DDIM) ----
    lt = torch_sample_logits(m, noise, NUM_STEPS, "cpu").numpy()
    lm = np.array(mlx_model.sample_logits(noise, NUM_STEPS))
    dmax = np.abs(lt - lm).max()
    agree = float((lt.argmax(-1) == lm.argmax(-1)).mean())
    print(f"\n[PARITY @ {NUM_STEPS} steps] logits max|Δ| = {dmax:.3e} | argmax-token agreement = {agree*100:.2f}%"
          f" -> {'PASS' if agree > 0.99 else 'CHECK'}")

    # ---- benchmark a full sample on each backend ----
    def bt(device):
        torch.manual_seed(0)
        t = time.time(); torch_sample_logits(m, noise, NUM_STEPS, device)
        if device == "mps": torch.mps.synchronize()
        return time.time() - t
    t_cpu = bt("cpu")
    t_mps = bt("mps") if torch.backends.mps.is_available() else float("nan")
    _ = mlx_model.sample_logits(noise, NUM_STEPS)  # warmup
    t0 = time.time(); mlx_model.sample_logits(noise, NUM_STEPS); t_mlx = time.time() - t0
    print(f"\n[FULL SAMPLE  seq_len={SEQ_LEN}, steps={NUM_STEPS}]")
    print(f"   torch-CPU  {t_cpu:6.2f} s")
    print(f"   torch-MPS  {t_mps:6.2f} s   ({t_cpu/t_mps:.1f}x vs CPU)")
    print(f"   MLX-GPU    {t_mlx:6.2f} s   ({t_cpu/t_mlx:.1f}x vs CPU, {t_mps/t_mlx:.1f}x vs MPS)")

    # ---- show an MLX-generated Shakespeare sample (decoded with the real tokenizer) ----
    tok = SimpleCharacterTokenizer(vocab_size=m.vocab_size)
    ids = mlx_model.sample(SEQ_LEN, NUM_STEPS, temperature=0.8, seed=1)[0]
    print("\n[MLX-GPU sample, temp=0.8]:\n" + "-" * 60 + f"\n{tok.decode(ids.tolist())}\n" + "-" * 60)


if __name__ == "__main__":
    main()
