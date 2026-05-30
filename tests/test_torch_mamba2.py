"""Tests for the pure-PyTorch Mamba-2 (SSD) mixer, ``TorchMamba2``.

These run on any machine (no CUDA / ``mamba_ssm`` needed). They verify:
  * the fast chunked scan matches the sequential reference,
  * the parameter tree matches ``mamba_ssm.Mamba2`` (names + shapes) so a CUDA-trained
    checkpoint loads with ``strict=True``,
  * the forward is finite and shape-correct.
"""

import pytest
import torch

from dimba.models.torch_mamba2 import TorchMamba2

# The DIMBA Shakespeare (dimbapeare1-30m) mixer config.
CFG = dict(d_model=384, d_state=128, d_conv=4, expand=2)

# Exact parameter names a mamba_ssm.Mamba2 checkpoint exposes per mixer.
EXPECTED_KEYS = {
    "in_proj.weight", "conv1d.weight", "conv1d.bias",
    "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight",
}


def _randomize(m):
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


@pytest.mark.parametrize("L", [50, 128, 200, 300])
def test_chunked_matches_sequential(L):
    torch.manual_seed(0)
    m = _randomize(TorchMamba2(chunk_size=128, **CFG).eval())
    x = torch.randn(2, L, CFG["d_model"])
    with torch.no_grad():
        m.use_chunked = False
        y_seq = m(x)
        m.use_chunked = True
        y_chk = m(x)
    assert torch.isfinite(y_chk).all()
    assert torch.allclose(y_seq, y_chk, atol=1e-4, rtol=1e-4)


def test_param_tree_matches_mamba_ssm():
    m = TorchMamba2(**CFG)
    keys = set(m.state_dict().keys())
    assert keys == EXPECTED_KEYS, f"unexpected param tree: {keys ^ EXPECTED_KEYS}"
    # Shapes implied by the config (d_inner=768, nheads=12, conv_dim=1024, d_in_proj=1804).
    sd = m.state_dict()
    assert sd["in_proj.weight"].shape == (1804, 384)
    assert sd["conv1d.weight"].shape == (1024, 1, 4)
    assert sd["dt_bias"].shape == (12,)
    assert sd["A_log"].shape == (12,)
    assert sd["D"].shape == (12,)
    assert sd["norm.weight"].shape == (768,)
    assert sd["out_proj.weight"].shape == (384, 768)


def test_strict_state_dict_roundtrip():
    a = _randomize(TorchMamba2(**CFG))
    b = TorchMamba2(**CFG)
    res = b.load_state_dict(a.state_dict(), strict=True)  # raises if any mismatch
    assert res.missing_keys == [] and res.unexpected_keys == []


def test_forward_shape_and_finite():
    m = _randomize(TorchMamba2(**CFG).eval())
    with torch.no_grad():
        out = m(torch.randn(3, 70, CFG["d_model"]))
    assert out.shape == (3, 70, CFG["d_model"])
    assert torch.isfinite(out).all()
