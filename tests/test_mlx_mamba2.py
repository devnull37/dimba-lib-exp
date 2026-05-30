"""MLX (Apple-GPU) Mamba-2 (SSD) parity tests.

Skipped automatically when ``mlx`` is not installed (i.e. everywhere except Apple Silicon),
so the suite stays green on CI/Linux. On a Mac these assert the MLX mixer is weight-compatible
with and numerically equal to the PyTorch reference :class:`TorchMamba2`.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # skips the whole module without MLX

from dimba.models.torch_mamba2 import TorchMamba2
from dimba.backends.mlx.mamba2 import MLXMamba2Mixer, load_torch_mamba2_state_dict

CFG = dict(d_model=384, d_state=128, d_conv=4, expand=2)  # dimbapeare1-30m mixer config


def _randomized_torch_mixer():
    torch.manual_seed(0)
    m = TorchMamba2(**CFG).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


@pytest.mark.parametrize("L", [64, 200, 300])
def test_mlx_mixer_matches_torch(L):
    """Same weights + same input -> MLX output matches TorchMamba2 (the chunked-scan path)."""
    tm = _randomized_torch_mixer()
    mm = MLXMamba2Mixer(**CFG)
    load_torch_mamba2_state_dict(mm, tm.state_dict())

    x = np.random.RandomState(0).randn(2, L, CFG["d_model"]).astype(np.float32)
    with torch.no_grad():
        y_torch = tm(torch.from_numpy(x)).numpy()
    y_mlx = np.array(mm(mx.array(x)))

    assert y_mlx.shape == y_torch.shape
    assert np.isfinite(y_mlx).all()
    assert np.abs(y_torch - y_mlx).max() < 1e-3


def test_mlx_param_tree_matches_mamba_ssm_names():
    """The MLX mixer exposes the same loadable parameter names as a mamba_ssm checkpoint."""
    mm = MLXMamba2Mixer(**CFG)
    names = {k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(mm.parameters())}
    expected = {
        "in_proj.weight", "conv1d.weight", "conv1d.bias",
        "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight",
    }
    assert expected <= names, f"missing MLX params: {expected - names}"
