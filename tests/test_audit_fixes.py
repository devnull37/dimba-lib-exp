"""Regression tests for the correctness/quality fixes from the repo audit.

Covers:
  * latent_scale is persisted through a state_dict round-trip (was lost on reload).
  * DenoisingHead optional norm + learnable logit scale sharpens logits.
  * top-p nucleus filtering keeps the token that crosses the threshold.
  * the DDIM final step (acp~1) returns x0_hat instead of dividing by ~0.
  * predict_token_logits converts v->x0 before decoding (v-prediction).
"""

import math
import torch

from dimba import DIMBA
from dimba.diffusion.sampling import top_k_top_p_filtering, _ddim_step


def _tiny(**kw):
    cfg = dict(
        vocab_size=64, d_model=32, d_prompt=16, num_diffusion_steps=16,
        num_denoiser_layers=1, d_state=8, latent_diffusion=True, d_latent=16,
        use_weight_tying=True, use_simple_mamba=True,
    )
    cfg.update(kw)
    return DIMBA(**cfg)


def test_latent_scale_persists_through_state_dict():
    m = _tiny()
    ids = torch.randint(0, 64, (2, 8))
    m.calibrate_latent_scale(ids)
    scale = m.latent_scale
    assert scale != 1.0  # calibration actually changed it

    sd = m.state_dict()
    assert "_latent_scale" in sd  # saved as a buffer, not a plain float

    m2 = _tiny()
    assert m2.latent_scale == 1.0  # fresh model starts at the default
    m2.load_state_dict(sd)
    assert abs(m2.latent_scale - scale) < 1e-5  # restored on reload


def test_head_norm_adds_logit_scale_and_sharpens():
    plain = _tiny(use_head_norm=False)
    normed = _tiny(use_head_norm=True)
    assert not hasattr(plain.output_head, "logit_scale")
    assert hasattr(normed.output_head, "logit_scale")
    # exp(init) == sqrt(d_model) -> logits get a meaningful temperature.
    assert math.isclose(normed.output_head.logit_scale.exp().item(), math.sqrt(32), rel_tol=1e-4)

    x = torch.randn(2, 8, 32)
    ew = normed.token_embed.get_weight()
    lo_plain = plain.output_head(x, embedding_weight=ew)
    lo_norm = normed.output_head(x, embedding_weight=ew)
    # The normed head produces a wider logit range (sharper distribution).
    assert lo_norm.std() > lo_plain.std()


def test_top_p_keeps_crossing_token():
    # softmax(logits) == [0.5, 0.4, 0.1]; with top_p=0.5 the nucleus must keep the
    # 0.4 token (it crosses the threshold), i.e. 2 finite logits, not 1.
    logits = torch.log(torch.tensor([[0.5, 0.4, 0.1]]))
    filt = top_k_top_p_filtering(logits.clone(), top_k=None, top_p=0.5)
    assert torch.isfinite(filt).sum().item() == 2


def test_ddim_final_step_returns_x0():
    x_t = torch.randn(1, 4, 8)
    x0 = torch.randn(1, 4, 8)
    acp_t = torch.tensor(1.0)        # final (cleanest) step
    acp_prev = torch.tensor(1.0)
    out = _ddim_step(x_t, x0, acp_t, acp_prev, eta=0.0)
    assert torch.allclose(out, x0)   # no (x_t-x0)/~0 blowup
    assert torch.isfinite(out).all()


def test_predict_token_logits_v_prediction_runs():
    m = _tiny(prediction_type="v")
    ids = torch.randint(0, 64, (2, 8))
    t = torch.randint(0, 16, (2,))
    logits = m.predict_token_logits(ids, t)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()
