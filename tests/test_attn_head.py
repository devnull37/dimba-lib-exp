"""Tests for the context-aware attention rounding head in DenoisingHead / DIMBA.

Covers:
- DIMBA(head_type="attn") builds and produces finite logits of the correct shape.
- Default head_type="linear" path is unaffected (shape + finite check).
- Context-awareness: with the attn head, changing position i's latent changes
  the logits at position j != i.  A linear head cannot do this, confirming the
  attn head genuinely mixes information across positions.
"""

import pytest
import torch
from dimba.models.diffusion import DIMBA

# Small model dimensions to keep tests fast.
_VOCAB = 128
_D_MODEL = 64   # divisible by 4 (head_attn_heads default)
_SEQ = 8
_BATCH = 2


@pytest.fixture(scope="module")
def attn_model():
    """DIMBA with the attention rounding head (head_type='attn')."""
    m = DIMBA(
        vocab_size=_VOCAB,
        d_model=_D_MODEL,
        d_prompt=_D_MODEL,
        num_diffusion_steps=10,
        num_denoiser_layers=2,
        use_simple_mamba=True,
        head_type="attn",
        head_attn_layers=2,
        head_attn_heads=4,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def linear_model():
    """DIMBA with the default linear rounding head (head_type='linear')."""
    m = DIMBA(
        vocab_size=_VOCAB,
        d_model=_D_MODEL,
        d_prompt=_D_MODEL,
        num_diffusion_steps=10,
        num_denoiser_layers=2,
        use_simple_mamba=True,
        head_type="linear",
    )
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Build + forward shape tests
# ---------------------------------------------------------------------------

class TestAttnHeadBuildAndShape:
    def test_builds_without_error(self, attn_model):
        assert attn_model is not None

    def test_logits_shape(self, attn_model):
        """Forward pass returns [B, L, V] logits."""
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
        t = torch.zeros(_BATCH, dtype=torch.long)
        with torch.no_grad():
            x_pred, _, _ = attn_model(ids, t)
            logits = attn_model.output_head(x_pred)
        assert logits.shape == (_BATCH, _SEQ, _VOCAB), (
            f"Expected logits shape ({_BATCH}, {_SEQ}, {_VOCAB}), got {logits.shape}"
        )

    def test_logits_finite(self, attn_model):
        """All output logits must be finite (no NaN / Inf)."""
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
        t = torch.zeros(_BATCH, dtype=torch.long)
        with torch.no_grad():
            x_pred, _, _ = attn_model(ids, t)
            logits = attn_model.output_head(x_pred)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_head_type_stored_in_config(self, attn_model):
        """head_type and attn kwargs survive into model._config."""
        cfg = attn_model.config
        assert cfg["head_type"] == "attn"
        assert cfg["head_attn_layers"] == 2
        assert cfg["head_attn_heads"] == 4


class TestLinearHeadUnchanged:
    def test_logits_shape(self, linear_model):
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
        t = torch.zeros(_BATCH, dtype=torch.long)
        with torch.no_grad():
            x_pred, _, _ = linear_model(ids, t)
            logits = linear_model.output_head(x_pred)
        assert logits.shape == (_BATCH, _SEQ, _VOCAB)

    def test_logits_finite(self, linear_model):
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
        t = torch.zeros(_BATCH, dtype=torch.long)
        with torch.no_grad():
            x_pred, _, _ = linear_model(ids, t)
            logits = linear_model.output_head(x_pred)
        assert torch.isfinite(logits).all()

    def test_head_type_stored_in_config(self, linear_model):
        assert linear_model.config["head_type"] == "linear"


# ---------------------------------------------------------------------------
# Context-awareness test
# ---------------------------------------------------------------------------

class TestContextAwareness:
    """Changing position i's decoded embedding must change logits at position j!=i.

    A pure per-position linear head cannot propagate information across positions,
    so this test distinguishes the attention head from the linear baseline.
    """

    def _get_logits_from_decoded(self, model, decoded: torch.Tensor) -> torch.Tensor:
        """Run only the output_head on a pre-decoded [1, L, d_model] tensor."""
        with torch.no_grad():
            return model.output_head(decoded)

    def _build_decoded(self, model) -> torch.Tensor:
        """Get a representative decoded tensor by running a real forward pass."""
        ids = torch.randint(0, _VOCAB, (1, _SEQ))
        t = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            x_pred, _, _ = model(ids, t)
        return x_pred.detach()

    def test_attn_head_context_aware(self, attn_model):
        """Perturbing position i changes logits at position j (i != j)."""
        decoded = self._build_decoded(attn_model)

        i, j = 0, _SEQ - 1   # first and last positions
        assert i != j

        logits_orig = self._get_logits_from_decoded(attn_model, decoded)

        # Perturb position i.
        decoded_perturbed = decoded.clone()
        decoded_perturbed[0, i, :] = decoded_perturbed[0, i, :] + 1.0

        logits_perturbed = self._get_logits_from_decoded(attn_model, decoded_perturbed)

        diff_at_j = (logits_perturbed[0, j] - logits_orig[0, j]).abs().max().item()
        assert diff_at_j > 0.0, (
            "Attn head: logits at position j did not change when position i was "
            f"perturbed (max diff = {diff_at_j}).  The head is not context-aware."
        )

    def test_linear_head_not_context_aware(self, linear_model):
        """Verify the linear baseline cannot propagate across positions (sanity check)."""
        decoded = self._build_decoded(linear_model)

        i, j = 0, _SEQ - 1
        logits_orig = self._get_logits_from_decoded(linear_model, decoded)

        decoded_perturbed = decoded.clone()
        decoded_perturbed[0, i, :] = decoded_perturbed[0, i, :] + 1.0

        logits_perturbed = self._get_logits_from_decoded(linear_model, decoded_perturbed)

        diff_at_j = (logits_perturbed[0, j] - logits_orig[0, j]).abs().max().item()
        # A linear head: perturbing position i must NOT change position j.
        assert diff_at_j == 0.0, (
            "Linear head: logits at position j changed when only position i was "
            f"perturbed (max diff = {diff_at_j}). Something is wrong with the baseline."
        )
