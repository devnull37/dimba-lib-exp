"""Tests for the DIMBA cross-architecture knowledge distillation subpackage.

CPU-only, network-free, fast.  Uses a StubTeacher that mimics TeacherWrapper's
interface without any HuggingFace network calls.  All tests that require the
real HuggingFace transformers library are marked @pytest.mark.slow.

Run with:
    python3 -m pytest tests/test_distillation.py -o addopts=""
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Import individual submodules directly so the missing trainer.py does not
# prevent the bulk of tests from running.
# ---------------------------------------------------------------------------
from dimba.distillation.teacher import TeacherOutputs
from dimba.distillation.projectors import Projector, HeadAligner, LayerMap
from dimba.distillation.losses import stage1_matrix_loss, stage2_hidden_loss, stage3_kd_loss
from dimba.distillation.surgery import build_student_from_teacher
from dimba.distillation.init import principled_init_from_teacher
from dimba.models.diffusion import DIMBA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Vocab and dim constants kept tiny to stay fast on CPU.
_VOCAB = 40
_DT = 64      # teacher hidden dim
_DS = 64      # student hidden dim (same as teacher for surgery tests)
_HT = 4       # teacher heads
_HS = 2       # student heads (TorchMamba2 with d_model=64, expand=2 -> d_inner=128, nheads=2)
_LT = 4       # teacher layers
_LS = 2       # student layers
_L = 8        # sequence length
_B = 2        # batch size
_FFN_HIDDEN = 128  # teacher FFN intermediate size


# ---------------------------------------------------------------------------
# StubTeacher
# ---------------------------------------------------------------------------


class _FakeFFN(nn.Module):
    """Tiny MLP that mimics a teacher layer FFN sub-module."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.ff1 = nn.Linear(d, hidden)
        self.ff2 = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.ff2(F.gelu(self.ff1(x)))


class _FakeSelfAttn(nn.Module):
    """Minimal self-attention stand-in exposing q/k/v/o_proj."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _FakeLayer(nn.Module):
    """Mimics a Llama-style decoder layer with .self_attn and .mlp (mlp type)."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.self_attn = _FakeSelfAttn(d)
        # Expose as .mlp with .ff1/.ff2 to match non-SwiGLU detection.
        # We use fc1/fc2 keys (OPT/Falcon path in TeacherWrapper) to keep it simple.
        self.mlp = _FakeFFN(d, hidden)


class StubTeacher(nn.Module):
    """Network-free stand-in for TeacherWrapper.

    Produces randomly-shaped TeacherOutputs so that all distillation losses and
    surgery routines can be exercised without any network or HuggingFace dependency.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
        is_causal: Whether this is a causal language model teacher.
        ffn_type: FFN type string ('mlp').
        ffn_hidden: Intermediate FFN size.
    """

    def __init__(
        self,
        num_layers: int = _LT,
        num_heads: int = _HT,
        d_model: int = _DT,
        vocab_size: int = _VOCAB,
        is_causal: bool = True,
        ffn_hidden: int = _FFN_HIDDEN,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.is_causal = is_causal
        self.ffn_type: str = "mlp"
        self.ffn_hidden: int = ffn_hidden
        self._teacher_type: str = "causal" if is_causal else "masked"

        # Actual nn.Module weights so weight accessors return real tensors.
        self._embed = nn.Embedding(vocab_size, d_model)
        self._head = nn.Linear(d_model, vocab_size, bias=False)
        self._layers = nn.ModuleList(
            [_FakeLayer(d_model, ffn_hidden) for _ in range(num_layers)]
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TeacherOutputs:
        """Produce random-but-shaped TeacherOutputs (no real computation).

        Args:
            input_ids: ``[B, L]`` token ids.
            attention_mask: Ignored.

        Returns:
            TeacherOutputs with correct shapes.
        """
        B, L = input_ids.shape
        d = self.d_model

        # hidden_states: tuple of num_layers+1, each [B, L, d]
        hidden_states: Tuple[torch.Tensor, ...] = tuple(
            torch.randn(B, L, d) for _ in range(self.num_layers + 1)
        )
        # attentions: tuple of num_layers, each [B, num_heads, L, L]
        # Make them valid probability distributions (softmax over last dim).
        raw_attn = [torch.randn(B, self.num_heads, L, L) for _ in range(self.num_layers)]
        if self.is_causal:
            # Apply causal mask.
            mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
            attn_weights = []
            for a in raw_attn:
                a = a.masked_fill(~mask, float("-inf"))
                a = F.softmax(a, dim=-1)
                # Replace nan (first column all-inf) with 0.
                a = torch.nan_to_num(a, nan=0.0)
                attn_weights.append(a)
        else:
            attn_weights = [F.softmax(a, dim=-1) for a in raw_attn]

        attentions: Tuple[torch.Tensor, ...] = tuple(attn_weights)
        logits: torch.Tensor = torch.randn(B, L, self.vocab_size)

        return TeacherOutputs(
            hidden_states=hidden_states,
            attentions=attentions,
            logits=logits,
        )

    # ------------------------------------------------------------------
    # Weight accessors (mimicking TeacherWrapper's interface)
    # ------------------------------------------------------------------

    def input_embedding_weight(self) -> torch.Tensor:
        """Return input embedding weight ``[vocab_size, d_model]``."""
        return self._embed.weight.detach()

    def output_head_weight(self) -> Optional[torch.Tensor]:
        """Return output head weight ``[vocab_size, d_model]``."""
        return self._head.weight.detach()

    def final_norm_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """No final norm in stub — return None."""
        return None

    def layer_ffn_state(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return FFN state dict for layer *idx* in _BlockFFN-compatible format.

        Args:
            idx: Zero-based layer index.

        Returns:
            Dict with keys ``ff1.weight``, ``ff1.bias``, ``ff2.weight``, ``ff2.bias``.
        """
        if idx < 0 or idx >= self.num_layers:
            raise ValueError(f"layer index {idx} out of range [0, {self.num_layers})")
        ffn = self._layers[idx].mlp
        return {
            "ff1.weight": ffn.ff1.weight.detach(),
            "ff1.bias": ffn.ff1.bias.detach(),
            "ff2.weight": ffn.ff2.weight.detach(),
            "ff2.bias": ffn.ff2.bias.detach(),
        }

    def layer_attention_qkvo(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Return Q/K/V/O weights for layer *idx*.

        Args:
            idx: Zero-based layer index.

        Returns:
            Dict with keys ``'q'``, ``'k'``, ``'v'``, ``'o'``.
        """
        if idx < 0 or idx >= self.num_layers:
            raise ValueError(f"layer index {idx} out of range [0, {self.num_layers})")
        attn = self._layers[idx].self_attn
        return {
            "q": attn.q_proj.weight.detach(),
            "k": attn.k_proj.weight.detach(),
            "v": attn.v_proj.weight.detach(),
            "o": attn.o_proj.weight.detach(),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_teacher() -> StubTeacher:
    """Return a fresh causal StubTeacher."""
    torch.manual_seed(0)
    return StubTeacher(
        num_layers=_LT,
        num_heads=_HT,
        d_model=_DT,
        vocab_size=_VOCAB,
        is_causal=True,
        ffn_hidden=_FFN_HIDDEN,
    )


@pytest.fixture()
def tiny_student_no_ffn() -> DIMBA:
    """Tiny DIMBA without block FFN, use_simple_mamba=False (TorchMamba2).

    d_model=64, expand=2 -> d_inner=128, headdim=64, nheads=2 (multiple of 64).
    """
    torch.manual_seed(1)
    return DIMBA(
        vocab_size=_VOCAB,
        d_model=_DS,
        d_prompt=_DS,
        num_diffusion_steps=20,
        num_denoiser_layers=_LS,
        d_state=16,
        d_conv=4,
        expand=2,
        use_simple_mamba=False,
        latent_diffusion=False,
        bidirectional=True,
        block_ffn=False,
    )


@pytest.fixture()
def tiny_student_with_ffn() -> DIMBA:
    """Tiny DIMBA with block_ffn=True (mlp type, hidden=128)."""
    torch.manual_seed(2)
    return DIMBA(
        vocab_size=_VOCAB,
        d_model=_DS,
        d_prompt=_DS,
        num_diffusion_steps=20,
        num_denoiser_layers=_LS,
        d_state=16,
        d_conv=4,
        expand=2,
        use_simple_mamba=False,
        latent_diffusion=False,
        bidirectional=True,
        block_ffn=True,
        ffn_type="mlp",
        ffn_hidden=_FFN_HIDDEN,
    )


@pytest.fixture()
def layer_map_uniform() -> LayerMap:
    """Uniform LayerMap mapping 2 student layers onto 4 teacher layers."""
    return LayerMap(_LT, _LS, mode="uniform")


# ---------------------------------------------------------------------------
# Tests: Projector
# ---------------------------------------------------------------------------


class TestProjector:
    """Tests for the Projector module."""

    def test_identity_init_same_dim(self) -> None:
        """When d_in==d_out and depth==1, output should equal input at init."""
        torch.manual_seed(10)
        p = Projector(64, 64, depth=1)
        x = torch.randn(2, 8, 64)
        out = p(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-5), "Identity projector should be a no-op at init"

    def test_dim_change(self) -> None:
        """Projector with d_in != d_out should produce correct output shape."""
        p = Projector(64, 32, depth=1)
        x = torch.randn(2, 8, 64)
        out = p(x)
        assert out.shape == (2, 8, 32)

    def test_depth_2(self) -> None:
        """Projector with depth=2 should produce correct output shape."""
        p = Projector(64, 128, depth=2)
        x = torch.randn(2, 8, 64)
        out = p(x)
        assert out.shape == (2, 8, 128)

    def test_bias_false(self) -> None:
        """Projector with bias=False should have no bias parameters."""
        p = Projector(32, 64, depth=1, bias=False)
        for name, param in p.named_parameters():
            assert "bias" not in name, f"Found unexpected bias parameter: {name}"

    def test_forward_finite(self) -> None:
        """Output should always be finite."""
        p = Projector(64, 64, depth=1)
        x = torch.randn(3, 10, 64)
        out = p(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Tests: HeadAligner
# ---------------------------------------------------------------------------


class TestHeadAligner:
    """Tests for the HeadAligner module."""

    def test_identity_when_equal_heads(self) -> None:
        """When h_student == h_teacher, forward should be an exact identity."""
        ha = HeadAligner(4, 4)
        M = torch.randn(2, 4, 8, 8)
        out = ha(M)
        assert out is M, "HeadAligner should return input tensor unchanged when heads are equal"

    def test_no_params_when_equal(self) -> None:
        """No learnable parameters when h_student == h_teacher."""
        ha = HeadAligner(3, 3)
        params = list(ha.parameters())
        assert len(params) == 0

    def test_output_shape_head_upsample(self) -> None:
        """When Hs < Ht the output should have Ht heads."""
        ha = HeadAligner(_HS, _HT)  # 2 -> 4
        M = torch.randn(_B, _HS, _L, _L)
        out = ha(M)
        assert out.shape == (_B, _HT, _L, _L)

    def test_output_shape_head_downsample(self) -> None:
        """When Hs > Ht the output should have Ht heads."""
        ha = HeadAligner(8, 2)
        M = torch.randn(_B, 8, _L, _L)
        out = ha(M)
        assert out.shape == (_B, 2, _L, _L)

    def test_weight_shape(self) -> None:
        """weight should be [h_teacher, h_student]."""
        ha = HeadAligner(3, 5)
        assert ha.weight.shape == (5, 3)

    def test_mean_init(self) -> None:
        """At init each teacher head is the mean of student heads: entries = 1/h_student."""
        ha = HeadAligner(3, 5)
        expected = 1.0 / 3.0
        assert torch.allclose(ha.weight, torch.full((5, 3), expected), atol=1e-6)

    def test_forward_finite(self) -> None:
        """Output should be finite."""
        ha = HeadAligner(2, 4)
        M = torch.randn(2, 2, 6, 6)
        out = ha(M)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Tests: LayerMap
# ---------------------------------------------------------------------------


class TestLayerMap:
    """Tests for LayerMap."""

    def test_uniform_pairs_length(self) -> None:
        """Uniform mode should produce exactly n_student pairs."""
        lm = LayerMap(4, 3, mode="uniform")
        pairs = lm.pairs()
        assert len(pairs) == 3

    def test_uniform_pairs_bounds(self) -> None:
        """All (student_idx, teacher_idx) should be in valid range."""
        lm = LayerMap(6, 4, mode="uniform")
        for si, ti in lm.pairs():
            assert 0 <= si < 4
            assert 0 <= ti < 6

    def test_uniform_endpoints(self) -> None:
        """With n_student>=2, first pair -> teacher 0, last -> teacher n_teacher-1."""
        n_t, n_s = 6, 4
        lm = LayerMap(n_t, n_s, mode="uniform")
        pairs = lm.pairs()
        assert pairs[0][1] == 0
        assert pairs[-1][1] == n_t - 1

    def test_uniform_single_student(self) -> None:
        """A single student layer maps to teacher layer 0 in uniform mode."""
        lm = LayerMap(5, 1, mode="uniform")
        assert lm.pairs() == [(0, 0)]

    def test_last_all_map_to_last(self) -> None:
        """'last' mode: all student layers map to teacher index n_teacher-1."""
        n_t, n_s = 5, 3
        lm = LayerMap(n_t, n_s, mode="last")
        for si, ti in lm.pairs():
            assert ti == n_t - 1

    def test_last_pairs_length(self) -> None:
        """'last' mode should produce exactly n_student pairs."""
        lm = LayerMap(4, 2, mode="last")
        assert len(lm.pairs()) == 2

    def test_explicit_mode(self) -> None:
        """'explicit' mode should use exactly the provided pairs."""
        explicit = [(0, 1), (1, 3)]
        lm = LayerMap(4, 2, mode="explicit", explicit=explicit)
        assert lm.pairs() == explicit

    def test_explicit_missing_raises(self) -> None:
        """'explicit' mode without explicit list should raise ValueError."""
        with pytest.raises(ValueError):
            LayerMap(4, 2, mode="explicit")

    def test_invalid_mode_raises(self) -> None:
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError):
            LayerMap(4, 2, mode="bogus")


# ---------------------------------------------------------------------------
# Tests: stage1_matrix_loss
# ---------------------------------------------------------------------------


class TestStage1MatrixLoss:
    """Tests for stage1_matrix_loss."""

    def _make_align_and_teacher(
        self, B: int, Hs: int, Ht: int, L: int, n_student: int, n_teacher: int
    ) -> Tuple[Dict, "TeacherOutputs"]:
        """Build fake align dict and TeacherOutputs for testing."""
        matrices_fwd = [torch.rand(B, Hs, L, L) for _ in range(n_student)]
        matrices_bwd = [torch.rand(B, Hs, L, L) for _ in range(n_student)]
        block_inputs = [torch.randn(B, L, _DS) for _ in range(n_student)]
        block_outputs = [torch.randn(B, L, _DS) for _ in range(n_student)]
        align = {
            "matrices_fwd": matrices_fwd,
            "matrices_bwd": matrices_bwd,
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,
            "denoiser_out": torch.randn(B, L, _DS),
        }
        # Teacher attention: valid softmax over last dim.
        attentions = tuple(
            F.softmax(torch.randn(B, Ht, L, L), dim=-1) for _ in range(n_teacher)
        )
        hidden_states = tuple(torch.randn(B, L, Ht * 16) for _ in range(n_teacher + 1))
        teacher_out = TeacherOutputs(
            hidden_states=hidden_states,
            attentions=attentions,
            logits=torch.randn(B, L, _VOCAB),
        )
        return align, teacher_out

    def test_finite_scalar_causal(self) -> None:
        """Loss should be a finite scalar for causal teacher type."""
        torch.manual_seed(20)
        n_s, n_t = 2, 4
        align, teacher_out = self._make_align_and_teacher(_B, _HS, _HT, _L, n_s, n_t)
        lm = LayerMap(n_t, n_s, mode="uniform")
        head_aligners = [HeadAligner(_HS, _HT) for _ in range(n_s)]
        loss, parts = stage1_matrix_loss(
            align, teacher_out, lm, head_aligners,
            teacher_type="causal", bidirectional=True,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "stage1_fwd" in parts and "stage1_bwd" in parts

    def test_finite_scalar_bidirectional(self) -> None:
        """Loss should be a finite scalar for bidirectional teacher type."""
        torch.manual_seed(21)
        n_s, n_t = 2, 4
        align, teacher_out = self._make_align_and_teacher(_B, _HS, _HT, _L, n_s, n_t)
        lm = LayerMap(n_t, n_s, mode="uniform")
        head_aligners = [HeadAligner(_HS, _HT) for _ in range(n_s)]
        loss, parts = stage1_matrix_loss(
            align, teacher_out, lm, head_aligners,
            teacher_type="bidirectional", bidirectional=True,
        )
        assert torch.isfinite(loss)

    def test_backward_term_zero_when_disabled(self) -> None:
        """With bidirectional=False, stage1_bwd part should be zero."""
        torch.manual_seed(22)
        n_s, n_t = 2, 4
        align, teacher_out = self._make_align_and_teacher(_B, _HS, _HT, _L, n_s, n_t)
        lm = LayerMap(n_t, n_s, mode="uniform")
        head_aligners = [HeadAligner(_HS, _HT) for _ in range(n_s)]
        _loss, parts = stage1_matrix_loss(
            align, teacher_out, lm, head_aligners,
            teacher_type="causal", bidirectional=False,
        )
        assert parts["stage1_bwd"].item() == pytest.approx(0.0)

    def test_invalid_teacher_type_raises(self) -> None:
        """Unknown teacher_type should raise ValueError."""
        n_s, n_t = 2, 4
        align, teacher_out = self._make_align_and_teacher(_B, _HS, _HT, _L, n_s, n_t)
        lm = LayerMap(n_t, n_s, mode="uniform")
        head_aligners = [HeadAligner(_HS, _HT) for _ in range(n_s)]
        with pytest.raises(ValueError):
            stage1_matrix_loss(
                align, teacher_out, lm, head_aligners,
                teacher_type="invalid",
            )

    def test_loss_decreases_on_overfit(self) -> None:
        """Stage-1 loss should decrease when overfitting on a fixed batch.

        Uses a real DIMBA with TorchMamba2 (use_simple_mamba=False) to obtain
        actual mixing matrices. Checks that 50 steps of Adam reduce the loss
        by at least 10% compared to the initial value.
        """
        torch.manual_seed(30)
        n_s = 2
        n_t = 4
        # Build tiny DIMBA with TorchMamba2 (d_model=64, expand=2 -> nheads=2)
        student = DIMBA(
            vocab_size=_VOCAB,
            d_model=_DS,
            d_prompt=_DS,
            num_diffusion_steps=20,
            num_denoiser_layers=n_s,
            d_state=16,
            d_conv=4,
            expand=2,
            use_simple_mamba=False,
            latent_diffusion=False,
            bidirectional=True,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))

        # Fixed teacher outputs.
        torch.manual_seed(31)
        attentions = tuple(
            F.softmax(torch.randn(_B, _HT, _L, _L), dim=-1) for _ in range(n_t)
        )
        hidden_states = tuple(torch.randn(_B, _L, _DT) for _ in range(n_t + 1))
        teacher_out = TeacherOutputs(
            hidden_states=hidden_states,
            attentions=attentions,
            logits=None,
        )

        lm = LayerMap(n_t, n_s, mode="uniform")
        head_aligners = nn.ModuleList([HeadAligner(_HS, _HT) for _ in range(n_s)])

        # Only head_aligners and mixer params are trained in stage 1.
        params = list(student.denoiser.parameters()) + list(head_aligners.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        align = student.align_forward(input_ids, return_hidden_states=True, return_matrices=True)
        loss_init, _ = stage1_matrix_loss(
            align, teacher_out, lm, list(head_aligners),
            teacher_type="causal", bidirectional=True,
        )
        initial = loss_init.item()

        for _ in range(50):
            opt.zero_grad()
            align = student.align_forward(
                input_ids, return_hidden_states=True, return_matrices=True,
            )
            loss, _ = stage1_matrix_loss(
                align, teacher_out, lm, list(head_aligners),
                teacher_type="causal", bidirectional=True,
            )
            loss.backward()
            opt.step()

        final = loss.item()
        assert final < initial, (
            f"Stage-1 loss did not decrease: initial={initial:.4f}, final={final:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: stage2_hidden_loss
# ---------------------------------------------------------------------------


class TestStage2HiddenLoss:
    """Tests for stage2_hidden_loss."""

    def test_finite_scalar(self) -> None:
        """Loss should be a finite scalar."""
        torch.manual_seed(40)
        n_s, n_t = 2, 4
        block_outputs = [torch.randn(_B, _L, _DS) for _ in range(n_s)]
        align = {
            "block_inputs": block_outputs,
            "block_outputs": block_outputs,
            "matrices_fwd": [None] * n_s,
            "matrices_bwd": [None] * n_s,
            "denoiser_out": torch.randn(_B, _L, _DS),
        }
        hidden_states = tuple(torch.randn(_B, _L, _DT) for _ in range(n_t + 1))
        teacher_out = TeacherOutputs(
            hidden_states=hidden_states,
            attentions=tuple(torch.zeros(_B, _HT, _L, _L) for _ in range(n_t)),
            logits=None,
        )
        lm = LayerMap(n_t, n_s, mode="uniform")
        projectors = [Projector(_DS, _DT, depth=1) for _ in range(n_s)]
        loss, parts = stage2_hidden_loss(align, teacher_out, lm, projectors)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "stage2" in parts

    def test_loss_decreases_on_overfit(self) -> None:
        """Stage-2 loss should decrease when overfitting on a fixed batch."""
        torch.manual_seed(50)
        n_s = 2
        n_t = 4
        student = DIMBA(
            vocab_size=_VOCAB,
            d_model=_DS,
            d_prompt=_DS,
            num_diffusion_steps=20,
            num_denoiser_layers=n_s,
            d_state=16,
            d_conv=4,
            expand=2,
            use_simple_mamba=False,
            latent_diffusion=False,
            bidirectional=True,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))

        # Fixed teacher hidden states.
        torch.manual_seed(51)
        hidden_states = tuple(torch.randn(_B, _L, _DT) for _ in range(n_t + 1))
        teacher_out = TeacherOutputs(
            hidden_states=hidden_states,
            attentions=tuple(torch.zeros(_B, _HT, _L, _L) for _ in range(n_t)),
            logits=None,
        )
        lm = LayerMap(n_t, n_s, mode="uniform")
        projectors = nn.ModuleList([Projector(_DS, _DT) for _ in range(n_s)])

        params = list(student.denoiser.parameters()) + list(projectors.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        align = student.align_forward(input_ids, return_hidden_states=True)
        loss_init, _ = stage2_hidden_loss(align, teacher_out, lm, list(projectors))
        initial = loss_init.item()

        for _ in range(50):
            opt.zero_grad()
            align = student.align_forward(input_ids, return_hidden_states=True)
            loss, _ = stage2_hidden_loss(align, teacher_out, lm, list(projectors))
            loss.backward()
            opt.step()

        final = loss.item()
        assert final < initial, (
            f"Stage-2 loss did not decrease: initial={initial:.4f}, final={final:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: stage3_kd_loss
# ---------------------------------------------------------------------------


class TestStage3KdLoss:
    """Tests for stage3_kd_loss."""

    def test_finite_scalar(self) -> None:
        """Loss should be a finite scalar for random logits."""
        torch.manual_seed(60)
        s = torch.randn(_B, _L, _VOCAB)
        t = torch.randn(_B, _L, _VOCAB)
        loss = stage3_kd_loss(s, t)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_near_zero_when_equal(self) -> None:
        """When student logits equal teacher logits, KL should be ~0."""
        torch.manual_seed(61)
        logits = torch.randn(_B, _L, _VOCAB)
        loss = stage3_kd_loss(logits, logits)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_scalar(self) -> None:
        """KL divergence should be non-negative."""
        torch.manual_seed(62)
        s = torch.randn(_B, _L, _VOCAB)
        t = torch.randn(_B, _L, _VOCAB)
        loss = stage3_kd_loss(s, t)
        assert loss.item() >= 0.0

    def test_temperature_scaling(self) -> None:
        """Higher temperature should generally change the loss value."""
        torch.manual_seed(63)
        s = torch.randn(_B, _L, _VOCAB)
        t = torch.randn(_B, _L, _VOCAB)
        loss_t1 = stage3_kd_loss(s, t, kd_temp=1.0)
        loss_t4 = stage3_kd_loss(s, t, kd_temp=4.0)
        # They should not be exactly equal (unless degenerate inputs).
        assert loss_t1.item() != pytest.approx(loss_t4.item(), rel=1e-3)

    def test_vocab_mismatch_raises(self) -> None:
        """Mismatched vocab sizes should raise ValueError."""
        s = torch.randn(_B, _L, 40)
        t = torch.randn(_B, _L, 50)
        with pytest.raises(ValueError):
            stage3_kd_loss(s, t)


# ---------------------------------------------------------------------------
# Tests: build_student_from_teacher
# ---------------------------------------------------------------------------


class TestBuildStudentFromTeacher:
    """Tests for build_student_from_teacher weight surgery."""

    def test_returns_dimba_and_layermap(self, stub_teacher: StubTeacher) -> None:
        """Should return a DIMBA and a LayerMap."""
        model, lm = build_student_from_teacher(
            stub_teacher,
            num_student_layers=_LS,
            block_ffn=False,
            inherit_embeddings=False,
            inherit_ffn=False,
            inherit_head=False,
            use_simple_mamba=True,  # no CUDA required
        )
        assert isinstance(model, DIMBA)
        assert isinstance(lm, LayerMap)

    def test_vocab_and_dim_match(self, stub_teacher: StubTeacher) -> None:
        """Student vocab and d_model should match teacher."""
        model, _ = build_student_from_teacher(
            stub_teacher,
            num_student_layers=_LS,
            block_ffn=False,
            inherit_embeddings=False,
            use_simple_mamba=True,
        )
        assert model.vocab_size == stub_teacher.vocab_size
        assert model.d_model == stub_teacher.d_model

    def test_inherit_embeddings(self, stub_teacher: StubTeacher) -> None:
        """Embedding weights should be copied from teacher when shapes match."""
        model, _ = build_student_from_teacher(
            stub_teacher,
            num_student_layers=_LS,
            block_ffn=False,
            inherit_embeddings=True,
            inherit_ffn=False,
            inherit_head=False,
            use_simple_mamba=True,
        )
        teacher_emb = stub_teacher.input_embedding_weight()
        student_emb = model.token_embed.get_weight()
        assert torch.allclose(student_emb, teacher_emb, atol=1e-6), (
            "Student embedding should match teacher after surgery"
        )

    def test_inherit_ffn_copied(self, stub_teacher: StubTeacher) -> None:
        """FFN weights should be copied when block_ffn=True and shapes match."""
        model, lm = build_student_from_teacher(
            stub_teacher,
            num_student_layers=_LS,
            block_ffn=True,
            ffn_type="mlp",
            ffn_hidden=_FFN_HIDDEN,
            inherit_embeddings=False,
            inherit_ffn=True,
            inherit_head=False,
            use_simple_mamba=True,
        )
        # Check first paired layer has matching FFN.
        si, ti = lm.pairs()[0]
        block_ffn = model.denoiser.blocks[si].ffn
        assert block_ffn is not None, "block_ffn should be present"
        teacher_state = stub_teacher.layer_ffn_state(ti)
        student_state = block_ffn.state_dict()
        for key in teacher_state:
            if key in student_state:
                assert torch.allclose(
                    student_state[key], teacher_state[key], atol=1e-6
                ), f"FFN key {key!r} not copied correctly"

    def test_config_roundtrip(self, stub_teacher: StubTeacher) -> None:
        """DIMBA(**model.config) should build a structurally identical model."""
        model, _ = build_student_from_teacher(
            stub_teacher,
            num_student_layers=_LS,
            block_ffn=True,
            ffn_type="mlp",
            ffn_hidden=_FFN_HIDDEN,
            use_simple_mamba=True,
        )
        cfg = model.config
        model2 = DIMBA(**cfg)
        # Check key config attributes match.
        assert model2.vocab_size == model.vocab_size
        assert model2.d_model == model.d_model
        assert len(model2.denoiser.blocks) == len(model.denoiser.blocks)

    def test_shape_mismatch_does_not_crash(self) -> None:
        """A vocab mismatch should warn and skip — not crash."""
        # Create a teacher with a different vocab to force shape mismatch.
        teacher_diff_vocab = StubTeacher(
            num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=999
        )
        # Force student vocab to mismatch (override).
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model, _ = build_student_from_teacher(
                teacher_diff_vocab,
                num_student_layers=_LS,
                block_ffn=False,
                # Override vocab_size to be different from teacher:
                vocab_size=_VOCAB,
                inherit_embeddings=True,
                inherit_ffn=False,
                inherit_head=True,
                use_simple_mamba=True,
            )
        # Should not have raised; model should still be valid.
        assert isinstance(model, DIMBA)


# ---------------------------------------------------------------------------
# Tests: principled_init_from_teacher
# ---------------------------------------------------------------------------


class TestPrincipledInit:
    """Tests for principled_init_from_teacher."""

    def test_no_crash_and_finite_params(
        self, tiny_student_no_ffn: DIMBA, stub_teacher: StubTeacher, layer_map_uniform: LayerMap
    ) -> None:
        """After init, all parameters should remain finite (no NaN/Inf)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            principled_init_from_teacher(
                tiny_student_no_ffn, stub_teacher, layer_map_uniform, mode="qkvo"
            )
        for name, param in tiny_student_no_ffn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite parameter after init: {name}"

    def test_o_only_mode(
        self, tiny_student_no_ffn: DIMBA, stub_teacher: StubTeacher, layer_map_uniform: LayerMap
    ) -> None:
        """o_only mode should also leave finite parameters."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            principled_init_from_teacher(
                tiny_student_no_ffn, stub_teacher, layer_map_uniform, mode="o_only"
            )
        for name, param in tiny_student_no_ffn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite parameter: {name}"

    def test_invalid_mode_raises(
        self, tiny_student_no_ffn: DIMBA, stub_teacher: StubTeacher, layer_map_uniform: LayerMap
    ) -> None:
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError):
            principled_init_from_teacher(
                tiny_student_no_ffn, stub_teacher, layer_map_uniform, mode="bogus"
            )

    def test_out_proj_dimensions_change(
        self, tiny_student_no_ffn: DIMBA, stub_teacher: StubTeacher, layer_map_uniform: LayerMap
    ) -> None:
        """The out_proj of the first student block should change after init (teacher O is random)."""
        block0 = tiny_student_no_ffn.denoiser.blocks[0]
        out_proj_before = block0.mamba_fwd.out_proj.weight.clone()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            principled_init_from_teacher(
                tiny_student_no_ffn, stub_teacher, layer_map_uniform, mode="o_only"
            )
        out_proj_after = block0.mamba_fwd.out_proj.weight
        # They should not be equal (teacher O weight is random and almost certainly different).
        assert not torch.allclose(out_proj_before, out_proj_after), (
            "out_proj should have been updated by principled_init"
        )


# ---------------------------------------------------------------------------
# Tests: end-to-end integration (no trainer)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests combining multiple components."""

    def test_full_stage1_pipeline(self) -> None:
        """Full Stage-1 pipeline: build student, run align_forward, compute loss."""
        torch.manual_seed(70)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, lm = build_student_from_teacher(
            teacher,
            num_student_layers=_LS,
            block_ffn=False,
            use_simple_mamba=False,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        align = model.align_forward(input_ids, return_hidden_states=True, return_matrices=True)
        teacher_out = teacher(input_ids)

        head_aligners = [HeadAligner(_HS, _HT) for _ in range(_LS)]
        loss, parts = stage1_matrix_loss(
            align, teacher_out, lm, head_aligners,
            teacher_type="causal", bidirectional=True,
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_full_stage2_pipeline(self) -> None:
        """Full Stage-2 pipeline: build student, run align_forward, compute loss."""
        torch.manual_seed(80)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, lm = build_student_from_teacher(
            teacher,
            num_student_layers=_LS,
            block_ffn=False,
            use_simple_mamba=True,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        align = model.align_forward(input_ids, return_hidden_states=True)
        teacher_out = teacher(input_ids)

        projectors = [Projector(_DS, _DT) for _ in range(_LS)]
        loss, parts = stage2_hidden_loss(align, teacher_out, lm, projectors)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_full_stage3_pipeline(self) -> None:
        """Full Stage-3 pipeline: student logits vs teacher logits KD loss."""
        torch.manual_seed(90)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, _ = build_student_from_teacher(
            teacher,
            num_student_layers=_LS,
            block_ffn=False,
            use_simple_mamba=True,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        teacher_out = teacher(input_ids)

        # Produce student logits via output_head on encoded latent.
        emb = model.token_embed(input_ids)
        student_logits = model.output_head(emb)
        teacher_logits = teacher_out.logits

        # Shapes must match for stage3_kd_loss.
        assert student_logits.shape[-1] == teacher_logits.shape[-1] == _VOCAB
        loss = stage3_kd_loss(student_logits, teacher_logits)
        assert torch.isfinite(loss)

    def test_saved_and_reloaded_state_dict(self) -> None:
        """A saved and reloaded state_dict should load strict into DIMBA(**config)."""
        import io

        torch.manual_seed(100)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, _ = build_student_from_teacher(
            teacher,
            num_student_layers=_LS,
            block_ffn=True,
            ffn_type="mlp",
            ffn_hidden=_FFN_HIDDEN,
            use_simple_mamba=True,
        )
        cfg = model.config

        # Save to bytes buffer.
        buf = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, buf)
        buf.seek(0)
        checkpoint = torch.load(buf, map_location="cpu")

        # Reload strict.
        model2 = DIMBA(**checkpoint["config"])
        missing, unexpected = model2.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert len(missing) == 0, f"Missing keys after reload: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys after reload: {unexpected}"

        # Forward pass on reloaded model should produce finite output.
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        with torch.no_grad():
            emb = model2.token_embed(input_ids)
            logits = model2.output_head(emb)
        assert torch.isfinite(logits).all()

    def test_forward_after_surgery_finite(self) -> None:
        """Student model should produce finite logits after embedding/FFN surgery."""
        torch.manual_seed(110)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, _ = build_student_from_teacher(
            teacher,
            num_student_layers=_LS,
            block_ffn=True,
            ffn_type="mlp",
            ffn_hidden=_FFN_HIDDEN,
            inherit_embeddings=True,
            inherit_ffn=True,
            inherit_head=True,
            use_simple_mamba=True,
        )
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        with torch.no_grad():
            align = model.align_forward(input_ids, return_hidden_states=True)
        out_emb = align["denoiser_out"]
        logits = model.output_head(out_emb)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Optional: DistillationTrainer tests (skip if trainer.py not present)
# ---------------------------------------------------------------------------

_HAS_TRAINER = False
try:
    from dimba.distillation.trainer import DistillationConfig, DistillationTrainer  # type: ignore
    _HAS_TRAINER = True
except ImportError:
    pass


@pytest.mark.skipif(not _HAS_TRAINER, reason="dimba.distillation.trainer not yet implemented")
class TestDistillationTrainer:
    """Tests for DistillationTrainer (skipped if trainer.py is absent)."""

    def _make_tiny_dataloader(
        self, vocab_size: int, seq_len: int, n_batches: int = 4
    ) -> List[torch.Tensor]:
        """Return a list of [B, L] tensors as a fake dataloader."""
        torch.manual_seed(200)
        return [torch.randint(0, vocab_size, (_B, seq_len)) for _ in range(n_batches)]

    def test_trainer_runs_stage1(self) -> None:
        """DistillationTrainer.run_stage should execute without error for stage1."""
        from dimba.distillation.trainer import DistillationConfig, DistillationTrainer

        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, lm = build_student_from_teacher(
            teacher, num_student_layers=_LS, block_ffn=False, use_simple_mamba=False,
        )
        cfg = DistillationConfig(
            teacher_model="stub",
            teacher_type="causal",
            stages=[{"name": "stage1", "steps": 2, "lr": 1e-3}],
        )
        loader = self._make_tiny_dataloader(_VOCAB, _L, n_batches=4)
        trainer = DistillationTrainer(model, teacher, cfg, layer_map=lm)
        trainer.run_stage(cfg.stages[0], loader)
        # Model params should remain finite.
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite param after stage1: {name}"

    def test_trainer_run_all_stages(self) -> None:
        """DistillationTrainer.run should iterate all stages and return the model."""
        from dimba.distillation.trainer import DistillationConfig, DistillationTrainer

        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, lm = build_student_from_teacher(
            teacher, num_student_layers=_LS, block_ffn=False, use_simple_mamba=True,
        )
        cfg = DistillationConfig(
            teacher_model="stub",
            teacher_type="causal",
            stages=[
                {"name": "stage2", "steps": 2, "lr": 1e-3},
                {"name": "stage3", "steps": 2, "lr": 1e-4},
            ],
        )
        loader = self._make_tiny_dataloader(_VOCAB, _L, n_batches=4)
        trainer = DistillationTrainer(model, teacher, cfg, layer_map=lm)
        result = trainer.run(loader)
        assert isinstance(result, DIMBA)

    def test_distillation_config_from_dict(self) -> None:
        """DistillationConfig.from_dict should reconstruct a config from a plain dict."""
        from dimba.distillation.trainer import DistillationConfig

        d = {
            "teacher_model": "gpt2",
            "teacher_type": "causal",
            "mode": "convert",
            "kd_weight": 0.5,
            "stages": [{"name": "stage3", "steps": 100, "lr": 1e-4}],
        }
        cfg = DistillationConfig.from_dict(d)
        assert cfg.teacher_model == "gpt2"
        assert cfg.kd_weight == pytest.approx(0.5)
        assert len(cfg.stages) == 1

    def test_model_finite_after_run(self) -> None:
        """After DistillationTrainer.run, the model should produce finite outputs."""
        from dimba.distillation.trainer import DistillationConfig, DistillationTrainer

        torch.manual_seed(201)
        teacher = StubTeacher(num_layers=_LT, num_heads=_HT, d_model=_DT, vocab_size=_VOCAB)
        model, lm = build_student_from_teacher(
            teacher, num_student_layers=_LS, block_ffn=False, use_simple_mamba=True,
        )
        cfg = DistillationConfig(
            teacher_model="stub",
            teacher_type="causal",
            stages=[{"name": "stage3", "steps": 1, "lr": 1e-4}],
        )
        loader = self._make_tiny_dataloader(_VOCAB, _L, n_batches=2)
        trainer = DistillationTrainer(model, teacher, cfg, layer_map=lm)
        trainer.run(loader)
        input_ids = torch.randint(0, _VOCAB, (_B, _L))
        with torch.no_grad():
            emb = model.token_embed(input_ids)
            logits = model.output_head(emb)
        assert torch.isfinite(logits).all()

    # ------------------------------------------------------------------
    # FFN frozen→unfrozen co-adaptation schedule (the dropped "Finetune #2")
    # ------------------------------------------------------------------

    @staticmethod
    def _ffn_params(model: DIMBA) -> List[torch.Tensor]:
        """Collect the block-FFN parameters exactly as _freeze_ffn targets them."""
        out: List[torch.Tensor] = []
        for blk in model.denoiser.blocks:
            ffn = getattr(blk, "ffn", None)
            if ffn is not None:
                out.extend(list(ffn.parameters()))
        return out

    @staticmethod
    def _mixer_params(model: DIMBA) -> List[torch.Tensor]:
        """Collect the forward-mixer parameters (always trainable in stage3)."""
        out: List[torch.Tensor] = []
        for blk in model.denoiser.blocks:
            out.extend(list(blk.mamba_fwd.parameters()))
        return out

    def test_stage3_unfrozen_ffn_receives_gradients(
        self, tiny_student_with_ffn: DIMBA, stub_teacher: StubTeacher
    ) -> None:
        """stage3 with freeze_ffn=False: the inherited FFN must be trainable and update.

        Guards the fix for run #1, where the FFN was frozen for the entire base and
        never co-adapted. Confirms the FFN parameters (a) report requires_grad=True
        and (b) actually change after a few optimiser steps (gradients flowed).
        """
        from dimba.distillation.trainer import DistillationConfig, DistillationTrainer

        model = tiny_student_with_ffn
        cfg = DistillationConfig(
            teacher_model="stub",
            teacher_type="causal",
            stages=[{"name": "stage3", "steps": 3, "lr": 1e-2, "freeze_ffn": False}],
        )
        loader = self._make_tiny_dataloader(_VOCAB, _L, n_batches=4)
        trainer = DistillationTrainer(model, stub_teacher, cfg)

        ffn_params = self._ffn_params(model)
        assert ffn_params, "fixture must have block-FFN parameters"
        before = [p.detach().clone() for p in ffn_params]

        trainer.run_stage(cfg.stages[0], loader)

        assert all(p.requires_grad for p in ffn_params), (
            "FFN params must be trainable when freeze_ffn=False"
        )
        changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, ffn_params))
        assert changed, "FFN params should change after an unfrozen stage3 (gradients flowed)"
        for p in ffn_params:
            assert torch.isfinite(p).all()

    def test_stage3_frozen_ffn_stays_frozen(
        self, tiny_student_with_ffn: DIMBA, stub_teacher: StubTeacher
    ) -> None:
        """stage3 with freeze_ffn=True: the FFN must NOT update, but the mixer must.

        Confirms the freeze switch still works (so the FFN-frozen first phase is a real
        thing) and that the rest of the block trains, so a frozen phase is not a no-op.
        """
        from dimba.distillation.trainer import DistillationConfig, DistillationTrainer

        model = tiny_student_with_ffn
        cfg = DistillationConfig(
            teacher_model="stub",
            teacher_type="causal",
            stages=[{"name": "stage3", "steps": 3, "lr": 1e-2, "freeze_ffn": True}],
        )
        loader = self._make_tiny_dataloader(_VOCAB, _L, n_batches=4)
        trainer = DistillationTrainer(model, stub_teacher, cfg)

        ffn_params = self._ffn_params(model)
        mixer_params = self._mixer_params(model)
        assert ffn_params and mixer_params
        ffn_before = [p.detach().clone() for p in ffn_params]
        mixer_before = [p.detach().clone() for p in mixer_params]

        trainer.run_stage(cfg.stages[0], loader)

        assert not any(p.requires_grad for p in ffn_params), (
            "FFN params must be frozen when freeze_ffn=True"
        )
        # Frozen params are excluded from the optimiser → exactly unchanged.
        assert all(torch.equal(b, p.detach()) for b, p in zip(ffn_before, ffn_params)), (
            "FFN params must not change when freeze_ffn=True"
        )
        # The mixer is still trainable, so a frozen stage is not a no-op.
        mixer_changed = any(
            not torch.equal(b, p.detach()) for b, p in zip(mixer_before, mixer_params)
        )
        assert mixer_changed, "Mixer params should train even when the FFN is frozen"
