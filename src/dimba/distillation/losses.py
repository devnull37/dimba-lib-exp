"""Pure-function losses for cross-architecture knowledge distillation in DIMBA.

Three distillation stages are implemented as standalone functions:

* :func:`stage1_matrix_loss` — align student Mamba-2 mixing matrices to teacher attention maps.
* :func:`stage2_hidden_loss` — align student block hidden states to teacher layer outputs.
* :func:`stage3_kd_loss` — soft-label KL-divergence over the vocabulary distribution.

All functions are stateless; they operate directly on the tensors contained in the
``align`` dict returned by ``DIMBA.align_forward`` and the ``TeacherOutputs`` dataclass.
Import only torch and typing so this module stays dependency-free.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Stage 1 — mixing-matrix / attention alignment
# ---------------------------------------------------------------------------


def stage1_matrix_loss(
    align: Dict,
    teacher_out,
    layer_map,
    head_aligners: List,
    *,
    teacher_type: str,
    bidirectional: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Align student Mamba-2 mixing matrices to teacher attention maps.

    For each ``(student_idx, teacher_idx)`` pair in ``layer_map.pairs()``:

    * The student forward mixing matrix ``Ms_fwd`` (lower-triangular, shape
      ``[B, Hs, L, L]``) is projected to teacher head count via
      ``head_aligners[student_idx]``.
    * For a causal teacher the target is the full attention matrix ``A``; for a
      bidirectional teacher the target is ``tril(A)`` (forward) and ``triu(A)``
      (backward).
    * If ``bidirectional=True`` and the student backward matrix is not ``None``,
      the backward matrix is similarly aligned and compared against the transposed
      (or upper-triangular) attention.

    All per-pair losses use mean squared error. The returned scalar is the mean
    across all pairs.

    Args:
        align: Dict returned by ``DIMBA.align_forward(..., return_matrices=True)``.
            Must contain ``'matrices_fwd'`` (list of ``[B, Hs, L, L]`` tensors or
            ``None`` entries) and ``'matrices_bwd'`` (same).
        teacher_out: ``TeacherOutputs`` instance. ``teacher_out.attentions`` is a
            tuple of length ``n_teacher_layers``, each ``[B, Ht, L, L]``.
        layer_map: Object with a ``pairs()`` method returning
            ``List[Tuple[int, int]]`` of ``(student_idx, teacher_idx)``.
        head_aligners: One :class:`HeadAligner` per *student* block. Indexed by
            ``student_idx``; maps ``[B, Hs, L, L] -> [B, Ht, L, L]``.
        teacher_type: ``'causal'`` or ``'bidirectional'``. Controls which part of
            the teacher attention matrix is used as the target.
        bidirectional: Whether to include a backward-scan loss term when the
            student backward matrix is available.

    Returns:
        A ``(loss, parts)`` tuple where ``loss`` is a scalar tensor and ``parts``
        is a dict with keys ``'stage1_fwd'`` and ``'stage1_bwd'`` (each a scalar
        tensor; ``'stage1_bwd'`` is zero when not computed).

    Raises:
        ValueError: If ``teacher_type`` is not ``'causal'`` or ``'bidirectional'``.
    """
    if teacher_type not in ("causal", "bidirectional"):
        raise ValueError(
            f"teacher_type must be 'causal' or 'bidirectional', got '{teacher_type}'"
        )

    matrices_fwd: List[Optional[Tensor]] = align["matrices_fwd"]
    matrices_bwd: List[Optional[Tensor]] = align.get("matrices_bwd", [])

    pairs = layer_map.pairs()
    if not pairs:
        device = matrices_fwd[0].device if matrices_fwd else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        return zero, {"stage1_fwd": zero.clone(), "stage1_bwd": zero.clone()}

    # Use device of first available matrix.
    device = None
    for m in matrices_fwd:
        if m is not None:
            device = m.device
            break
    if device is None:
        device = torch.device("cpu")

    loss_fwd_acc = torch.tensor(0.0, device=device)
    loss_bwd_acc = torch.tensor(0.0, device=device)
    fwd_count = 0
    bwd_count = 0

    for si, ti in pairs:
        # ---- forward mixing matrix ----
        ms_fwd: Optional[Tensor] = matrices_fwd[si] if si < len(matrices_fwd) else None
        attn: Tensor = teacher_out.attentions[ti].float()  # [B, Ht, L, L]  # teacher fp32; explicit is safe

        if ms_fwd is not None:
            # Cast student matrix to fp32 before the HeadAligner einsum so that
            # the aligner weight (fp32, kept out of bf16 student dtype cast) and
            # the input matrix are the same dtype.  mse_loss is then fp32 too.
            aligned_fwd: Tensor = head_aligners[si](ms_fwd.float())  # [B, Ht, L, L]
            if teacher_type == "causal":
                target_fwd = attn
            else:
                target_fwd = torch.tril(attn)
            loss_fwd_acc = loss_fwd_acc + F.mse_loss(aligned_fwd, target_fwd)
            fwd_count += 1

        # ---- backward mixing matrix (optional) ----
        if bidirectional:
            ms_bwd: Optional[Tensor] = (
                matrices_bwd[si] if matrices_bwd and si < len(matrices_bwd) else None
            )
            if ms_bwd is not None:
                aligned_bwd: Tensor = head_aligners[si](ms_bwd.float())  # [B, Ht, L, L]
                if teacher_type == "causal":
                    target_bwd = attn.transpose(-1, -2)
                else:
                    target_bwd = torch.triu(attn)
                loss_bwd_acc = loss_bwd_acc + F.mse_loss(aligned_bwd, target_bwd)
                bwd_count += 1

    # Average over pairs (avoid division by zero).
    if fwd_count > 0:
        loss_fwd_avg = loss_fwd_acc / fwd_count
    else:
        loss_fwd_avg = loss_fwd_acc

    if bwd_count > 0:
        loss_bwd_avg = loss_bwd_acc / bwd_count
    else:
        loss_bwd_avg = loss_bwd_acc

    total = loss_fwd_avg + loss_bwd_avg
    parts: Dict[str, Tensor] = {
        "stage1_fwd": loss_fwd_avg,
        "stage1_bwd": loss_bwd_avg,
    }
    return total, parts


# ---------------------------------------------------------------------------
# Stage 2 — hidden-state alignment
# ---------------------------------------------------------------------------


def stage2_hidden_loss(
    align: Dict,
    teacher_out,
    layer_map,
    projectors: List,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Align student block output hidden states to teacher layer outputs.

    For each ``(student_idx, teacher_idx)`` in ``layer_map.pairs()``:

    * Retrieves the student block output ``align['block_outputs'][student_idx]``
      with shape ``[B, L, d_student]``.
    * Projects it to teacher dimension via ``projectors[student_idx]``,
      yielding ``[B, L, d_teacher]``.
    * Compares against ``teacher_out.hidden_states[teacher_idx + 1]`` (the output
      of teacher layer ``teacher_idx``; index 0 is the token embedding).

    Loss per pair is mean-squared error averaged over all elements. The total is
    averaged across pairs.

    Args:
        align: Dict from ``DIMBA.align_forward(..., return_hidden_states=True)``.
            Must contain ``'block_outputs'``: list of ``[B, L, d_student]`` tensors.
        teacher_out: ``TeacherOutputs`` instance. ``teacher_out.hidden_states`` is
            a tuple of length ``n_teacher_layers + 1``, each ``[B, L, d_teacher]``.
        layer_map: Object with ``pairs()`` returning
            ``List[Tuple[int, int]]`` of ``(student_idx, teacher_idx)``.
        projectors: One :class:`Projector` per *student* block. Indexed by
            ``student_idx``; maps ``[B, L, d_student] -> [B, L, d_teacher]``.

    Returns:
        A ``(loss, parts)`` tuple where ``loss`` is a scalar tensor and ``parts``
        is a dict with key ``'stage2'``.
    """
    block_outputs: List[Tensor] = align["block_outputs"]
    pairs = layer_map.pairs()

    if not pairs:
        device = block_outputs[0].device if block_outputs else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        return zero, {"stage2": zero.clone()}

    device = block_outputs[0].device if block_outputs else torch.device("cpu")
    loss_acc = torch.tensor(0.0, device=device)

    for si, ti in pairs:
        so: Tensor = block_outputs[si]  # [B, L, d_student]
        proj: Tensor = projectors[si](so)  # [B, L, d_teacher]
        # teacher hidden_states[0] = token embeddings; [ti+1] = output of layer ti.
        target: Tensor = teacher_out.hidden_states[ti + 1]  # [B, L, d_teacher]
        loss_acc = loss_acc + F.mse_loss(proj.float(), target.float())

    loss_avg = loss_acc / len(pairs)
    parts: Dict[str, Tensor] = {"stage2": loss_avg}
    return loss_avg, parts


# ---------------------------------------------------------------------------
# Stage 3 — soft-label KL-divergence
# ---------------------------------------------------------------------------


def stage3_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    kd_temp: float = 2.0,
) -> Tensor:
    """KL-divergence knowledge distillation loss (soft labels).

    Computes:

    .. math::
        \\mathcal{L}_{KD} = T^2 \\cdot
            \\mathrm{KL}\\!\\left(
                \\mathrm{softmax}\\!\\left(\\frac{z_t}{T}\\right) \\,\\Big\\|\\,
                \\log\\mathrm{softmax}\\!\\left(\\frac{z_s}{T}\\right)
            \\right)

    averaged over batch and sequence positions.

    The ``T^2`` scaling re-weights gradients to be independent of temperature,
    following Hinton et al. (2015).

    Both ``student_logits`` and ``teacher_logits`` must have the same shape
    ``[B, L, V]`` (identical vocabulary size ``V``).

    Args:
        student_logits: Raw (un-normalized) student logits, shape ``[B, L, Vs]``.
        teacher_logits: Raw (un-normalized) teacher logits, shape ``[B, L, Vt]``.
            Must satisfy ``Vs == Vt``.
        kd_temp: Distillation temperature ``T > 0`` (default 2.0).

    Returns:
        A scalar tensor with the mean KL loss scaled by ``T^2``.

    Raises:
        ValueError: If the vocabulary sizes of student and teacher differ.
    """
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"Vocabulary size mismatch: student has {student_logits.shape[-1]} "
            f"but teacher has {teacher_logits.shape[-1]}. "
            "Both must share the same vocabulary for stage-3 KD."
        )

    T = float(kd_temp)
    # Scale logits by temperature.
    s_scaled = student_logits.float() / T  # [B, L, V]
    t_scaled = teacher_logits.float() / T  # [B, L, V]

    # Teacher provides the soft target distribution (detach so no gradient flows back).
    soft_targets = F.softmax(t_scaled, dim=-1).detach()  # [B, L, V]
    log_student = F.log_softmax(s_scaled, dim=-1)  # [B, L, V]

    # KL(teacher || student) = sum_v teacher_v * (log teacher_v - log student_v)
    # F.kl_div expects (log_input, target) where it computes sum(target*(log target - log input)).
    # reduction='batchmean' divides by B; we need mean over B*L, so use 'sum' + manual divide.
    B, L, _V = student_logits.shape
    denom = max(1, B * L)  # avoid 0/0 -> nan on an empty/degenerate batch
    kl = F.kl_div(log_student, soft_targets, reduction="sum") / denom

    return kl * (T * T)
