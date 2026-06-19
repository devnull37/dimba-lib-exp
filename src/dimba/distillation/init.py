"""Principled parameter initialization from a teacher Transformer into a DIMBA student.

This module provides :func:`principled_init_from_teacher`, a best-effort, shape-guarded
routine that seeds a DIMBA model's mixer projections with weights derived from a pretrained
HuggingFace teacher's attention matrices. The approach follows the MOHAWK philosophy:
the teacher's output projection (O) initialises the student mixer's ``out_proj``, and the
Q/K/V projections are used to approximately nudge the student's ``in_proj``.

All weight copies are wrapped in shape checks. Any incompatibility produces a warning and
is silently skipped; the function never raises on a shape mismatch.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _try_copy(dst: nn.Parameter, src: torch.Tensor, name: str) -> bool:
    """Copy ``src`` into ``dst`` in-place, guarded by a shape check.

    Args:
        dst: Destination parameter.
        src: Source tensor to copy from.
        name: Human-readable label used in warning messages.

    Returns:
        True if the copy succeeded, False if shapes were incompatible.
    """
    if dst.shape != src.shape:
        warnings.warn(
            f"principled_init: shape mismatch for {name}: "
            f"dst {tuple(dst.shape)} vs src {tuple(src.shape)} — skipping.",
            UserWarning,
            stacklevel=3,
        )
        return False
    with torch.no_grad():
        dst.data.copy_(src.to(dst.dtype))
    return True


def _get_nheads(block: nn.Module) -> Optional[int]:
    """Derive the number of SSM heads from a Mamba2Block's forward mixer.

    Tries the ``nheads`` attribute exposed by :class:`~dimba.models.torch_mamba2.TorchMamba2`
    and compatible CUDA Mamba-2 instances.  Falls back to ``None`` when absent.

    Args:
        block: A :class:`~dimba.models.denoiser.Mamba2Block` instance.

    Returns:
        Integer number of heads, or ``None`` if not determinable.
    """
    mixer = getattr(block, "mamba_fwd", None)
    if mixer is None:
        return None
    return getattr(mixer, "nheads", None)


def principled_init_from_teacher(
    model: "dimba.models.diffusion.DIMBA",  # type: ignore[name-defined]  # noqa: F821
    teacher: "dimba.distillation.teacher.TeacherWrapper",  # type: ignore[name-defined]  # noqa: F821
    layer_map: "dimba.distillation.projectors.LayerMap",  # type: ignore[name-defined]  # noqa: F821
    *,
    mode: str = "qkvo",
) -> None:
    """Seed a DIMBA student's mixer projections from the teacher's attention weights.

    This is a best-effort, experimental initialisation.  For each ``(student_idx,
    teacher_idx)`` pair produced by ``layer_map``:

    1. **out_proj from O** — The teacher layer's output projection weight
       ``O [d_t, d_t]`` is copied into the student mixer's ``out_proj.weight
       [d_s, d_inner]``.  Both ``out_proj`` attributes (forward and, when the block
       is bidirectional, backward) are initialised.

    2. **in_proj nudge from Q/K/V** (``mode="qkvo"`` only) — The stacked Q/K/V
       weight ``[3*d_t, d_t]`` is used to approximate the first ``3*d_t`` rows of
       the student's ``in_proj.weight``.  This is intentionally approximate because
       the Mamba-2 ``in_proj`` layout
       ``[z | xBC | dt]  =  [d_inner | conv_dim | nheads]`` does not cleanly factor
       into Q/K/V; we overwrite only as many rows as are available and warn that this
       is an approximation.

    Every copy is guarded by a shape check.  On any incompatibility the copy is
    skipped with a :mod:`warnings` warning and execution continues.

    Args:
        model: DIMBA student to mutate in-place.
        teacher: :class:`~dimba.distillation.teacher.TeacherWrapper` providing
            ``layer_attention_qkvo(idx)``.
        layer_map: :class:`~dimba.distillation.projectors.LayerMap` whose
            ``pairs()`` determines which student/teacher layers correspond.
        mode: Initialisation mode.  ``"qkvo"`` (default) copies O and nudges
            in_proj with Q/K/V.  ``"o_only"`` copies only O and skips in_proj.

    Returns:
        None.  Modifies ``model`` in-place.

    Raises:
        ValueError: If ``mode`` is not one of ``"qkvo"`` or ``"o_only"``.

    Note:
        The in_proj nudge is documented as *approximate*.  Do not rely on it for
        correctness — it is a warm-start heuristic that may or may not help
        convergence depending on the architecture match.
    """
    if mode not in ("qkvo", "o_only"):
        raise ValueError(f"principled_init_from_teacher: unknown mode {mode!r}; "
                         f"expected 'qkvo' or 'o_only'.")

    blocks = model.denoiser.blocks  # ModuleList[Mamba2Block]
    pairs = layer_map.pairs()

    for student_idx, teacher_idx in pairs:
        if student_idx >= len(blocks):
            warnings.warn(
                f"principled_init: student_idx {student_idx} out of range "
                f"(model has {len(blocks)} blocks) — skipping pair.",
                UserWarning,
                stacklevel=2,
            )
            continue

        block = blocks[student_idx]

        # ── Retrieve Q/K/V/O from teacher (best-effort) ──────────────────────
        try:
            qkvo = teacher.layer_attention_qkvo(teacher_idx)
        except Exception as exc:
            warnings.warn(
                f"principled_init: could not retrieve qkvo for teacher layer "
                f"{teacher_idx}: {exc} — skipping pair ({student_idx}, {teacher_idx}).",
                UserWarning,
                stacklevel=2,
            )
            continue

        if qkvo is None:
            warnings.warn(
                f"principled_init: teacher returned None for layer_attention_qkvo("
                f"{teacher_idx}) — skipping pair ({student_idx}, {teacher_idx}).",
                UserWarning,
                stacklevel=2,
            )
            continue

        o_weight: Optional[torch.Tensor] = qkvo.get("o")
        q_weight: Optional[torch.Tensor] = qkvo.get("q")
        k_weight: Optional[torch.Tensor] = qkvo.get("k")
        v_weight: Optional[torch.Tensor] = qkvo.get("v")

        # ── 1. Copy O -> out_proj.weight ─────────────────────────────────────
        # out_proj.weight shape: [d_model_s, d_inner_s]  (nn.Linear convention)
        # o_weight shape expected from teacher: [d_t, d_t]
        # We handle the case where d_model_s == d_t (typical after build_student_from_teacher).
        _init_out_proj(block, o_weight, student_idx, teacher_idx, direction="fwd")

        if getattr(block, "bidirectional", False) and block.mamba_bwd is not None:
            _init_out_proj(block, o_weight, student_idx, teacher_idx, direction="bwd")

        # ── 2. Nudge in_proj.weight with Q/K/V (approximate) ─────────────────
        if mode == "qkvo":
            _nudge_in_proj(block, q_weight, k_weight, v_weight, student_idx, teacher_idx)

    logger.debug(
        "principled_init_from_teacher: completed %d layer pair(s) with mode=%r.",
        len(pairs),
        mode,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _init_out_proj(
    block: nn.Module,
    o_weight: Optional[torch.Tensor],
    student_idx: int,
    teacher_idx: int,
    direction: str,
) -> None:
    """Copy the teacher O weight into a student mixer's out_proj.

    ``out_proj.weight`` has shape ``[d_model, d_inner]`` (nn.Linear convention, so
    weight is ``[out, in]``).  The teacher's O weight is ``[d_t, d_t]`` where
    ``d_t == d_model_student`` when the student was built with the same embedding
    dimension.  If ``d_inner > d_t`` we tile/truncate along the input (inner)
    dimension; if dimensions differ completely we warn and skip.

    Args:
        block: :class:`~dimba.models.denoiser.Mamba2Block` to update.
        o_weight: Teacher O projection weight ``[d_t, d_t]``, or None.
        student_idx: Student block index (for warnings).
        teacher_idx: Teacher layer index (for warnings).
        direction: ``"fwd"`` or ``"bwd"`` — which mixer to update.
    """
    if o_weight is None:
        warnings.warn(
            f"principled_init: O weight is None for teacher layer {teacher_idx} "
            f"(student {student_idx}, {direction}) — skipping out_proj init.",
            UserWarning,
            stacklevel=4,
        )
        return

    mixer = block.mamba_fwd if direction == "fwd" else block.mamba_bwd
    if mixer is None:
        return

    out_proj: Optional[nn.Linear] = getattr(mixer, "out_proj", None)
    if out_proj is None or not hasattr(out_proj, "weight"):
        warnings.warn(
            f"principled_init: mixer ({direction}) at student block {student_idx} "
            f"has no out_proj.weight — skipping.",
            UserWarning,
            stacklevel=4,
        )
        return

    dst_w = out_proj.weight  # [d_model_s, d_inner_s]
    d_model_s, d_inner_s = dst_w.shape
    d_t_out, d_t_in = o_weight.shape  # [d_t, d_t]

    if d_model_s != d_t_out:
        warnings.warn(
            f"principled_init: out_proj output dim mismatch at block ({student_idx}, "
            f"{teacher_idx}, {direction}): student d_model={d_model_s}, teacher d_out="
            f"{d_t_out} — skipping.",
            UserWarning,
            stacklevel=4,
        )
        return

    # Build a source tensor of shape [d_model_s, d_inner_s] from o_weight [d_t, d_t].
    # d_inner_s is typically expand*d_model_s; we tile the teacher weight along dim-1.
    if d_inner_s == d_t_in:
        src = o_weight
    elif d_inner_s > d_t_in:
        # Tile teacher weight across the expanded inner dimension.
        repeats = (d_inner_s + d_t_in - 1) // d_t_in  # ceil
        src = o_weight.repeat(1, repeats)[:, :d_inner_s]
        logger.debug(
            "principled_init: tiled out_proj source (%d -> %d) for block (%d, %d, %s).",
            d_t_in,
            d_inner_s,
            student_idx,
            teacher_idx,
            direction,
        )
    else:
        # d_inner_s < d_t_in: truncate
        src = o_weight[:, :d_inner_s]
        logger.debug(
            "principled_init: truncated out_proj source (%d -> %d) for block (%d, %d, %s).",
            d_t_in,
            d_inner_s,
            student_idx,
            teacher_idx,
            direction,
        )

    _try_copy(
        dst_w,
        src,
        name=f"out_proj.weight[block={student_idx}, teacher={teacher_idx}, {direction}]",
    )


def _nudge_in_proj(
    block: nn.Module,
    q_weight: Optional[torch.Tensor],
    k_weight: Optional[torch.Tensor],
    v_weight: Optional[torch.Tensor],
    student_idx: int,
    teacher_idx: int,
) -> None:
    """Approximately nudge a student mixer's in_proj with teacher Q/K/V weights.

    The Mamba-2 ``in_proj`` has shape ``[2*d_inner + conv_dim + nheads, d_model]``
    which does **not** factorize cleanly into Q/K/V.  This helper copies as many rows
    as are available from the stacked ``[Q; K; V]`` tensor into the first rows of
    ``in_proj.weight``, and warns that the mapping is approximate.  Only the forward
    mixer is nudged (the backward mixer reuses the same nudge since the backward scan
    is a reversed view of the same sequence, not a separately attended sub-sequence).

    Args:
        block: :class:`~dimba.models.denoiser.Mamba2Block` to update.
        q_weight: Teacher Q weight ``[d_t, d_t]`` or None.
        k_weight: Teacher K weight ``[d_t, d_t]`` or None.
        v_weight: Teacher V weight ``[d_t, d_t]`` or None.
        student_idx: Student block index (for warnings).
        teacher_idx: Teacher layer index (for warnings).
    """
    mixer = getattr(block, "mamba_fwd", None)
    if mixer is None:
        return

    in_proj: Optional[nn.Linear] = getattr(mixer, "in_proj", None)
    if in_proj is None or not hasattr(in_proj, "weight"):
        warnings.warn(
            f"principled_init: mamba_fwd at student block {student_idx} has no "
            f"in_proj.weight — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    available = [w for w in [q_weight, k_weight, v_weight] if w is not None]
    if not available:
        warnings.warn(
            f"principled_init: all of Q/K/V are None for teacher layer {teacher_idx} "
            f"(student {student_idx}) — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    # Stack available projections: shape [n_avail * d_t, d_t]
    try:
        stacked = torch.cat(available, dim=0)  # [n * d_t, d_t]
    except Exception as exc:
        warnings.warn(
            f"principled_init: could not stack Q/K/V for teacher layer {teacher_idx}: "
            f"{exc} — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    dst_w = in_proj.weight  # [d_in_proj, d_model_s]
    n_src_rows, d_src_cols = stacked.shape
    n_dst_rows, d_dst_cols = dst_w.shape

    if d_src_cols != d_dst_cols:
        warnings.warn(
            f"principled_init: in_proj column dim mismatch at block ({student_idx}, "
            f"{teacher_idx}): student d_model={d_dst_cols}, teacher d={d_src_cols} "
            f"— skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    rows_to_copy = min(n_src_rows, n_dst_rows)
    warnings.warn(
        f"principled_init: in_proj nudge at block ({student_idx}, {teacher_idx}) is "
        f"APPROXIMATE — copying {rows_to_copy} of {n_dst_rows} in_proj rows from "
        f"stacked Q/K/V ({n_src_rows} rows).  The Mamba-2 in_proj layout does not "
        f"factor cleanly into Q/K/V; treat this as a warm-start heuristic only.",
        UserWarning,
        stacklevel=4,
    )

    with torch.no_grad():
        src_slice = stacked[:rows_to_copy].to(dst_w.dtype)
        dst_w.data[:rows_to_copy].copy_(src_slice)
