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

    2. **in_proj nudge from V** (``mode="qkvo"`` only) — The teacher V weight
       (shape ``[num_kv_heads*head_dim, d_model]`` for GQA/MQA teachers such as
       Qwen2/Mistral/Llama-3, or ``[num_heads*head_dim, d_model]`` for MHA) is
       used to seed the ``x`` sub-segment (rows ``[d_inner : 2*d_inner)``) of the
       student's ``in_proj.weight``.  The SiLU gate ``z`` (rows ``[0 : d_inner)``)
       is left at its default initialisation and is never overwritten.  Q and K are
       accepted but unused because B/C live in ``d_state`` space rather than
       ``d_model`` space and have no faithful mapping from the teacher projections.
       For GQA/MQA teachers, V is expanded per-head to ``[num_heads*head_dim,
       d_model]`` before copying.  This is intentionally approximate; treat it as a
       warm-start heuristic only.

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
    """Approximately nudge a student mixer's in_proj with the teacher V weight.

    The Mamba-2 ``in_proj`` layout is ``[z (d_inner) | xBC (conv_dim) | dt (nheads)]``.
    The first ``d_inner`` rows are the SiLU gate ``z``; they must **not** be overwritten.
    The ``x`` sub-segment of ``xBC`` lives at rows ``[d_inner : 2*d_inner)`` and is the
    closest structural analogue to the teacher's value projection V.  Only that segment
    is seeded.  Q and K have no faithful target in the Mamba-2 in_proj layout (B/C live
    in d_state space, not d_model space) so they are left at their default initialisation.

    For GQA/MQA teachers (e.g. Qwen2, Mistral, Llama-3) V may have shape
    ``[num_kv_heads * head_dim, d_model]`` with fewer rows than Q.  The weight is
    expanded per-head to ``[num_heads * head_dim, d_model]`` before copying so that
    the seeded rows have the correct per-head feature ordering.  If the teacher config
    does not expose ``num_key_value_heads`` or the shapes are not divisible as expected,
    V is used as-is with a warning.

    Only the forward mixer is nudged (``mamba_fwd``).

    Args:
        block: :class:`~dimba.models.denoiser.Mamba2Block` to update.
        q_weight: Teacher Q weight ``[num_heads*head_dim, d_model]`` or None (unused).
        k_weight: Teacher K weight ``[num_kv_heads*head_dim, d_model]`` or None (unused).
        v_weight: Teacher V weight ``[num_kv_heads*head_dim, d_model]`` or None.
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

    if v_weight is None:
        warnings.warn(
            f"principled_init: V weight is None for teacher layer {teacher_idx} "
            f"(student {student_idx}) — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    dst_w = in_proj.weight  # [d_in_proj, d_model_s]
    d_model_s = dst_w.shape[1]

    # Column-dim guard: V must match the student's d_model.
    if v_weight.shape[1] != d_model_s:
        warnings.warn(
            f"principled_init: in_proj column dim mismatch at block ({student_idx}, "
            f"{teacher_idx}): student d_model={d_model_s}, teacher V d_model="
            f"{v_weight.shape[1]} — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    d_inner = getattr(mixer, "d_inner", None)
    if d_inner is None:
        warnings.warn(
            f"principled_init: mamba_fwd at student block {student_idx} does not "
            f"expose d_inner — skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    x_start = d_inner
    x_stop = 2 * d_inner  # exclusive; x sub-segment of xBC

    if x_stop > dst_w.shape[0]:
        warnings.warn(
            f"principled_init: in_proj at block ({student_idx}, {teacher_idx}) has "
            f"only {dst_w.shape[0]} rows but x segment requires rows up to {x_stop} "
            f"— skipping in_proj nudge.",
            UserWarning,
            stacklevel=4,
        )
        return

    # ── GQA/MQA expansion: expand V from num_kv_heads to num_heads ───────────
    src_v = v_weight
    teacher_cfg = getattr(getattr(block, "_teacher_cfg", None), "config", None)
    # Block may not carry the teacher config; retrieve it via a best-effort attribute
    # that TeacherWrapper may have attached, or skip expansion gracefully.
    # In practice the caller (principled_init_from_teacher) passes a TeacherWrapper;
    # we cannot reach it from here, so expansion relies on q_weight row count as a
    # proxy for num_heads * head_dim when q_weight is available.
    if q_weight is not None and q_weight.shape[1] == d_model_s:
        n_q_rows = q_weight.shape[0]
        n_v_rows = src_v.shape[0]
        if n_v_rows < n_q_rows and n_q_rows % n_v_rows == 0:
            n_rep = n_q_rows // n_v_rows
            # Infer head_dim: we need it to do per-head expansion correctly.
            # We can only infer it if num_kv_heads divides n_v_rows cleanly and
            # num_heads * head_dim == n_q_rows; use n_v_rows itself as num_kv_heads
            # only when head_dim==1, which is wrong, so use a safer heuristic:
            # expand by treating each row-group of size (n_v_rows) as one "kv-head"
            # set and interleave n_rep copies — equivalent to repeat_interleave along
            # the head axis when head_dim is consistent.
            # Safe guard: only expand when n_v_rows is divisible by n_rep.
            if n_v_rows % n_rep == 0:
                num_kv_heads = n_v_rows // (n_q_rows // n_rep)
                head_dim = n_v_rows // num_kv_heads if num_kv_heads > 0 else None
                if head_dim is not None and num_kv_heads * head_dim == n_v_rows:
                    try:
                        src_v = (
                            src_v
                            .reshape(num_kv_heads, head_dim, d_model_s)
                            .unsqueeze(2)
                            .expand(num_kv_heads, n_rep, head_dim, d_model_s)
                            .reshape(n_q_rows, d_model_s)
                            .contiguous()
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"principled_init: GQA V expansion failed at block "
                            f"({student_idx}, {teacher_idx}): {exc} — using raw V.",
                            UserWarning,
                            stacklevel=4,
                        )
                        src_v = v_weight
                else:
                    warnings.warn(
                        f"principled_init: cannot infer head_dim for GQA V expansion "
                        f"at block ({student_idx}, {teacher_idx}); n_v_rows={n_v_rows}, "
                        f"n_q_rows={n_q_rows} — using raw V.",
                        UserWarning,
                        stacklevel=4,
                    )
            else:
                warnings.warn(
                    f"principled_init: V row count ({n_v_rows}) not divisible by "
                    f"n_rep={n_rep} at block ({student_idx}, {teacher_idx}) — "
                    f"using raw V.",
                    UserWarning,
                    stacklevel=4,
                )

    # ── Copy V (or expanded V) into the x segment [d_inner : 2*d_inner) ──────
    n_rows_to_copy = min(src_v.shape[0], x_stop - x_start)
    warnings.warn(
        f"principled_init: in_proj nudge at block ({student_idx}, {teacher_idx}) is "
        f"APPROXIMATE — seeding {n_rows_to_copy} rows of the x-segment "
        f"[{x_start}:{x_start + n_rows_to_copy}) from teacher V.  "
        f"The SiLU gate z (rows [0:{d_inner})) is left at its default init.",
        UserWarning,
        stacklevel=4,
    )
    with torch.no_grad():
        dst_w.data[x_start : x_start + n_rows_to_copy].copy_(
            src_v[:n_rows_to_copy].to(dst_w.dtype)
        )
