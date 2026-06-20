"""Cross-architecture weight surgery for DIMBA distillation.

Provides :func:`build_student_from_teacher`, which instantiates a DIMBA student
whose architecture mirrors a HuggingFace Transformer teacher: same vocabulary,
same hidden dimension, optional FFN structure, and optionally inherits the
teacher's embedding, output-head, and per-layer FFN weights.

Shape mismatches are always handled by warning and skipping — this function
never raises on a mismatch so that experimental configurations do not block
training runs.
"""

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..models.diffusion import DIMBA
from .projectors import LayerMap

logger = logging.getLogger(__name__)


def _warn(msg: str) -> None:
    """Emit a warning through both the logging and warnings subsystems."""
    logger.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def build_student_from_teacher(
    teacher,
    *,
    num_student_layers: Optional[int] = None,
    block_ffn: bool = True,
    inherit_embeddings: bool = True,
    inherit_ffn: bool = True,
    inherit_head: bool = True,
    layer_map_mode: str = "uniform",
    padding_idx: Optional[int] = None,
    use_simple_mamba: bool = False,
    **dimba_overrides,
) -> "tuple[DIMBA, LayerMap]":
    """Instantiate a DIMBA student initialized (optionally) from a teacher's weights.

    Builds a DIMBA with the same vocabulary and hidden dimension as *teacher*,
    constructs a :class:`~dimba.distillation.projectors.LayerMap` between them,
    and — when the relevant flags are set — copies the teacher's embedding matrix,
    output-head weights, and per-layer FFN state dicts into the student.

    Shape mismatches between teacher and student tensors are handled gracefully:
    a warning is emitted and the affected copy is skipped. The function never
    raises on a shape mismatch.

    Args:
        teacher: A :class:`~dimba.distillation.teacher.TeacherWrapper` instance
            (or any object exposing the same interface).
        num_student_layers: Number of Mamba blocks in the student. Defaults to
            ``teacher.num_layers``.
        block_ffn: Whether to add a position-wise FFN sub-layer to each block.
            When True, the student's FFN type/hidden dim mirrors the teacher.
        inherit_embeddings: Copy ``teacher.input_embedding_weight()`` into the
            student's ``token_embed``.
        inherit_ffn: Copy ``teacher.layer_ffn_state(ti)`` into each matched
            student block's ``ffn`` sub-layer (only when ``block_ffn=True``).
        inherit_head: Copy ``teacher.output_head_weight()`` into the student's
            ``output_head.projection`` (when present and the head is a Linear).
        layer_map_mode: Passed to :class:`LayerMap` (``"uniform"`` or ``"last"``).
        padding_idx: Optional padding token index for the student embedding.
        use_simple_mamba: Force the pure-PyTorch SimpleMamba2 fallback (no CUDA
            kernel required; disables matrix-orientation distillation).
        **dimba_overrides: Additional keyword arguments forwarded verbatim to the
            :class:`~dimba.models.diffusion.DIMBA` constructor, allowing callers to
            override any default (e.g. ``d_state=32``, ``expand=4``).

    Returns:
        A ``(model, layer_map)`` tuple where *model* is the freshly constructed
        :class:`DIMBA` and *layer_map* is the :class:`LayerMap` relating student
        block indices to teacher layer indices.

    Raises:
        TypeError: If *teacher* does not expose the expected interface.
    """
    n_student = num_student_layers if num_student_layers is not None else teacher.num_layers
    n_teacher = teacher.num_layers

    # Build FFN kwargs: mirror the teacher when block_ffn is requested.
    ffn_type: str = teacher.ffn_type if block_ffn else "mlp"
    ffn_hidden: Optional[int] = teacher.ffn_hidden if block_ffn else None

    # Construct DIMBA with teacher-matched vocabulary and hidden dim.
    dimba_kwargs = dict(
        vocab_size=teacher.vocab_size,
        d_model=teacher.d_model,
        num_denoiser_layers=n_student,
        block_ffn=block_ffn,
        ffn_type=ffn_type,
        ffn_hidden=ffn_hidden,
        latent_diffusion=False,
        padding_idx=padding_idx,
        use_simple_mamba=use_simple_mamba,
    )
    dimba_kwargs.update(dimba_overrides)

    model = DIMBA(**dimba_kwargs)

    # Build the layer mapping (student_idx -> teacher_idx).
    layer_map = LayerMap(n_teacher, n_student, mode=layer_map_mode)

    # ------------------------------------------------------------------ embeddings
    if inherit_embeddings:
        try:
            teacher_emb = teacher.input_embedding_weight()  # [Vt, dt]
            student_emb = model.token_embed.get_weight()    # [Vs, ds]
            if teacher_emb.shape != student_emb.shape:
                _warn(
                    f"surgery: embedding shape mismatch — teacher {tuple(teacher_emb.shape)} "
                    f"vs student {tuple(student_emb.shape)}; skipping embedding copy."
                )
            else:
                with torch.no_grad():
                    student_emb.copy_(teacher_emb)
                logger.info("surgery: copied teacher input embeddings -> student token_embed.")
        except Exception as exc:  # noqa: BLE001
            _warn(f"surgery: could not copy embeddings ({exc}); skipping.")

    # ------------------------------------------------------------------ output head
    # Detect whether the teacher is weight-tied (lm_head shares storage with wte).
    # When tied, the same matrix will be copied into both token_embed and output_head,
    # but the student is NOT weight-tied by default, so they will train independently.
    try:
        _teacher_tied = (
            teacher.output_head_weight() is teacher.input_embedding_weight()
        )
    except Exception:  # noqa: BLE001
        _teacher_tied = False
    if _teacher_tied and not dimba_kwargs.get("use_weight_tying", False):
        _warn(
            "surgery: teacher is weight-tied (lm_head shares storage with wte) but "
            "student is not; embedding and head are initialized from the same matrix "
            "but will train independently (tying invariant intentionally broken). "
            "Pass use_weight_tying=True in dimba_overrides to preserve tying."
        )

    if inherit_head:
        try:
            teacher_head_w = teacher.output_head_weight()  # [Vt, dt] or None
            if teacher_head_w is not None:
                head_proj = model.output_head.projection
                if not isinstance(head_proj, nn.Linear):
                    _warn(
                        "surgery: student output_head.projection is not an nn.Linear "
                        "(weight-tied or custom head); skipping head copy."
                    )
                else:
                    # nn.Linear.weight is [out_features, in_features] == [vocab, d_model].
                    if head_proj.weight.shape != teacher_head_w.shape:
                        _warn(
                            f"surgery: output head shape mismatch — teacher "
                            f"{tuple(teacher_head_w.shape)} vs student "
                            f"{tuple(head_proj.weight.shape)}; skipping head copy."
                        )
                    else:
                        with torch.no_grad():
                            head_proj.weight.copy_(teacher_head_w)
                        logger.info("surgery: copied teacher output head -> student output_head.")
        except Exception as exc:  # noqa: BLE001
            _warn(f"surgery: could not copy output head ({exc}); skipping.")

    # ------------------------------------------------------------------ per-layer FFN
    if inherit_ffn and block_ffn:
        n_pairs = 0
        n_loaded = 0
        for student_idx, teacher_idx in layer_map.pairs():
            n_pairs += 1
            block = model.denoiser.blocks[student_idx]
            if block.ffn is None:
                # block_ffn=True but this particular block has no ffn (shouldn't happen,
                # but guard defensively).
                _warn(
                    f"surgery: block[{student_idx}].ffn is None despite block_ffn=True; "
                    f"skipping FFN copy for layer pair ({student_idx}, {teacher_idx})."
                )
                continue

            try:
                teacher_ffn_sd = teacher.layer_ffn_state(teacher_idx)
            except Exception as exc:  # noqa: BLE001
                _warn(
                    f"surgery: could not fetch FFN state for teacher layer {teacher_idx} "
                    f"({exc}); skipping pair ({student_idx}, {teacher_idx})."
                )
                continue

            # Shape-check each key before loading.
            student_ffn_sd = block.ffn.state_dict()
            all_ok = True
            for key, t_param in teacher_ffn_sd.items():
                if key not in student_ffn_sd:
                    _warn(
                        f"surgery: teacher FFN key '{key}' not found in student "
                        f"block[{student_idx}].ffn; skipping pair ({student_idx}, {teacher_idx})."
                    )
                    all_ok = False
                    break
                s_param = student_ffn_sd[key]
                if t_param.shape != s_param.shape:
                    _warn(
                        f"surgery: FFN weight shape mismatch for key '{key}' — teacher "
                        f"{tuple(t_param.shape)} vs student {tuple(s_param.shape)}; "
                        f"skipping pair ({student_idx}, {teacher_idx})."
                    )
                    all_ok = False
                    break

            if not all_ok:
                continue

            # All shapes match — load with strict=False to tolerate any extra keys
            # that might exist in the student FFN but are absent from teacher state.
            try:
                missing, unexpected = block.ffn.load_state_dict(teacher_ffn_sd, strict=False)
                if missing:
                    _warn(
                        f"surgery: block[{student_idx}].ffn load_state_dict missing keys: "
                        f"{missing}; pair ({student_idx}, {teacher_idx})."
                    )
                logger.info(
                    "surgery: loaded teacher FFN (layer %d) -> student block[%d].ffn.",
                    teacher_idx,
                    student_idx,
                )
                n_loaded += 1
            except Exception as exc:  # noqa: BLE001
                _warn(
                    f"surgery: load_state_dict failed for block[{student_idx}].ffn "
                    f"({exc}); skipping pair ({student_idx}, {teacher_idx})."
                )

        if n_pairs > 0 and n_loaded == 0:
            _warn(
                "surgery: inherit_ffn=True and block_ffn=True but ZERO of "
                f"{n_pairs} FFN layer pairs were inherited — every pair was skipped. "
                "This almost always means an ffn_type/key-name or shape mismatch "
                "between teacher and student (e.g. student built as 'mlp' but teacher "
                "FFN is swiglu). The student will train with RANDOMLY-INITIALIZED FFNs. "
                "Check that teacher.ffn_type matches the teacher's actual MLP structure."
            )
        elif 0 < n_loaded < n_pairs:
            _warn(f"surgery: only {n_loaded}/{n_pairs} FFN pairs inherited; rest skipped.")

    return model, layer_map
