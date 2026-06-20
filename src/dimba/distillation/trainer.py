"""Distillation trainer and configuration for cross-architecture knowledge distillation.

Provides:
- :class:`DistillationConfig`: Dataclass holding all distillation hyperparameters.
- :class:`DistillationTrainer`: Orchestrates multi-stage distillation of a pretrained
  HuggingFace Transformer teacher into a DIMBA bidirectional Mamba-2 diffusion student.

The trainer supports three distillation stages:

* **Stage 1** — mixing-matrix / attention alignment (student Mamba-2 matrices vs.
  teacher attention maps).
* **Stage 2** — hidden-state alignment (student block outputs projected to teacher dim).
* **Stage 3** — standard DIMBA diffusion objective + optional soft-label KD.

Each stage freezes / unfreezes parameters as described in the spec, builds a fresh
AdamW optimiser over the trainable parameters, and loops over the dataloader for the
configured number of steps.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW

from ..models.diffusion import DIMBA
from ..training.trainer import compute_dimba_losses
from .losses import stage1_matrix_loss, stage2_hidden_loss, stage3_kd_loss
from .projectors import HeadAligner, LayerMap, Projector
from .teacher import TeacherOutputs, TeacherWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DistillationConfig
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Configuration for cross-architecture knowledge distillation.

    Attributes:
        teacher_model: HuggingFace model identifier or local path for the teacher.
        teacher_type: ``'causal'`` (decoder-only) or ``'masked'`` (encoder-only).
        mode: ``'convert'`` builds the student from the teacher via
            :func:`~dimba.distillation.surgery.build_student_from_teacher`;
            any other value expects the caller to supply a pre-built DIMBA.
        block_ffn: Whether the student DIMBA should have per-block FFN sub-layers.
        principled_init: Whether to run
            :func:`~dimba.distillation.init.principled_init_from_teacher` after
            surgery.
        layer_map_mode: Mode passed to :class:`~dimba.distillation.projectors.LayerMap`
            (``'uniform'``, ``'last'``, or ``'explicit'``).
        num_student_layers: Number of Mamba blocks in the student. ``None`` defaults
            to the teacher's number of layers.
        share_vocab: When ``True``, stage-3 adds a soft-label KD term via
            :func:`~dimba.distillation.losses.stage3_kd_loss`.
        kd_weight: Weight applied to the KD loss in stage 3.
        kd_temp: Temperature for the soft-label KD loss.
        stages: List of stage-configuration dicts; see :meth:`DistillationTrainer.run_stage`.
        device: Device string for the teacher and student.
    """

    teacher_model: str
    teacher_type: str = "causal"
    mode: str = "convert"
    block_ffn: bool = True
    principled_init: bool = False
    layer_map_mode: str = "uniform"
    num_student_layers: Optional[int] = None
    share_vocab: bool = False
    kd_weight: float = 1.0
    kd_temp: float = 2.0
    stages: List[Dict[str, Any]] = field(default_factory=list)
    device: str = "cpu"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistillationConfig":
        """Construct a :class:`DistillationConfig` from a plain dictionary.

        Unknown keys are silently ignored so that YAML configs may contain extra
        fields for documentation purposes.

        Args:
            d: Mapping of field names to values.

        Returns:
            A :class:`DistillationConfig` instance.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_student_nheads(block: nn.Module, fallback: int) -> int:
    """Derive the number of SSM heads from a Mamba2Block's forward mixer.

    Tries ``block.mamba_fwd.nheads`` (set by :class:`~dimba.models.torch_mamba2.TorchMamba2`
    and compatible CUDA kernels).  Falls back to *fallback* when the attribute is absent.

    Args:
        block: A :class:`~dimba.models.denoiser.Mamba2Block` instance.
        fallback: Value to use when ``nheads`` is not available.

    Returns:
        Integer number of SSM heads.
    """
    mixer = getattr(block, "mamba_fwd", None)
    if mixer is not None:
        nheads = getattr(mixer, "nheads", None)
        if nheads is not None:
            return int(nheads)
    return fallback


def _iter_batches(
    dataloader: Iterable,
) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Yield ``(input_ids, attention_mask)`` pairs from a dataloader regardless of batch format.

    Accepts dataloaders that yield:
    - plain ``torch.Tensor`` of shape ``[B, L]`` — attention_mask will be None.
    - dicts / mappings with an ``'input_ids'`` key — attention_mask taken from
      ``'attention_mask'`` if present, else None.

    Args:
        dataloader: Any iterable that produces batches.

    Yields:
        ``(input_ids, attention_mask)`` where attention_mask may be None.
    """
    for batch in dataloader:
        if isinstance(batch, torch.Tensor):
            yield batch, None
        elif isinstance(batch, dict):
            yield batch["input_ids"], batch.get("attention_mask")
        else:
            # Try index 0 for tuple batches
            yield batch[0], None


# ---------------------------------------------------------------------------
# DistillationTrainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Orchestrates multi-stage cross-architecture knowledge distillation.

    The trainer builds per-block :class:`~dimba.distillation.projectors.HeadAligner`
    and :class:`~dimba.distillation.projectors.Projector` modules and stores them as
    ``nn.ModuleList`` attributes on the trainer (NOT registered on the student model).
    Freezing and unfreezing of student parameters is managed per-stage.

    Args:
        model: DIMBA student model to train in-place.
        teacher: :class:`~dimba.distillation.teacher.TeacherWrapper` providing
            teacher forward passes and weight accessors.
        config: :class:`DistillationConfig` with all hyperparameters.
        layer_map: Pre-built :class:`~dimba.distillation.projectors.LayerMap`; if
            ``None`` a uniform map is constructed from the model and teacher sizes.
    """

    def __init__(
        self,
        model: DIMBA,
        teacher: TeacherWrapper,
        config: DistillationConfig,
        layer_map: Optional[LayerMap] = None,
    ) -> None:
        self.model = model
        self.teacher = teacher
        self.config = config

        if layer_map is None:
            n_student = len(model.denoiser.blocks)
            n_teacher = teacher.num_layers
            layer_map = LayerMap(n_teacher, n_student, mode=config.layer_map_mode)
        self.layer_map = layer_map

        n_blocks = len(model.denoiser.blocks)
        d_latent: int = model.d_latent
        d_teacher: int = teacher.d_model
        t_heads: int = teacher.num_heads

        # Build one HeadAligner and one Projector per student block.
        head_aligner_list: List[nn.Module] = []
        projector_list: List[nn.Module] = []

        for i, block in enumerate(model.denoiser.blocks):
            s_heads = _get_student_nheads(block, fallback=t_heads)
            head_aligner_list.append(HeadAligner(h_student=s_heads, h_teacher=t_heads))
            projector_list.append(Projector(d_in=d_latent, d_out=d_teacher))

        # Store as ModuleList on the trainer so parameters are accessible for
        # optimisation but NOT registered on the student model.
        self.head_aligners: nn.ModuleList = nn.ModuleList(head_aligner_list)
        self.projectors: nn.ModuleList = nn.ModuleList(projector_list)

        # Move auxiliary modules to the student device.
        # head_aligners are kept in fp32 so that the einsum with fp32 mixing matrices
        # (from TorchMamba2.materialize_mixing_matrix) is dtype-consistent and the
        # stage1 MSE loss is computed in fp32 — matching the fp32 teacher attentions.
        # projectors are cast to the student dtype so they stay on the same compute path
        # as the student block outputs in stage2.
        _student_param = next(model.parameters())
        self.head_aligners.to(device=_student_param.device)
        self.projectors.to(device=_student_param.device, dtype=_student_param.dtype)

        logger.info(
            "DistillationTrainer: %d student blocks, d_latent=%d, d_teacher=%d, "
            "t_heads=%d.",
            n_blocks,
            d_latent,
            d_teacher,
            t_heads,
        )

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_all_model(self) -> None:
        """Freeze all student model parameters."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _unfreeze_all_model(self) -> None:
        """Unfreeze all student model parameters."""
        for p in self.model.parameters():
            p.requires_grad_(True)

    def _freeze_ffn(self) -> None:
        """Freeze FFN sub-layers in all student Mamba2Blocks."""
        for block in self.model.denoiser.blocks:
            ffn = getattr(block, "ffn", None)
            if ffn is not None:
                for p in ffn.parameters():
                    p.requires_grad_(False)

    def _set_stage1_trainable(self) -> None:
        """Stage 1: only mixers and head_aligners trainable."""
        self._freeze_all_model()
        # Unfreeze mixer parameters (mamba_fwd, mamba_bwd) in each block.
        for block in self.model.denoiser.blocks:
            for attr in ("mamba_fwd", "mamba_bwd"):
                mixer = getattr(block, attr, None)
                if mixer is not None:
                    for p in mixer.parameters():
                        p.requires_grad_(True)
        # Unfreeze head_aligners.
        for p in self.head_aligners.parameters():
            p.requires_grad_(True)
        # Projectors frozen in stage 1.
        for p in self.projectors.parameters():
            p.requires_grad_(False)

    def _set_stage2_trainable(self) -> None:
        """Stage 2: blocks and projectors trainable."""
        self._freeze_all_model()
        # Unfreeze all Mamba2Block parameters.
        for block in self.model.denoiser.blocks:
            for p in block.parameters():
                p.requires_grad_(True)
        # Unfreeze projectors.
        for p in self.projectors.parameters():
            p.requires_grad_(True)
        # Head aligners not needed for stage 2.
        for p in self.head_aligners.parameters():
            p.requires_grad_(False)

    def _set_stage3_trainable(self, freeze_ffn: bool = False) -> None:
        """Stage 3: whole model trainable (optionally freeze FFN)."""
        self._unfreeze_all_model()
        if freeze_ffn:
            self._freeze_ffn()
        # Projectors and head_aligners not used in stage 3 — freeze them.
        for p in self.head_aligners.parameters():
            p.requires_grad_(False)
        for p in self.projectors.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    def run_stage(self, stage: Dict[str, Any], dataloader: Iterable) -> None:
        """Run one distillation stage for a fixed number of optimisation steps.

        The stage dict must contain:

        - ``'name'``: one of ``'stage1'``, ``'stage2'``, ``'stage3'``.
        - ``'steps'``: number of minibatch steps to run.
        - ``'lr'``: learning rate for the fresh AdamW optimiser.

        Optional keys:

        - ``'freeze_ffn'`` (stage 3 only): freeze FFN sub-layers.
        - ``'kd_weight'``: override ``config.kd_weight`` for this stage.
        - ``'kd_temp'``: override ``config.kd_temp`` for this stage.
        - ``'ce_loss_weight'``: weight for the cross-entropy anchor in stage 3.
        - ``'min_snr_gamma'``: min-SNR gamma for stage 3.

        Args:
            stage: Stage configuration dictionary.
            dataloader: Iterable that yields batches (tensors or dicts with
                ``'input_ids'``).

        Raises:
            ValueError: If ``stage['name']`` is not a recognised stage name.
        """
        stage_name: str = stage["name"]
        n_steps: int = int(stage["steps"])
        lr: float = float(stage["lr"])

        if stage_name not in ("stage1", "stage2", "stage3"):
            raise ValueError(
                f"DistillationTrainer.run_stage: unknown stage name {stage_name!r}; "
                "expected 'stage1', 'stage2', or 'stage3'."
            )

        # ---- Set trainability masks ----
        if stage_name == "stage1":
            self._set_stage1_trainable()
            trainable_params = list(
                filter(lambda p: p.requires_grad, self.model.parameters())
            ) + list(filter(lambda p: p.requires_grad, self.head_aligners.parameters()))
        elif stage_name == "stage2":
            self._set_stage2_trainable()
            trainable_params = list(
                filter(lambda p: p.requires_grad, self.model.parameters())
            ) + list(filter(lambda p: p.requires_grad, self.projectors.parameters()))
        else:  # stage3
            freeze_ffn: bool = bool(stage.get("freeze_ffn", False))
            self._set_stage3_trainable(freeze_ffn=freeze_ffn)
            trainable_params = list(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )

        if not trainable_params:
            warnings.warn(
                f"DistillationTrainer.run_stage({stage_name}): no trainable parameters found.",
                UserWarning,
                stacklevel=2,
            )

        # ---- Build fresh AdamW ----
        optimizer = AdamW(trainable_params, lr=lr)

        # ---- Determine whether stage 1 needs mixing matrices ----
        needs_matrices = stage_name == "stage1"
        use_simple = bool(self.model.config.get("use_simple_mamba", False))
        if needs_matrices and use_simple:
            # Short-circuit: stage1 is a no-op with use_simple_mamba=True because
            # there are no mixing matrices to align.  Skip entirely rather than
            # burning n_steps of zero-loss optimizer steps.
            warnings.warn(
                "DistillationTrainer: stage1 is a no-op with use_simple_mamba=True "
                "(no mixing matrices to align); skipping the stage entirely instead "
                "of running %d zero-loss steps." % n_steps,
                UserWarning,
                stacklevel=2,
            )
            return

        # ---- KD / loss overrides ----
        kd_weight: float = float(stage.get("kd_weight", self.config.kd_weight))
        kd_temp: float = float(stage.get("kd_temp", self.config.kd_temp))
        ce_loss_weight: float = float(stage.get("ce_loss_weight", 1.0))
        min_snr_gamma: float = float(stage.get("min_snr_gamma", 5.0))

        teacher_type: str = self.config.teacher_type
        # Normalise vocabulary: 'masked' (TeacherWrapper/DistillationConfig term for
        # encoder-only models) maps to 'bidirectional' (stage1_matrix_loss term).
        if teacher_type == "masked":
            teacher_type = "bidirectional"

        # ---- Training loop ----
        self.model.train()
        self.head_aligners.train()
        self.projectors.train()
        batch_iterator = _iter_batches(dataloader)
        step = 0

        while step < n_steps:
            try:
                input_ids, attention_mask = next(batch_iterator)  # type: ignore[call-overload]
            except StopIteration:
                # Restart the dataloader iterator when exhausted.
                batch_iterator = _iter_batches(dataloader)
                try:
                    input_ids, attention_mask = next(batch_iterator)  # type: ignore[call-overload]
                except StopIteration:
                    logger.warning(
                        "DistillationTrainer: dataloader exhausted after %d steps "
                        "(requested %d). Stopping stage early.",
                        step,
                        n_steps,
                    )
                    break

            _device = next(self.model.parameters()).device
            input_ids = input_ids.to(_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(_device)

            optimizer.zero_grad()

            if stage_name == "stage1":
                loss = self._stage1_step(input_ids, teacher_type, needs_matrices)
            elif stage_name == "stage2":
                loss = self._stage2_step(input_ids)
            else:
                loss = self._stage3_step(
                    input_ids,
                    ce_loss_weight=ce_loss_weight,
                    min_snr_gamma=min_snr_gamma,
                    kd_weight=kd_weight,
                    kd_temp=kd_temp,
                    loss_mask=attention_mask,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            step += 1

            if step % max(1, n_steps // 10) == 0 or step == n_steps:
                logger.info(
                    "DistillationTrainer [%s] step %d/%d — loss=%.6f",
                    stage_name,
                    step,
                    n_steps,
                    loss.item(),
                )

    # ------------------------------------------------------------------
    # Per-stage loss computations
    # ------------------------------------------------------------------

    def _stage1_step(
        self,
        input_ids: torch.Tensor,
        teacher_type: str,
        needs_matrices: bool,
    ) -> torch.Tensor:
        """Compute the stage-1 mixing-matrix alignment loss for one minibatch.

        Args:
            input_ids: Token ids ``[B, L]``.
            teacher_type: ``'causal'`` or ``'bidirectional'``.
            needs_matrices: Whether to request Mamba-2 mixing matrices. When
                ``False`` (e.g. ``use_simple_mamba=True``), returns a zero loss.

        Returns:
            Scalar loss tensor with gradients.
        """
        if not needs_matrices:
            device = input_ids.device
            return torch.tensor(0.0, device=device, requires_grad=True)

        try:
            align = self.model.align_forward(
                input_ids,
                return_hidden_states=False,
                return_matrices=True,
                drop_cond=True,
            )
        except NotImplementedError:
            logger.warning(
                "align_forward raised NotImplementedError for return_matrices=True; "
                "stage1 loss will be zero."
            )
            device = input_ids.device
            return torch.tensor(0.0, device=device, requires_grad=True)

        teacher_out: TeacherOutputs = self.teacher(input_ids)

        is_bidir: bool = getattr(self.model, "bidirectional", True)
        loss, _parts = stage1_matrix_loss(
            align=align,
            teacher_out=teacher_out,
            layer_map=self.layer_map,
            head_aligners=list(self.head_aligners),
            teacher_type=teacher_type,
            bidirectional=is_bidir,
        )
        return loss

    def _stage2_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the stage-2 hidden-state alignment loss for one minibatch.

        Args:
            input_ids: Token ids ``[B, L]``.

        Returns:
            Scalar loss tensor with gradients.
        """
        align = self.model.align_forward(
            input_ids,
            return_hidden_states=True,
            return_matrices=False,
            drop_cond=True,
        )
        teacher_out: TeacherOutputs = self.teacher(input_ids)

        loss, _parts = stage2_hidden_loss(
            align=align,
            teacher_out=teacher_out,
            layer_map=self.layer_map,
            projectors=list(self.projectors),
        )
        return loss

    def _stage3_step(
        self,
        input_ids: torch.Tensor,
        *,
        ce_loss_weight: float,
        min_snr_gamma: float,
        kd_weight: float,
        kd_temp: float,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the stage-3 diffusion + optional KD loss for one minibatch.

        Args:
            input_ids: Token ids ``[B, L]``.
            ce_loss_weight: Weight for the cross-entropy anchor term.
            min_snr_gamma: Min-SNR gamma clamp value.
            kd_weight: Weight applied to the soft-label KD term.
            kd_temp: Temperature for the soft-label KD distribution.
            loss_mask: Optional padding mask ``[B, L]`` (1 = real token, 0 = pad).

        Returns:
            Scalar loss tensor with gradients.
        """
        device = input_ids.device
        B = input_ids.shape[0]

        # Sample random timesteps uniformly for the diffusion objective.
        t = torch.randint(
            0,
            self.model.num_diffusion_steps,
            (B,),
            device=device,
        )

        loss, _parts = compute_dimba_losses(
            self.model,
            input_ids,
            t,
            ce_loss_weight=ce_loss_weight,
            min_snr_gamma=min_snr_gamma,
            loss_mask=loss_mask,
        )

        if self.config.share_vocab and kd_weight > 0.0:
            teacher_out: TeacherOutputs = self.teacher(input_ids)
            if teacher_out.logits is not None:
                # Obtain student logits via a t=0 clean pass through the full
                # decode-then-head pipeline.  predict_token_logits handles
                # encode_latent -> denoiser -> _to_x0_latent -> decode_latent ->
                # output_head correctly for both latent and non-latent modes,
                # avoiding the shape mismatch that arises from passing the raw
                # d_latent denoiser output directly to output_head (which expects
                # d_model decoded-embedding space).
                t_zero = torch.zeros(B, dtype=torch.long, device=device)
                student_logits: torch.Tensor = self.model.predict_token_logits(
                    input_ids, t_zero
                )

                teacher_logits: torch.Tensor = teacher_out.logits

                # Require matching vocab sizes for soft-label KD.
                if student_logits.shape[-1] == teacher_logits.shape[-1]:
                    kd_loss = stage3_kd_loss(
                        student_logits,
                        teacher_logits.to(student_logits.dtype),
                        kd_temp=kd_temp,
                    )
                    loss = loss + kd_weight * kd_loss
                else:
                    warnings.warn(
                        f"DistillationTrainer stage3: vocabulary size mismatch "
                        f"(student={student_logits.shape[-1]}, "
                        f"teacher={teacher_logits.shape[-1]}); "
                        "KD loss skipped. Set config.share_vocab=False to suppress.",
                        UserWarning,
                        stacklevel=2,
                    )

        return loss

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, dataloader: Iterable) -> DIMBA:
        """Run all configured distillation stages in order.

        Iterates over ``config.stages``, calling :meth:`run_stage` for each.
        After all stages complete, the student model is returned.

        Args:
            dataloader: Iterable that yields batches (tensors ``[B, L]`` or dicts
                with ``'input_ids'``).

        Returns:
            The trained DIMBA student model (mutated in-place).
        """
        if not self.config.stages:
            logger.warning(
                "DistillationTrainer.run: config.stages is empty — no training performed."
            )
            return self.model

        for stage_cfg in self.config.stages:
            stage_name = stage_cfg.get("name", "<unnamed>")
            logger.info("DistillationTrainer: starting stage %r.", stage_name)
            self.run_stage(stage_cfg, dataloader)
            logger.info("DistillationTrainer: finished stage %r.", stage_name)

        return self.model
