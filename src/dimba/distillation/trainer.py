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
configured optimizer over the trainable parameters, and loops over the dataloader for
the configured number of steps.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW

from ..models.diffusion import DIMBA
from ..training.distributed import (
    DistributedContext,
    raise_distributed_failure,
    reduce_metrics,
    restore_rng_state as restore_distributed_rng_state,
    wrap_ddp,
)
from ..training.optimizers import HybridMuon, build_optimizer
from ..training.trainer import _run_dimba_loss_forward, compute_dimba_losses
from ..utils.checkpointing import restore_rng_state
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


class _Stage3LossModule(nn.Module):
    """Register the student while keeping the complete Stage-3 loss in one DDP forward."""

    def __init__(self, trainer: "DistillationTrainer") -> None:
        super().__init__()
        self.model = trainer.model
        self._trainer = trainer

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        ce_loss_weight: float,
        min_snr_gamma: float,
        snr_floor: float,
        ce_time_fade: bool,
        kd_weight: float,
        kd_temp: float,
        kd_time_fade: bool,
        loss_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._trainer._stage3_step(
            input_ids,
            ce_loss_weight=ce_loss_weight,
            min_snr_gamma=min_snr_gamma,
            snr_floor=snr_floor,
            ce_time_fade=ce_time_fade,
            kd_weight=kd_weight,
            kd_temp=kd_temp,
            kd_time_fade=kd_time_fade,
            loss_mask=loss_mask,
        )


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
        log_hook: Optional[Callable[[str, int, int, float, Any], Optional[str]]] = None,
        distributed_context: Optional[DistributedContext] = None,
    ) -> None:
        self.model = model
        self.teacher = teacher
        self.config = config
        self.distributed_context = distributed_context
        # Optional callback fired at every log point with
        # (stage_name, step, n_steps, loss, optimizer). Returning the string
        # "stop" ends the current stage early. scripts/train_4090.py uses this to
        # write training_state.json DURING distillation (so scripts/monitor.py is
        # not blind here — run_distill writes no state otherwise) and to apply
        # one-shot live overrides (lr / stop) from training_state_override.json.
        self._log_hook = log_hook

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

    def run_stage(
        self,
        stage: Dict[str, Any],
        dataloader: Iterable,
        resume_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Run one distillation stage for a fixed number of optimisation steps.

        The stage dict must contain:

        - ``'name'``: one of ``'stage1'``, ``'stage2'``, ``'stage3'``.
        - ``'steps'``: number of minibatch steps to run.
        - ``'lr'``: learning rate for the fresh optimiser.

        Optional keys:

        - ``'freeze_ffn'`` (stage 3 only): freeze FFN sub-layers.
        - ``'kd_weight'``: override ``config.kd_weight`` for this stage.
        - ``'kd_temp'``: override ``config.kd_temp`` for this stage.
        - ``'ce_loss_weight'``: weight for the cross-entropy anchor in stage 3.
        - ``'min_snr_gamma'``: min-SNR gamma for stage 3.
        - ``'snr_floor'``: minimum diffusion-regression weight in stage 3.
        - ``'ce_time_fade'``: fade the CE anchor toward zero at maximum noise.
        - ``'kd_time_fade'``: fade teacher KD toward zero at maximum noise.
        - ``'optimizer'``: ``'adamw'`` (default) or ``'muon'`` (stage 3 only).
        - ``'weight_decay'``: optimizer weight decay (default ``0.01``).
        - ``'id'``: stable phase identifier used by resumable checkpoints.

        Args:
            stage: Stage configuration dictionary.
            dataloader: Iterable that yields batches (tensors or dicts with
                ``'input_ids'``).
            resume_state: Full checkpoint mapping for an exact within-stage resume.

        Raises:
            ValueError: If ``stage['name']`` is not a recognised stage name.
        """
        stage_name: str = stage["name"]
        n_steps: int = int(stage["steps"])
        lr: float = float(stage["lr"])
        self.current_stage_id = str(stage.get("id", stage_name))
        self.stage_total_steps = n_steps
        self.stage_step = 0
        self.stage_completed = False

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
            raise RuntimeError(
                f"DistillationTrainer.run_stage({stage_name}): no trainable parameters found"
            )

        # Muon is intentionally Stage-3-only: Stages 1/2 optimize temporary
        # alignment modules that are not part of the student model.
        optimizer_name = str(stage.get("optimizer", "adamw")).lower()
        weight_decay = float(stage.get("weight_decay", 0.01))
        if optimizer_name == "muon":
            if stage_name != "stage3":
                raise ValueError("Muon is only supported for distillation stage3")
            optimizer = build_optimizer(
                self.model,
                name="muon",
                lr=lr,
                weight_decay=weight_decay,
                fused=next(self.model.parameters()).device.type == "cuda",
            )
            assert isinstance(optimizer, HybridMuon)
            logger.info(
                "DistillationTrainer [stage3] Muon split: %d hidden matrices; "
                "%d embedding/head/vector tensors on AdamW",
                len(optimizer.muon_parameter_names),
                len(optimizer.adamw_parameter_names),
            )
        elif optimizer_name == "adamw":
            optimizer = AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                fused=next(self.model.parameters()).device.type == "cuda",
            )
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer_name!r}; expected 'adamw' or 'muon'"
            )
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name

        stage3_loss_module: Optional[nn.Module] = None
        context = getattr(self, "distributed_context", None)
        if stage_name == "stage3" and context is not None and context.enabled:
            # DDP must see the complete loss forward. Calling the raw student through
            # ``ddp.module`` would bypass reducer bookkeeping for the diffusion, CE,
            # and chunked KD branches. The teacher remains replicated/no-grad.
            stage3_loss_module = wrap_ddp(
                _Stage3LossModule(self),
                context,
                find_unused_parameters=True,
            )
        self._stage3_loss_module = stage3_loss_module

        start_step = 0
        if resume_state is not None:
            if resume_state.get("stage_id") != self.current_stage_id:
                raise ValueError(
                    f"checkpoint stage {resume_state.get('stage_id')!r} does not match "
                    f"requested stage {self.current_stage_id!r}"
                )
            saved_optimizer = resume_state.get("optimizer_name")
            if saved_optimizer != optimizer_name:
                raise ValueError(
                    f"checkpoint optimizer {saved_optimizer!r} does not match "
                    f"requested optimizer {optimizer_name!r}"
                )
            rng_key = "rng_states" if context is not None and context.enabled else "rng_state"
            if "optimizer_state_dict" not in resume_state or rng_key not in resume_state:
                raise ValueError("checkpoint is missing optimizer or RNG state")
            if isinstance(optimizer, HybridMuon):
                saved_policy = resume_state.get("optimizer_policy", {})
                if tuple(saved_policy.get("muon_parameter_names", ())) != tuple(
                    optimizer.muon_parameter_names
                ) or tuple(saved_policy.get("adamw_parameter_names", ())) != tuple(
                    optimizer.adamw_parameter_names
                ):
                    raise ValueError("checkpoint Muon parameter-group policy changed")
            start_step = int(resume_state.get("stage_step", -1))
            if not 0 <= start_step <= n_steps:
                raise ValueError(
                    f"checkpoint stage_step={start_step} is outside [0, {n_steps}]"
                )
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if context is not None and context.enabled:
                restore_distributed_rng_state(resume_state, context)
            else:
                restore_rng_state(resume_state["rng_state"])
            logger.info(
                "DistillationTrainer [%s] restored optimizer/RNG at step %d/%d",
                self.current_stage_id,
                start_step,
                n_steps,
            )

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
            self._stage1_warned = True
            return

        # ---- KD / loss overrides ----
        kd_weight: float = float(stage.get("kd_weight", self.config.kd_weight))
        kd_temp: float = float(stage.get("kd_temp", self.config.kd_temp))
        ce_loss_weight: float = float(stage.get("ce_loss_weight", 1.0))
        min_snr_gamma: float = float(stage.get("min_snr_gamma", 5.0))
        snr_floor: float = float(stage.get("snr_floor", 1e-3))
        ce_time_fade: bool = bool(stage.get("ce_time_fade", False))
        kd_time_fade: bool = bool(stage.get("kd_time_fade", False))

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
        step = start_step
        self.stage_step = step
        self.stage_completed = step >= n_steps

        while step < n_steps:
            try:
                input_ids, attention_mask = next(batch_iterator)  # type: ignore[call-overload]
            except StopIteration:
                # Restart the dataloader iterator when exhausted.
                batch_iterator = _iter_batches(dataloader)
                try:
                    input_ids, attention_mask = next(batch_iterator)  # type: ignore[call-overload]
                except StopIteration:
                    raise RuntimeError(
                        "DistillationTrainer: dataloader is empty; refusing to mark "
                        f"{stage_name} complete at step {step}/{n_steps}"
                    )

            _device = next(self.model.parameters()).device
            input_ids = input_ids.to(_device, non_blocking=_device.type == "cuda")
            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    _device, non_blocking=_device.type == "cuda"
                )

            optimizer.zero_grad()

            loss_error: Optional[Exception] = None
            loss: Optional[torch.Tensor] = None
            try:
                if stage_name == "stage1":
                    loss = self._stage1_step(input_ids, teacher_type, needs_matrices)
                elif stage_name == "stage2":
                    loss = self._stage2_step(input_ids)
                elif stage3_loss_module is not None:
                    loss = stage3_loss_module(
                        input_ids,
                        ce_loss_weight=ce_loss_weight,
                        min_snr_gamma=min_snr_gamma,
                        snr_floor=snr_floor,
                        ce_time_fade=ce_time_fade,
                        kd_weight=kd_weight,
                        kd_temp=kd_temp,
                        kd_time_fade=kd_time_fade,
                        loss_mask=attention_mask,
                    )
                else:
                    loss = self._stage3_step(
                        input_ids,
                        ce_loss_weight=ce_loss_weight,
                        min_snr_gamma=min_snr_gamma,
                        snr_floor=snr_floor,
                        ce_time_fade=ce_time_fade,
                        kd_weight=kd_weight,
                        kd_temp=kd_temp,
                        kd_time_fade=kd_time_fade,
                        loss_mask=attention_mask,
                    )
            except Exception as exc:  # synchronize forward/OOM failure before backward
                if stage3_loss_module is None:
                    raise
                loss_error = exc

            if stage3_loss_module is not None and context is not None:
                raise_distributed_failure(
                    context,
                    f"{self.current_stage_id} forward at step {step}",
                    error=loss_error,
                    valid=torch.isfinite(loss).detach() if loss is not None else None,
                )
            assert loss is not None

            if stage_name == "stage1" and getattr(self, "_stage1_warned", False):
                break
            finite = torch.isfinite(loss).detach()
            if stage3_loss_module is not None:
                pass  # checked collectively above so no rank can strand its peers
            elif loss.device.type == "cuda" and hasattr(torch, "_assert_async"):
                torch._assert_async(
                    finite, f"non-finite loss in {self.current_stage_id} at step {step}"
                )
            elif not bool(finite):
                raise FloatingPointError(
                    f"non-finite loss in {self.current_stage_id} at step {step}"
                )

            update_error: Optional[Exception] = None
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            except Exception as exc:  # backward/state allocation can OOM on step 1
                if stage3_loss_module is None:
                    raise
                update_error = exc
            if stage3_loss_module is not None and context is not None:
                raise_distributed_failure(
                    context,
                    f"{self.current_stage_id} update at step {step}",
                    error=update_error,
                )

            step += 1
            self.stage_step = step

            # Log at least every 50 steps so long stages (e.g. stage3 with 30k steps)
            # show progress promptly instead of staying silent for thousands of steps
            # (n_steps//10 alone would be every 3000 steps for stage3).
            log_every = max(1, min(n_steps // 10, 50))
            if step % log_every == 0 or step == n_steps:
                if stage3_loss_module is not None and context is not None:
                    loss_value = reduce_metrics({"loss": loss}, context)["loss"].item()
                else:
                    loss_value = loss.item()
                if context is None or not context.enabled or context.is_main:
                    logger.info(
                        "DistillationTrainer [%s] step %d/%d — loss=%.6f",
                        stage_name,
                        step,
                        n_steps,
                        loss_value,
                    )
                if self._log_hook is not None:
                    if (
                        self._log_hook(stage_name, step, n_steps, loss_value, optimizer)
                        == "stop"
                    ):
                        if context is None or not context.enabled or context.is_main:
                            logger.warning(
                                "DistillationTrainer [%s]: early stop requested via "
                                "log_hook at step %d/%d.", stage_name, step, n_steps,
                            )
                        break

        self.stage_completed = step >= n_steps

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
        except (NotImplementedError, RuntimeError, AttributeError, TypeError) as exc:
            # Mark Stage 1 unavailable and let the caller enforce its run policy on:
            #   • NotImplementedError — mixer exposes no matrix path
            #   • RuntimeError        — OOM guard at large B*L (2.3 GB+ at B=32, L=512)
            #   • AttributeError/TypeError — the param-based materialization for the CUDA
            #     mamba_ssm.Mamba2 kernel hit an unexpected attribute name / signature on
            #     this mamba_ssm version (the math is verified, but the API could drift).
            if not getattr(self, "_stage1_warned", False):
                logger.warning(
                    "stage1 matrix loss unavailable (%s: %s); stopping this stage.",
                    type(exc).__name__, exc,
                )
                self._stage1_warned = True
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
        snr_floor: float = 1e-3,
        ce_time_fade: bool = False,
        kd_weight: float,
        kd_temp: float,
        kd_time_fade: bool = False,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the stage-3 diffusion + optional KD loss for one minibatch.

        Args:
            input_ids: Token ids ``[B, L]``.
            ce_loss_weight: Weight for the cross-entropy anchor term.
            min_snr_gamma: Min-SNR gamma clamp value.
            snr_floor: Minimum diffusion-regression weight at high noise.
            ce_time_fade: Fade CE toward zero as the sampled noise approaches one.
            kd_weight: Weight applied to the soft-label KD term.
            kd_temp: Temperature for the soft-label KD distribution.
            kd_time_fade: Fade KD as ``1-t`` so a per-sample clean teacher
                target is not imposed when the student input is pure noise.
            loss_mask: Optional padding mask ``[B, L]`` (1 = real token, 0 = pad).

        Returns:
            Scalar loss tensor with gradients.
        """
        device = input_ids.device
        B = input_ids.shape[0]

        # Sample timesteps for the diffusion objective.
        #
        # The budget28 post-mortem found a razor-sharp failure at pure noise. For a
        # logit-normal flow model, sample the validated 50/50 mixture: logit-normal
        # covers the useful mid-noise band while uniform draws keep pressure on the
        # t≈1 endpoint. Logit-normal alone undersamples that endpoint; uniform alone
        # undersamples the middle. Each row independently selects a component so the
        # distribution remains a true 50/50 mixture even for batch size one.
        _fm = bool(getattr(self.model, "use_flow_matching", False))
        _fm_logit_normal = _fm and bool(
            getattr(getattr(self.model, "flow_schedule", None), "logit_normal_sampling", False)
        )
        if _fm_logit_normal:
            t_uniform = torch.randint(
                0,
                self.model.num_diffusion_steps,
                (B,),
                device=device,
            )
            t_logit_normal = self.model.noise_schedule.sample_timesteps(
                B, device, mode="logit_normal"
            )
            use_uniform = torch.rand(B, device=device) < 0.5
            t = torch.where(use_uniform, t_uniform, t_logit_normal)
        else:
            t = torch.randint(
                0,
                self.model.num_diffusion_steps,
                (B,),
                device=device,
            )

        # The diffusion/CE objective and teacher KD both consume the same predicted
        # clean embedding. Reuse one student pass instead of paying for a second
        # full Mamba forward on every KD-enabled step.
        kd_pred, kd_info = _run_dimba_loss_forward(self.model, input_ids, t)
        loss, _parts = compute_dimba_losses(
            self.model,
            input_ids,
            t,
            ce_loss_weight=ce_loss_weight,
            min_snr_gamma=min_snr_gamma,
            snr_floor=snr_floor,
            ce_time_fade=ce_time_fade,
            loss_mask=loss_mask,
            _forward_output=(kd_pred, kd_info),
        )

        if self.config.share_vocab and kd_weight > 0.0:
            logits_only = getattr(self.teacher, "logits_only", None)
            with torch.no_grad():
                teacher_out: TeacherOutputs = (
                    logits_only(input_ids, attention_mask=loss_mask)
                    if callable(logits_only)
                    else self.teacher(input_ids)
                )
            if teacher_out.logits is not None:
                # Distil the continuous diffusion path at the same sampled noise
                # level as the base objective. predict_token_logits is the discrete
                # masked-token API; feeding it clean target ids at t=0 lets the
                # bidirectional student copy the answer and does not train the path
                # used by continuous generation.
                teacher_logits: torch.Tensor = teacher_out.logits
                student_vocab = int(self.model.token_embed.get_weight().shape[0])

                # Require matching vocab sizes for soft-label KD.
                if student_vocab == teacher_logits.shape[-1]:
                    # Align targets: a causal teacher's logits at position i are its
                    # distribution for token i+1, while the diffusion student's logits
                    # at position i reconstruct token i. Shift so both describe the
                    # same token: teacher[:, :-1] (predicting tokens 1..L-1) vs
                    # student[:, 1:] (reconstructing tokens 1..L-1). Unshifted, the
                    # KD term teaches next-token prediction against the CE loss's
                    # same-token target — two contradictory objectives per position.
                    # ponytail: causal teacher predicts i+1 so shift; masked/bidir predicts i, no shift.
                    kd_fade = (
                        1.0
                        - (t.float() / (self.model.num_diffusion_steps - 1)).clamp(
                            0.0, 1.0
                        )
                        if kd_time_fade
                        else None
                    )
                    # Project and reduce in batch chunks. Keeping the decoded
                    # features resident is cheap; a full [B,L,V] student tensor and
                    # its fp32 KL intermediates are not (multiple GB at H100 B=64).
                    kd_loss = loss.new_zeros(())
                    logit_chunk = 16
                    for start in range(0, B, logit_chunk):
                        stop = min(B, start + logit_chunk)
                        student_logits = self.model.output_head(kd_pred[start:stop])
                        teacher_chunk = teacher_logits[start:stop].to(
                            device=student_logits.device,
                            dtype=student_logits.dtype,
                        )
                        if self.teacher.is_causal:
                            s_lg = student_logits[:, 1:]
                            t_lg = teacher_chunk[:, :-1]
                            kd_mask = (
                                loss_mask[start:stop, 1:]
                                if loss_mask is not None
                                else None
                            )
                        else:
                            s_lg, t_lg = student_logits, teacher_chunk
                            kd_mask = (
                                loss_mask[start:stop]
                                if loss_mask is not None
                                else None
                            )
                        chunk_loss = stage3_kd_loss(
                            s_lg,
                            t_lg,
                            kd_temp=kd_temp,
                            loss_mask=kd_mask,
                            sample_weight=(
                                kd_fade[start:stop] if kd_fade is not None else None
                            ),
                            chunk_size=logit_chunk,
                        )
                        kd_loss = kd_loss + chunk_loss * ((stop - start) / B)
                    loss = loss + kd_weight * kd_loss
                else:
                    warnings.warn(
                        f"DistillationTrainer stage3: vocabulary size mismatch "
                        f"(student={student_vocab}, "
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
            if not self.stage_completed:
                raise RuntimeError(
                    f"DistillationTrainer: stage {stage_name!r} stopped at "
                    f"{self.stage_step}/{self.stage_total_steps}"
                )
            logger.info("DistillationTrainer: finished stage %r.", stage_name)

        return self.model
