"""Optimizer construction for DIMBA training."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


NamedParameters = List[Tuple[str, nn.Parameter]]


def _newton_schulz_zeropower(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Moonlight's five-step Newton-Schulz orthogonalization."""
    if gradient.ndim != 2:
        raise ValueError("Muon requires 2D parameter gradients")
    if not 0 < steps < 100:
        raise ValueError("Muon Newton-Schulz steps must be between 1 and 99")

    a, b, c = 3.4445, -4.7750, 2.0315
    work_dtype = torch.bfloat16 if gradient.device.type == "cuda" else torch.float32
    update = gradient.to(work_dtype)
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.mT
    update = update / update.norm().clamp(min=1e-7)

    for _ in range(steps):
        gram = update @ update.mT
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, gram_update, update, beta=a)

    if transposed:
        update = update.mT
    return update.to(gradient.dtype)


class _MuonBackport(Optimizer):
    """PyTorch 2.9 Muon behavior for runtimes where ``torch.optim.Muon`` is absent."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid Muon learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid Muon momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid Muon weight decay: {weight_decay}")
        if adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Unsupported Muon LR adjustment: {adjust_lr_fn!r}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters; "
                        f"received shape {tuple(parameter.shape)}"
                    )

    @staticmethod
    def _adjust_lr(lr: float, shape: torch.Size, mode: str) -> float:
        rows, columns = shape
        if mode == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(rows, columns))
        return lr * math.sqrt(max(1.0, rows / columns))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if torch.is_complex(parameter):
                    raise RuntimeError("Muon does not support complex parameters")

                state = self.state[parameter]
                momentum_buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(gradient)
                )
                momentum_buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = gradient.lerp(momentum_buffer, group["momentum"])
                update = _newton_schulz_zeropower(update, group["ns_steps"])
                adjusted_lr = self._adjust_lr(
                    group["lr"], parameter.shape, group["adjust_lr_fn"]
                )

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


class HybridMuon(Optimizer):
    """One interface over Muon and AdamW; supports replicated DDP, not sharded matrices."""

    def __init__(
        self,
        muon_parameters: NamedParameters,
        adamw_parameters: NamedParameters,
        *,
        lr: float,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        ns_steps: int = 5,
        fused: Optional[bool] = None,
    ) -> None:
        if not muon_parameters:
            raise ValueError("Muon selected, but the model has no eligible hidden matrices")
        if not adamw_parameters:
            raise ValueError("Muon requires an AdamW group for embeddings, heads, and vectors")

        muon_params = [parameter for _, parameter in muon_parameters]
        adamw_params = [parameter for _, parameter in adamw_parameters]
        if set(map(id, muon_params)) & set(map(id, adamw_params)):
            raise ValueError("Muon and AdamW parameter groups must be disjoint")

        muon_type = getattr(torch.optim, "Muon", _MuonBackport)
        self.muon = muon_type(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            adjust_lr_fn="match_rms_adamw",
        )
        adamw_kwargs = dict(lr=lr, weight_decay=weight_decay, betas=betas)
        if fused is not None:
            adamw_kwargs["fused"] = fused
        self.adamw = AdamW(adamw_params, **adamw_kwargs)

        # Initialize Optimizer's hook/checkpoint machinery, then expose the two
        # underlying groups as one scheduler-compatible optimizer.
        super().__init__(muon_params + adamw_params, dict(lr=lr))
        self.muon_parameter_names = tuple(name for name, _ in muon_parameters)
        self.adamw_parameter_names = tuple(name for name, _ in adamw_parameters)
        self._refresh_views()

    def _refresh_views(self) -> None:
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        self.state = {**self.muon.state, **self.adamw.state}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        self._refresh_views()
        return loss

    def load_state_dict(self, state_dict) -> None:
        """Restore a standard flat Optimizer state dict into both inner optimizers."""
        super().load_state_dict(state_dict)
        loaded_state = self.state
        loaded_groups = list(self.param_groups)
        muon_group_count = len(self.muon.param_groups)

        def restore(optimizer: Optimizer, groups) -> None:
            parameters = [parameter for group in groups for parameter in group["params"]]
            state = defaultdict(
                dict,
                (
                    (parameter, loaded_state[parameter])
                    for parameter in parameters
                    if parameter in loaded_state
                ),
            )
            optimizer.__setstate__({"state": state, "param_groups": groups})

        restore(self.muon, loaded_groups[:muon_group_count])
        restore(self.adamw, loaded_groups[muon_group_count:])
        self._refresh_views()


def split_muon_parameters(model: nn.Module) -> Tuple[NamedParameters, NamedParameters]:
    """Split hidden 2D matrices from parameters that should remain on AdamW."""
    excluded_ids = set()
    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1].lower()
        parent_name = module_name.rsplit(".", 1)[0].rsplit(".", 1)[-1].lower()
        class_name = type(module).__name__.lower()
        is_embedding = (
            isinstance(module, (nn.Embedding, nn.EmbeddingBag))
            or "embedding" in class_name
        )
        is_norm = "norm" in class_name
        is_output_projection = (
            isinstance(module, nn.Linear)
            and leaf_name in {"output_head", "lm_head"}
        ) or (
            leaf_name == "projection" and parent_name in {"output_head", "lm_head"}
        )
        if is_embedding or is_norm or is_output_projection:
            excluded_ids.update(
                id(parameter) for parameter in module.parameters(recurse=False)
            )

    muon_parameters: NamedParameters = []
    adamw_parameters: NamedParameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        basename = name.rsplit(".", 1)[-1].lower()
        use_muon = (
            parameter.ndim == 2
            and id(parameter) not in excluded_ids
            and basename not in {"a_log"}
        )
        (muon_parameters if use_muon else adamw_parameters).append((name, parameter))

    return muon_parameters, adamw_parameters


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    lr: float,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    fused: Optional[bool] = None,
) -> Optimizer:
    """Build AdamW or the production Muon+AdamW hybrid for a model."""
    optimizer_name = name.lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Cannot build an optimizer for a model with no trainable parameters")

    if optimizer_name == "adamw":
        kwargs = dict(lr=lr, weight_decay=weight_decay, betas=betas)
        if fused is not None:
            kwargs["fused"] = fused
        return AdamW(parameters, **kwargs)
    if optimizer_name != "muon":
        raise ValueError(f"Unknown optimizer {name!r}; expected 'adamw' or 'muon'")

    muon_parameters, adamw_parameters = split_muon_parameters(model)
    return HybridMuon(
        muon_parameters,
        adamw_parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        momentum=muon_momentum,
        ns_steps=muon_ns_steps,
        fused=fused,
    )


__all__ = ["HybridMuon", "build_optimizer", "split_muon_parameters"]
