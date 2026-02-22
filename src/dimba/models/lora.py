"""LoRA utilities for DIMBA models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "in_proj",
    "out_proj",
    "x_proj",
    "dt_proj",
    "b_proj",
    "c_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "time_proj",
)


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` layer with a low-rank LoRA adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects base_layer to be an nn.Linear instance.")
        if r <= 0:
            raise ValueError("LoRA rank `r` must be > 0.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("LoRA dropout must be in [0.0, 1.0).")

        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        self.reset_parameters()

        # Base layer stays frozen for parameter-efficient adaptation.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.merged = False

    @property
    def weight(self) -> torch.Tensor:
        """Expose base linear weight for compatibility."""
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Expose base linear bias for compatibility."""
        return self.base_layer.bias

    def reset_parameters(self) -> None:
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base linear transform plus LoRA update."""
        output = self.base_layer(x)
        if self.merged:
            return output

        lora_update = F.linear(self.lora_dropout(x), self.lora_A)
        lora_update = F.linear(lora_update, self.lora_B)
        return output + lora_update * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Merge LoRA update into base linear weight."""
        if self.merged:
            return

        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        delta_w = delta_w.to(dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)
        self.base_layer.weight.add_(delta_w)
        self.merged = True


def _normalize_target_modules(target_modules: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if target_modules is None:
        return DEFAULT_LORA_TARGET_MODULES
    if isinstance(target_modules, str):
        return (target_modules,)
    return tuple(target_modules)


def _module_name_matches(module_name: str, target_modules: Sequence[str]) -> bool:
    full_name = module_name.lower()
    leaf_name = module_name.rsplit(".", 1)[-1].lower()

    for target in target_modules:
        target_norm = target.strip().lower()
        if not target_norm:
            continue

        if "." in target_norm:
            if full_name == target_norm or full_name.endswith(f".{target_norm}"):
                return True
        elif leaf_name == target_norm or leaf_name.endswith(target_norm):
            return True

    return False


def _iter_lora_modules(model: nn.Module) -> Iterable[Tuple[str, LoRALinear]]:
    ddp_cls = getattr(nn.parallel, "DistributedDataParallel", None)
    if isinstance(model, nn.DataParallel) or (ddp_cls is not None and isinstance(model, ddp_cls)):
        model = model.module

    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module_name, module


def _set_only_lora_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for _, module in _iter_lora_modules(model):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True


def inject_lora_to_model(
    model: nn.Module,
    target_modules: Optional[Sequence[str]] = None,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> List[str]:
    """Inject ``LoRALinear`` wrappers into matching ``nn.Linear`` modules.

    The model is switched to parameter-efficient mode after injection:
    all non-LoRA parameters are frozen.

    Returns:
        List of module names that were wrapped.
    """
    targets = _normalize_target_modules(target_modules)
    wrapped_modules: List[str] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not _module_name_matches(module_name, targets):
            continue

        if "." in module_name:
            parent_name, child_name = module_name.rsplit(".", 1)
        else:
            parent_name, child_name = "", module_name
        parent_module = model.get_submodule(parent_name) if parent_name else model
        setattr(parent_module, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        wrapped_modules.append(module_name)

    if not any(True for _ in _iter_lora_modules(model)):
        raise ValueError(
            "No LoRA modules found after injection. Check `target_modules` against model module names."
        )
    _set_only_lora_trainable(model)
    return wrapped_modules


def merge_lora_weights(model: nn.Module) -> List[str]:
    """Merge all injected LoRA weights into their base linear layers."""
    merged_modules: List[str] = []
    for module_name, module in _iter_lora_modules(model):
        module.merge()
        merged_modules.append(module_name)
    return merged_modules


def save_lora_weights(model: nn.Module, path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Save LoRA-only parameters to ``path`` and return the saved state dict."""
    lora_state: Dict[str, torch.Tensor] = {}

    for module_name, module in _iter_lora_modules(model):
        lora_state[f"{module_name}.lora_A"] = module.lora_A.detach().cpu()
        lora_state[f"{module_name}.lora_B"] = module.lora_B.detach().cpu()
        lora_state[f"{module_name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
        lora_state[f"{module_name}.r"] = torch.tensor(module.r, dtype=torch.int64)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state_dict": lora_state}, path)
    return lora_state


def _resolve_state_module_name(
    model_module_name: str,
    state_modules: set[str],
    used_state_modules: set[str],
) -> Optional[str]:
    """Resolve a model LoRA module name to a checkpoint module name.

    This handles common DataParallel/DDP prefix differences, e.g.
    ``module.foo.bar`` <-> ``foo.bar``.
    """

    candidates: List[str] = [model_module_name]

    if model_module_name.startswith("module."):
        stripped = model_module_name[len("module.") :]
        candidates.append(stripped)
        while stripped.startswith("module."):
            stripped = stripped[len("module.") :]
            candidates.append(stripped)
    else:
        prefixed = f"module.{model_module_name}"
        candidates.append(prefixed)
        prefixed = f"module.{prefixed}"
        candidates.append(prefixed)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in state_modules and candidate not in used_state_modules:
            return candidate

    return None


def load_lora_weights(
    model: nn.Module,
    path: Union[str, Path],
    strict: bool = True,
) -> Dict[str, List[str]]:
    """Load LoRA-only parameters from ``path`` into an injected model.

    Returns:
        Report with keys: ``loaded``, ``missing``, ``unexpected``, ``mismatched``.
    """
    checkpoint = torch.load(Path(path), map_location="cpu")
    if isinstance(checkpoint, dict) and "lora_state_dict" in checkpoint:
        lora_state = checkpoint["lora_state_dict"]
    elif isinstance(checkpoint, dict):
        lora_state = checkpoint
    else:
        raise TypeError("LoRA checkpoint must be a dict or include key 'lora_state_dict'.")

    lora_modules = dict(_iter_lora_modules(model))
    state_modules = {
        key.rsplit(".", 1)[0]
        for key in lora_state.keys()
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }

    loaded: List[str] = []
    missing: List[str] = []
    mismatched: List[str] = []
    matched_state_modules: set[str] = set()

    for module_name, module in lora_modules.items():
        state_module_name = _resolve_state_module_name(module_name, state_modules, matched_state_modules)
        if state_module_name is None:
            missing.append(module_name)
            continue
        matched_state_modules.add(state_module_name)

        key_a = f"{state_module_name}.lora_A"
        key_b = f"{state_module_name}.lora_B"
        if key_a not in lora_state or key_b not in lora_state:
            missing.append(module_name)
            continue

        tensor_a = lora_state[key_a]
        tensor_b = lora_state[key_b]
        if tuple(tensor_a.shape) != tuple(module.lora_A.shape):
            mismatched.append(module_name)
            continue
        if tuple(tensor_b.shape) != tuple(module.lora_B.shape):
            mismatched.append(module_name)
            continue

        r_key = f"{state_module_name}.r"
        if r_key in lora_state:
            loaded_r = int(torch.as_tensor(lora_state[r_key]).item())
            if loaded_r != module.r:
                mismatched.append(module_name)
                continue

        module.lora_A.data.copy_(tensor_a.to(device=module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(tensor_b.to(device=module.lora_B.device, dtype=module.lora_B.dtype))

        alpha_key = f"{state_module_name}.alpha"
        if alpha_key in lora_state:
            module.alpha = float(torch.as_tensor(lora_state[alpha_key]).item())
            module.scaling = module.alpha / float(module.r)

        module.merged = False
        loaded.append(module_name)

    unexpected = sorted(module_name for module_name in state_modules if module_name not in matched_state_modules)

    report = {
        "loaded": sorted(set(loaded)),
        "missing": sorted(set(missing)),
        "unexpected": unexpected,
        "mismatched": sorted(set(mismatched)),
    }
    if strict and (report["missing"] or report["unexpected"] or report["mismatched"]):
        raise RuntimeError(
            "Failed to strictly load LoRA weights. "
            f"missing={report['missing']}, "
            f"unexpected={report['unexpected']}, "
            f"mismatched={report['mismatched']}"
        )

    if lora_modules:
        _set_only_lora_trainable(model)
    return report
