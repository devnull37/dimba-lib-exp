"""Projector modules and layer mapping utilities for cross-architecture knowledge distillation.

Provides:
- Projector: MLP that projects student hidden states into teacher's dimensionality.
- HeadAligner: Learned linear mixing of student attention heads to match teacher head count.
- LayerMap: Mapping between student and teacher layer indices for alignment losses.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Projects student hidden states into the teacher's hidden dimension.

    When ``d_in == d_out`` and ``depth == 1``, the single linear layer is initialized
    to the identity matrix so the projector starts as a no-op.

    Args:
        d_in: Input (student) hidden dimension.
        d_out: Output (teacher) hidden dimension.
        depth: Number of linear layers. Must be >= 1.
            depth=1 -> single linear; depth>1 -> (depth-1) hidden layers with GELU activations
            followed by a final linear layer, all of width d_out.
        bias: Whether to include bias terms in the linear layers.
    """

    def __init__(self, d_in: int, d_out: int, depth: int = 1, bias: bool = True) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: List[nn.Module] = []
        if depth == 1:
            layers.append(nn.Linear(d_in, d_out, bias=bias))
        else:
            # First layer: d_in -> d_out
            layers.append(nn.Linear(d_in, d_out, bias=bias))
            layers.append(nn.GELU())
            # Intermediate layers: d_out -> d_out
            for _ in range(depth - 2):
                layers.append(nn.Linear(d_out, d_out, bias=bias))
                layers.append(nn.GELU())
            # Final layer: d_out -> d_out
            layers.append(nn.Linear(d_out, d_out, bias=bias))

        self.net = nn.Sequential(*layers)
        self._d_in = d_in
        self._d_out = d_out
        self._depth = depth
        self._identity = d_in == d_out and depth == 1

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights; identity init when d_in==d_out and depth==1."""
        if self._identity:
            # The single linear layer: make it an identity transform.
            linear: nn.Linear = self.net[0]  # type: ignore[assignment]
            with torch.no_grad():
                nn.init.eye_(linear.weight)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
        else:
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states from student to teacher dimension.

        Args:
            x: Hidden states of shape ``[B, L, d_in]``.

        Returns:
            Projected tensor of shape ``[B, L, d_out]``.
        """
        return self.net(x)


class HeadAligner(nn.Module):
    """Learns a linear mixing over student attention heads to match the teacher's head count.

    The mixing is performed via an einsum ``'oi,bilm->bolm'`` where the weight matrix
    has shape ``[h_teacher, h_student]``.

    When ``h_student == h_teacher``, the module stores no learnable parameters and returns
    the input unchanged (identity shortcut).

    Args:
        h_student: Number of student attention heads.
        h_teacher: Number of teacher attention heads.
    """

    def __init__(self, h_student: int, h_teacher: int) -> None:
        super().__init__()
        self._h_student = h_student
        self._h_teacher = h_teacher
        self._identity = h_student == h_teacher

        if not self._identity:
            # weight[o, i]: contribution of student head i to teacher head o.
            # Init: each teacher head = mean of all student heads -> all entries = 1/h_student.
            self.weight = nn.Parameter(
                torch.full((h_teacher, h_student), fill_value=1.0 / h_student)
            )
        # When identity, no parameter is registered.

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """Mix student attention matrix heads to produce teacher-sized head dimension.

        Args:
            M: Student attention matrices of shape ``[B, Hs, L, L]``.

        Returns:
            Mixed matrices of shape ``[B, Ht, L, L]``.
        """
        if self._identity:
            return M
        return torch.einsum("oi,bilm->bolm", self.weight, M)


class LayerMap:
    """Maps student layer indices to teacher layer indices for distillation alignment.

    Supports three modes:

    - ``'uniform'``: Spread ``n_student`` picks evenly across teacher layers.
      Student layer ``i`` maps to teacher layer
      ``round(i * (n_teacher - 1) / max(1, n_student - 1))``.
    - ``'last'``: All student layers map to the last teacher layer (index ``n_teacher - 1``).
    - ``'explicit'``: Use a caller-provided list of ``(student_idx, teacher_idx)`` pairs.

    Args:
        n_teacher: Total number of teacher layers.
        n_student: Total number of student layers.
        mode: One of ``'uniform'``, ``'last'``, or ``'explicit'``.
        explicit: Required when ``mode='explicit'``; a list of ``(student_idx, teacher_idx)``
            tuples.

    Raises:
        ValueError: On invalid mode or missing/invalid explicit list.
    """

    def __init__(
        self,
        n_teacher: int,
        n_student: int,
        mode: str = "uniform",
        explicit: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        if mode not in ("uniform", "last", "explicit"):
            raise ValueError(f"mode must be 'uniform', 'last', or 'explicit', got '{mode}'")
        if mode == "explicit" and explicit is None:
            raise ValueError("explicit list must be provided when mode='explicit'")

        self._n_teacher = n_teacher
        self._n_student = n_student
        self._mode = mode
        self._pairs: List[Tuple[int, int]] = self._build_pairs(explicit)

    def _build_pairs(
        self, explicit: Optional[List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """Construct the list of (student_idx, teacher_idx) pairs."""
        if self._mode == "explicit":
            assert explicit is not None
            return list(explicit)
        if self._mode == "last":
            return [(si, self._n_teacher - 1) for si in range(self._n_student)]
        # 'uniform'
        pairs: List[Tuple[int, int]] = []
        denom = max(1, self._n_student - 1)
        for si in range(self._n_student):
            ti = round(si * (self._n_teacher - 1) / denom)
            # Clamp to valid range just in case of rounding edge cases.
            ti = max(0, min(ti, self._n_teacher - 1))
            pairs.append((si, ti))
        return pairs

    def pairs(self) -> List[Tuple[int, int]]:
        """Return the list of ``(student_idx, teacher_idx)`` pairs.

        Returns:
            A list of tuples ``(student_layer_index, teacher_layer_index)``.
        """
        return list(self._pairs)

    @property
    def n_teacher(self) -> int:
        """Number of teacher layers."""
        return self._n_teacher

    @property
    def n_student(self) -> int:
        """Number of student layers."""
        return self._n_student

    @property
    def mode(self) -> str:
        """Mapping mode (``'uniform'``, ``'last'``, or ``'explicit'``)."""
        return self._mode
