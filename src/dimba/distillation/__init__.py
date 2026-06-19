"""Cross-architecture knowledge distillation subpackage for DIMBA.

Provides a complete pipeline for distilling a pretrained HuggingFace Transformer
teacher into DIMBA's bidirectional Mamba-2 diffusion student, including:

- :class:`TeacherWrapper` / :class:`TeacherOutputs`: unified teacher interface.
- :class:`Projector`, :class:`HeadAligner`, :class:`LayerMap`: alignment utilities.
- :func:`stage1_matrix_loss`, :func:`stage2_hidden_loss`, :func:`stage3_kd_loss`:
  per-stage distillation losses.
- :func:`build_student_from_teacher`: weight surgery to construct a matched student.
- :func:`principled_init_from_teacher`: attention-weight warm-start for the student.
- :class:`DistillationTrainer` / :class:`DistillationConfig`: training orchestration.
"""

from .teacher import TeacherOutputs, TeacherWrapper
from .projectors import HeadAligner, LayerMap, Projector
from .losses import stage1_matrix_loss, stage2_hidden_loss, stage3_kd_loss
from .surgery import build_student_from_teacher
from .init import principled_init_from_teacher
from .trainer import DistillationConfig, DistillationTrainer

__all__ = [
    # teacher
    "TeacherWrapper",
    "TeacherOutputs",
    # projectors
    "Projector",
    "HeadAligner",
    "LayerMap",
    # losses
    "stage1_matrix_loss",
    "stage2_hidden_loss",
    "stage3_kd_loss",
    # surgery
    "build_student_from_teacher",
    # init
    "principled_init_from_teacher",
    # trainer
    "DistillationTrainer",
    "DistillationConfig",
]
