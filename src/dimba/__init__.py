"""DIMBA: Diffusion-based Mamba for non-autoregressive text generation."""

__version__ = "0.1.0"
__author__ = "Faris Allafi"

from .models.diffusion import DIMBA
from .diffusion.schedules import CosineNoiseSchedule, LinearNoiseSchedule
from .diffusion.sampling import sample_from_model, DDIMSampler

__all__ = [
    "DIMBA",
    "CosineNoiseSchedule",
    "LinearNoiseSchedule",
    "sample_from_model",
    "DDIMSampler",
]
