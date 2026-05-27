"""Import smoke tests for the ``dimba`` package.

These tests import the top-level package and every public submodule to catch
breakage (syntax errors, broken intra-package imports, renamed symbols) early.

Some submodules have *hard* dependencies on optional third-party libraries that
are not installed in the CPU-only CI environment (for example ``pytorch_lightning``
for :mod:`dimba.training` and ``datasets`` for :mod:`dimba.data`). When a submodule
fails to import *solely* because such an optional dependency is missing, the test
is skipped rather than failed -- a genuine break inside ``dimba`` still fails the
test because the missing module name would belong to ``dimba`` itself.
"""

import importlib

import pytest

# Public submodules that should always import on a bare torch-only install.
CORE_SUBMODULES = [
    "dimba",
    "dimba.models",
    "dimba.models.diffusion",
    "dimba.models.denoiser",
    "dimba.models.embeddings",
    "dimba.models.simple_mamba",
    "dimba.models.vae",
    "dimba.models.lora",
    "dimba.models.quantization",
    "dimba.diffusion",
    "dimba.diffusion.schedules",
    "dimba.diffusion.sampling",
    "dimba.tokenizers",
    "dimba.tokenizers.base",
    "dimba.tokenizers.simple",
    "dimba.tokenizers.bpe",
    "dimba.evaluation",
    "dimba.evaluation.metrics",
    "dimba.utils",
    "dimba.utils.checkpointing",
]

# Submodules that may require optional third-party packages to import.
# These are imported best-effort and skipped if only the optional dep is missing.
OPTIONAL_SUBMODULES = [
    "dimba.training",
    "dimba.training.trainer",
    "dimba.data",
    "dimba.data.dataset",
    "dimba.data.finetuning",
]


def _import_or_skip_on_optional_dep(module_name: str) -> None:
    """Import ``module_name``; skip if a *non-dimba* dependency is missing.

    Args:
        module_name: Fully-qualified module path to import.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", "") or ""
        # If the missing module is part of dimba itself, this is a real break.
        if missing.startswith("dimba"):
            raise
        pytest.skip(f"Skipping {module_name}: optional dependency missing ({exc}).")


@pytest.mark.parametrize("module_name", CORE_SUBMODULES)
def test_import_core_submodule(module_name: str) -> None:
    """Every core submodule must import without error."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", OPTIONAL_SUBMODULES)
def test_import_optional_submodule(module_name: str) -> None:
    """Optional submodules import, or are skipped if an optional dep is missing."""
    _import_or_skip_on_optional_dep(module_name)


def test_package_exposes_public_api() -> None:
    """The top-level package exposes its documented public symbols."""
    import dimba

    for name in [
        "DIMBA",
        "CosineNoiseSchedule",
        "sample_from_model",
        "DDIMSampler",
        "BaseTokenizer",
        "SimpleCharacterTokenizer",
        "BPETokenizer",
    ]:
        assert hasattr(dimba, name), f"dimba is missing public symbol: {name}"


def test_all_listed_symbols_importable() -> None:
    """Everything in ``dimba.__all__`` is actually importable from the package."""
    import dimba

    for name in dimba.__all__:
        assert hasattr(dimba, name), f"dimba.__all__ lists missing symbol: {name}"
