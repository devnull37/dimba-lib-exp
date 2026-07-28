"""Import smoke tests for the public ``dimba`` package."""

import importlib

CORE_SUBMODULES = [
    "dimba",
    "dimba.models",
    "dimba.models.diffusion",
    "dimba.models.denoiser",
    "dimba.models.embeddings",
    "dimba.models.parallel_scan",
    "dimba.models.simple_mamba",
    "dimba.models.torch_mamba2",
    "dimba.models.vae",
    "dimba.backends",
    "dimba.backends.mlx",
    "dimba.diffusion",
    "dimba.diffusion.corruption",
    "dimba.diffusion.masked_sampling",
    "dimba.diffusion.rerank",
    "dimba.diffusion.schedules",
    "dimba.diffusion.sampling",
    "dimba.inference",
    "dimba.inference.block_cot",
    "dimba.tokenizers",
    "dimba.tokenizers.base",
    "dimba.tokenizers.simple",
    "dimba.tokenizers.bpe",
    "dimba.evaluation",
    "dimba.evaluation.metrics",
    "dimba.utils",
    "dimba.utils.compile",
    "dimba.utils.cuda_graphs",
]

def test_import_core_submodules() -> None:
    """Every core submodule must import without error."""
    for module_name in CORE_SUBMODULES:
        importlib.import_module(module_name)


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
