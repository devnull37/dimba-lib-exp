"""Smoke tests for the _FineWebStreaming dataset and _make_fineweb_streaming_loader.

These tests verify:
1. The streaming loader yields correctly-shaped [B, seq_len] int64 batches.
2. Worker sharding is disjoint — two workers do NOT yield identical first tokens.
3. scripts/train_4090.py still imports cleanly (no syntax/import errors).

Network: load_dataset(..., streaming=True) makes HTTP requests. The tests marked
@pytest.mark.slow download a small slice of FineWeb.  Run manually with:
    python -m pytest tests/test_fineweb_streaming.py -v -m slow

The sharding test uses unittest.mock to patch get_worker_info so it works without
spawning real DataLoader workers (which would fail to re-import train_4090 in
a fresh process since the module is loaded dynamically from the scripts dir).
"""

import importlib.util
import os
import sys
import types
from unittest.mock import patch, MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Add scripts dir so train_4090 is importable by name (needed for multiprocessing pickle)
_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _import_train_script():
    """Import scripts/train_4090.py as a module (handles the sys.path insertion)."""
    spec = importlib.util.spec_from_file_location(
        "train_4090",
        os.path.join(_SCRIPTS_DIR, "train_4090.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test: clean import
# ---------------------------------------------------------------------------


def test_train_script_imports_cleanly():
    """scripts/train_4090.py must import without errors and expose new symbols."""
    mod = _import_train_script()
    assert hasattr(mod, "_FineWebStreaming"), (
        "_FineWebStreaming class not found in train_4090"
    )
    assert hasattr(mod, "_make_fineweb_streaming_loader"), (
        "_make_fineweb_streaming_loader function not found in train_4090"
    )
    # Legacy cached path must still exist (Stage 1+2 align uses it).
    assert hasattr(mod, "_build_fineweb_stream"), (
        "_build_fineweb_stream (cached align path) must not be deleted"
    )
    assert hasattr(mod, "_PackedChunks"), (
        "_PackedChunks must not be deleted"
    )
    assert hasattr(mod, "_make_fineweb_loader"), (
        "_make_fineweb_loader must not be deleted"
    )


# ---------------------------------------------------------------------------
# Test: loader shape  (requires network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_streaming_loader_yields_correct_shape():
    """Streaming loader must yield [B, seq_len] int64 tensors of real tokens."""
    from transformers import AutoTokenizer

    mod = _import_train_script()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    seq_len = 64
    batch_size = 4
    loader = mod._make_fineweb_streaming_loader(
        tokenizer, seq_len=seq_len, batch_size=batch_size, num_workers=0
    )

    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor), f"expected Tensor, got {type(batch)}"
    assert batch.dtype == torch.long, f"expected int64, got {batch.dtype}"
    assert batch.shape == (batch_size, seq_len), (
        f"expected ({batch_size}, {seq_len}), got {batch.shape}"
    )
    vocab_size = tokenizer.vocab_size
    assert batch.min().item() >= 0, "token ids must be non-negative"
    assert batch.max().item() < vocab_size, (
        f"token id {batch.max().item()} out of vocab range {vocab_size}"
    )
    print(f"\n[shape test] batch shape: {batch.shape}, dtype: {batch.dtype}, "
          f"min_id: {batch.min().item()}, max_id: {batch.max().item()}")


# ---------------------------------------------------------------------------
# Test: worker sharding — disjoint shard per worker  (requires network)
#
# Strategy: instead of spawning real DataLoader workers (which would fail to
# re-import train_4090 loaded via spec_from_file_location), we patch
# torch.utils.data.get_worker_info directly inside __iter__ calls.
# This exercises the exact same branching code path as production workers.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_worker_sharding_no_duplication():
    """Worker shard calls must produce DISJOINT documents (no data duplication).

    We simulate worker_id=0 and worker_id=1 by mocking get_worker_info inside
    _FineWebStreaming.__iter__ and collecting the first seq_len tokens each would
    see. They must NOT be identical.
    """
    from transformers import AutoTokenizer

    mod = _import_train_script()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    seq_len = 64
    ds = mod._FineWebStreaming(tokenizer, seq_len)

    def _first_chunk(worker_id: int, num_workers: int = 2) -> torch.Tensor:
        """Run __iter__ pretending to be DataLoader worker `worker_id`."""
        fake_info = MagicMock()
        fake_info.id = worker_id
        fake_info.num_workers = num_workers
        # Patch get_worker_info in the module where _FineWebStreaming calls it.
        with patch("torch.utils.data.get_worker_info", return_value=fake_info):
            return next(iter(ds))

    chunk_w0 = _first_chunk(worker_id=0)
    chunk_w1 = _first_chunk(worker_id=1)

    print(f"\n[sharding test] worker-0 first chunk[:8]: {chunk_w0[:8].tolist()}")
    print(f"[sharding test] worker-1 first chunk[:8]: {chunk_w1[:8].tolist()}")
    are_equal = torch.equal(chunk_w0, chunk_w1)
    print(f"[sharding test] identical: {are_equal}  (must be False for correct sharding)")

    assert not are_equal, (
        "SHARDING BUG: worker-0 and worker-1 produced identical first chunks. "
        "Both workers read the same shard — data would be duplicated in training."
    )
