"""Tests for the live-override mechanism consumed by the training loops.

scripts/monitor.py (the /loop babysitter) writes training_state_override.json;
the trainers poll + CONSUME it via train_4090._read_override and apply lr changes
via train_4090._override_set_lr. These tests guard that:
  1. the override file is read AND deleted (consume-once — no stale re-apply),
  2. a malformed/absent file never crashes training,
  3. an lr override sticks even under a LambdaLR scheduler (initial_lr patched).

CPU-only, network-free, fast.
"""

import importlib.util
import json
import os
import sys

import torch

_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _import_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_4090", os.path.join(_SCRIPTS_DIR, "train_4090.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_override_symbols_exist():
    mod = _import_train_script()
    assert hasattr(mod, "OVERRIDE_FILE")
    assert hasattr(mod, "_read_override")
    assert hasattr(mod, "_override_set_lr")


def test_read_override_absent_returns_none(tmp_path):
    mod = _import_train_script()
    assert mod._read_override(str(tmp_path / "nope.json")) is None


def test_read_override_consumes_file(tmp_path):
    """Reading an override returns its content (minus bookkeeping) AND deletes the
    file, so the same override is never applied twice."""
    mod = _import_train_script()
    p = tmp_path / "ovr.json"
    p.write_text(json.dumps({"lr": 1e-5, "stop": True, "written_at": 123.0}))

    out = mod._read_override(str(p))
    assert out == {"lr": 1e-5, "stop": True}      # "written_at" stripped
    assert not p.exists()                          # consumed
    assert mod._read_override(str(p)) is None      # nothing left to re-apply


def test_read_override_malformed_returns_none(tmp_path):
    """A half-written / invalid file must never crash the training loop."""
    mod = _import_train_script()
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json")
    assert mod._read_override(str(p)) is None


def test_override_set_lr_sticks_under_scheduler():
    """_override_set_lr must patch initial_lr so a LambdaLR scheduler recomputes
    from the new base next step instead of clobbering the override."""
    mod = _import_train_script()
    param = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)

    mod._override_set_lr(opt, 5e-4, sched)
    assert opt.param_groups[0]["lr"] == 5e-4
    assert opt.param_groups[0]["initial_lr"] == 5e-4
    assert sched.base_lrs == [5e-4]

    # The next scheduler step (lambda=1.0) must keep the overridden base, not snap
    # back to the original 1e-3.
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == 5e-4

    # Without a scheduler (the distillation path), the raw lr is set directly.
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    mod._override_set_lr(opt2, 2e-4)
    assert opt2.param_groups[0]["lr"] == 2e-4
