"""Minimal native PyTorch distributed-training helpers."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and device selected from ``torchrun`` environment variables."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _raise_rank_zero_failure(
    context: DistributedContext,
    operation: str,
    error: Optional[Mapping[str, str]],
) -> None:
    """Broadcast a rank-zero failure so peers raise instead of deadlocking."""
    payload = [error if context.is_main else None]
    if context.enabled:
        dist.broadcast_object_list(payload, src=0)
    error = payload[0]
    if error is not None:
        raise RuntimeError(
            f"{operation} failed on rank 0: {error['type']}: {error['message']}"
        )


def _serialized_error(exc: Exception) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _resolve_device(requested: str, local_rank: int, distributed: bool) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested, but CUDA is unavailable")
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
        return device
    if requested == "mps":
        if distributed:
            raise RuntimeError("Multi-process MPS training is unsupported; use one MPS process")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS training requested, but MPS is unavailable")
        return torch.device("mps")
    if requested != "cpu":
        raise ValueError(f"Unknown device {requested!r}; expected auto, cuda, mps, or cpu")
    return torch.device("cpu")


def init_distributed(requested_device: str = "cuda") -> DistributedContext:
    """Initialize ``env://`` DDP when launched with more than one process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    device = _resolve_device(requested_device, local_rank, enabled)

    owns_group = False
    if enabled and not dist.is_initialized():
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        owns_group = True
    if enabled:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    return DistributedContext(rank, local_rank, world_size, device, owns_group)


def seed_process(seed: int, context: DistributedContext) -> None:
    """Give each data-parallel replica a deterministic, distinct RNG stream."""
    process_seed = seed + context.rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(process_seed)
    elif context.device.type == "mps":
        torch.mps.manual_seed(process_seed)


def make_sampler(
    dataset: Dataset,
    context: DistributedContext,
    *,
    seed: int,
    shuffle: bool = True,
) -> DistributedSampler:
    """Shard a map dataset without DistributedSampler's duplicate padding."""
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )


def cycling_batches(
    loader: DataLoader,
    sampler: DistributedSampler,
    *,
    epoch: int = 0,
    batch_in_epoch: int = 0,
) -> Iterator[Tuple[T, int, int]]:
    """Yield forever while exposing the exact map-dataset resume cursor."""
    while True:
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(batch_in_epoch):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Checkpoint data cursor exceeds the epoch length") from exc
        for batch in iterator:
            batch_in_epoch += 1
            yield batch, epoch, batch_in_epoch
        epoch += 1
        batch_in_epoch = 0


def wrap_ddp(
    module: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap the complete loss module so DDP forward/backward bookkeeping is correct."""
    if not context.enabled:
        return module
    cuda = context.device.type == "cuda"
    return DistributedDataParallel(
        module,
        device_ids=[context.local_rank] if cuda else None,
        output_device=context.local_rank if cuda else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        find_unused_parameters=find_unused_parameters,
    )


def backward_sync_context(module: nn.Module, *, sync: bool):
    """Skip redundant gradient all-reduces on accumulation microsteps."""
    if not sync and isinstance(module, DistributedDataParallel):
        return module.no_sync()
    return nullcontext()


def reduce_metrics(
    metrics: Mapping[str, torch.Tensor], context: DistributedContext
) -> Dict[str, torch.Tensor]:
    """Average scalar metrics across ranks; call only at logging cadence."""
    keys = sorted(metrics)
    values = torch.stack([metrics[key].detach().float() for key in keys])
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return dict(zip(keys, values.unbind()))


def raise_distributed_failure(
    context: DistributedContext,
    operation: str,
    *,
    error: Optional[Exception] = None,
    valid: Optional[torch.Tensor] = None,
) -> None:
    """Raise the same error on every rank before the next DDP collective.

    ``valid`` is a scalar boolean tensor, typically ``isfinite(loss)``. The cheap
    scalar all-reduce runs on every rank; Python error details are gathered only on
    failure.
    """
    if not context.enabled:
        if error is not None:
            raise error
        if valid is not None and not bool(valid):
            raise RuntimeError(f"{operation} failed validation")
        return

    local_ok = torch.ones((), dtype=torch.int32, device=context.device)
    if error is not None:
        local_ok.zero_()
    elif valid is not None:
        local_ok.copy_(valid.detach().to(device=context.device, dtype=torch.int32))
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    if bool(local_ok):
        return

    detail = _serialized_error(error) if error is not None else None
    if detail is None and valid is not None and not bool(valid):
        detail = {"type": "FloatingPointError", "message": "non-finite loss"}
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, detail)
    failed_rank, failure = next(
        (rank, item) for rank, item in enumerate(gathered) if item is not None
    )
    message = (
        f"{operation} failed on rank {failed_rank}: "
        f"{failure['type']}: {failure['message']}"
    )
    raise RuntimeError(message) from error


def _rng_state(context: DistributedContext) -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if context.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(context.device)
    elif context.device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _all_rng_states(context: DistributedContext):
    local = _rng_state(context)
    if not context.enabled:
        return [local]
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    return gathered


def restore_rng_state(checkpoint: Mapping[str, object], context: DistributedContext) -> None:
    """Restore this rank's RNG stream from an exact same-world-size checkpoint."""
    states = checkpoint.get("rng_states")
    saved_world_size = int(checkpoint.get("world_size", 1))
    if not isinstance(states, list) or saved_world_size != context.world_size:
        raise ValueError(
            "Exact resume requires RNG states from the same world size "
            f"(checkpoint={saved_world_size}, current={context.world_size})"
        )
    state = states[context.rank]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if context.device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], context.device)
    elif context.device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def save_training_checkpoint(
    path: str,
    *,
    context: DistributedContext,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Atomically save complete training state once while all ranks contribute RNG state."""
    rng_states = _all_rng_states(context)
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    error = None
    if context.is_main:
        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "step": step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "world_size": context.world_size,
                "rng_states": rng_states,
            }
            if extra:
                checkpoint.update(extra)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, temporary)
            os.replace(temporary, destination)
        except Exception as exc:
            error = _serialized_error(exc)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                if error is None:
                    error = _serialized_error(exc)
    _raise_rank_zero_failure(context, "checkpoint save", error)


def restore_training_state(
    checkpoint: Mapping[str, object],
    *,
    context: DistributedContext,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> Tuple[int, int, int]:
    """Restore optimizer, schedule, RNG, and map-dataset cursor after model loading."""
    required = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "step",
        "epoch",
        "batch_in_epoch",
        "rng_states",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is not an exact training resume; missing {sorted(missing)}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint, context)
    return int(checkpoint["step"]), int(checkpoint["epoch"]), int(
        checkpoint["batch_in_epoch"]
    )


def rank_zero_torch_cache(
    path: str, context: DistributedContext, build: Callable[[], T]
) -> T:
    """Build tensor data once, then memory-map it in single- or multi-GPU runs."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    error = None
    if context.is_main:
        try:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                value = build()
                torch.save(value, temporary)
                os.replace(temporary, destination)
                del value
        except Exception as exc:
            error = _serialized_error(exc)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                if error is None:
                    error = _serialized_error(exc)
    _raise_rank_zero_failure(context, "cache build", error)
    try:
        return torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # torch 2.0 has no mmap/weights_only keyword pair
        return torch.load(destination, map_location="cpu")


__all__ = [
    "DistributedContext",
    "backward_sync_context",
    "cycling_batches",
    "init_distributed",
    "make_sampler",
    "rank_zero_torch_cache",
    "raise_distributed_failure",
    "reduce_metrics",
    "restore_training_state",
    "save_training_checkpoint",
    "seed_process",
    "wrap_ddp",
]
