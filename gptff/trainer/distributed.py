from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from gptff.trainer._utils import normalize_distributed_mode


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str | None = None
    owns_process_group: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def disabled(cls, *, device: str | None = None) -> DistributedContext:
        return cls(enabled=False, device=device)


def initialize_distributed(mode: str | bool, *, requested_device: str) -> DistributedContext:
    normalized = normalize_distributed_mode(mode)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_enable = normalized == "true" or (normalized == "auto" and world_size > 1)
    if not should_enable:
        return DistributedContext.disabled(device=requested_device)
    if world_size <= 1:
        raise RuntimeError(
            "training.distributed=true requires a torchrun environment with WORLD_SIZE > 1."
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requested CUDA, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = requested_device
        backend = "gloo"

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend=backend)
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        owns_process_group=owns_process_group,
    )


def wrap_distributed_model(
    model: torch.nn.Module,
    context: DistributedContext,
) -> torch.nn.Module:
    if not context.enabled:
        return model
    device = torch.device(context.device or "cpu")
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=False,
        )
    return DistributedDataParallel(model, find_unused_parameters=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        device = torch.device(context.device or "cpu")
        if device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def all_reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_max(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and context.owns_process_group and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except RuntimeError as exc:
            print(
                f"Warning: failed to destroy distributed process group: {exc}",
                file=sys.stderr,
                flush=True,
            )
