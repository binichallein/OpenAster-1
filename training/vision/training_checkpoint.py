import math
import os
import random
from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import Sampler


TRAINING_STATE_NAME = "training_state.pt"
RANK_STATE_TEMPLATE = "training_state_rank{rank}.pt"


def _atomic_torch_save(obj: Any, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda_all" in state:
        cuda_states = [
            cuda_state.cpu() if isinstance(cuda_state, torch.Tensor) else cuda_state
            for cuda_state in state["cuda_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


class ResumableDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        start_index: int = 0,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.start_index = start_index

        dataset_len = len(self.dataset)
        if self.drop_last and dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil((dataset_len - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(dataset_len / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_index(self, start_index: int) -> None:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                repeats = math.ceil(padding_size / len(indices))
                indices += (indices * repeats)[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(self.num_samples - self.start_index, 0)


def save_training_checkpoint_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    running_loss: float,
    rank: int,
    world_size: int,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    rng_state = capture_rng_state()

    rank_state = {
        "rank": rank,
        "world_size": world_size,
        "rng_state": rng_state,
    }
    _atomic_torch_save(rank_state, os.path.join(checkpoint_dir, RANK_STATE_TEMPLATE.format(rank=rank)))

    if rank != 0:
        return

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "running_loss": running_loss,
        "world_size": world_size,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "extra": extra or {},
        "rng_state": rng_state,
    }
    _atomic_torch_save(state, os.path.join(checkpoint_dir, TRAINING_STATE_NAME))


def load_training_checkpoint_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    rank: int,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_dir = os.fspath(checkpoint_dir)
    state = torch.load(
        os.path.join(checkpoint_dir, TRAINING_STATE_NAME),
        map_location=map_location,
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    rank_state_path = os.path.join(checkpoint_dir, RANK_STATE_TEMPLATE.format(rank=rank))
    if os.path.exists(rank_state_path):
        rank_state = torch.load(rank_state_path, map_location=map_location, weights_only=False)
        restore_rng_state(rank_state["rng_state"])
    else:
        restore_rng_state(state.get("rng_state", {}))
    return state
