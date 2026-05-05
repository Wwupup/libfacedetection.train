from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch.utils.data import DataLoader

from .types import PoseBatch


class PoseCriterion(Protocol):
    def __call__(
        self,
        preds: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        *,
        boxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        keypoints: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...


class LRScheduler(Protocol):
    def step(self, *, epoch: int) -> list[float]:
        ...


@dataclass(frozen=True)
class PoseTrainStats:
    loss: float
    loss_cls: float
    loss_bbox: float
    loss_obj: float
    loss_kpt: float
    loss_kpt_vis: float
    steps: int


def move_pose_batch_to_device(batch: PoseBatch, device: torch.device | str) -> PoseBatch:
    return PoseBatch(
        images=batch.images.to(device, non_blocking=True),
        boxes=[boxes.to(device, non_blocking=True) for boxes in batch.boxes],
        labels=[labels.to(device, non_blocking=True) for labels in batch.labels],
        keypoints=[keypoints.to(device, non_blocking=True) for keypoints in batch.keypoints],
        metas=batch.metas,
    )


def train_pose_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: PoseCriterion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int = 1,
    lr_scheduler: LRScheduler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    logger: Callable[[str], None] | None = None,
) -> PoseTrainStats:
    model.train()
    totals = _empty_totals()
    steps = 0

    for batch in data_loader:
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch)
        batch = move_pose_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(
            model(batch.images),
            boxes=batch.boxes,
            labels=batch.labels,
            keypoints=batch.keypoints,
        )
        loss = sum(losses.values())
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        steps += 1
        _accumulate(totals, losses, loss)
        if log_interval > 0 and logger is not None and (steps == 1 or steps % log_interval == 0):
            logger(_format_step("train", epoch, steps, len(data_loader), totals))

    if steps == 0:
        raise ValueError("data_loader yielded no batches")
    return _stats_from_totals(totals, steps)


@torch.no_grad()
def evaluate_pose_loss(
    *,
    model: torch.nn.Module,
    criterion: PoseCriterion,
    data_loader: DataLoader,
    device: torch.device | str,
) -> PoseTrainStats:
    model.eval()
    totals = _empty_totals()
    steps = 0
    for batch in data_loader:
        batch = move_pose_batch_to_device(batch, device)
        losses = criterion(
            model(batch.images),
            boxes=batch.boxes,
            labels=batch.labels,
            keypoints=batch.keypoints,
        )
        loss = sum(losses.values())
        steps += 1
        _accumulate(totals, losses, loss)

    if steps == 0:
        raise ValueError("data_loader yielded no batches")
    return _stats_from_totals(totals, steps)


def _empty_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "loss_cls": 0.0,
        "loss_bbox": 0.0,
        "loss_obj": 0.0,
        "loss_kpt": 0.0,
        "loss_kpt_vis": 0.0,
    }


def _accumulate(totals: dict[str, float], losses: dict[str, torch.Tensor], loss: torch.Tensor) -> None:
    totals["loss"] += float(loss.detach().cpu())
    for name in ("loss_cls", "loss_bbox", "loss_obj", "loss_kpt", "loss_kpt_vis"):
        totals[name] += float(losses[name].detach().cpu())


def _stats_from_totals(totals: dict[str, float], steps: int) -> PoseTrainStats:
    return PoseTrainStats(
        loss=totals["loss"] / steps,
        loss_cls=totals["loss_cls"] / steps,
        loss_bbox=totals["loss_bbox"] / steps,
        loss_obj=totals["loss_obj"] / steps,
        loss_kpt=totals["loss_kpt"] / steps,
        loss_kpt_vis=totals["loss_kpt_vis"] / steps,
        steps=steps,
    )


def _format_step(prefix: str, epoch: int, steps: int, total_steps: int, totals: dict[str, float]) -> str:
    return (
        f"{prefix} epoch={epoch} "
        f"step={steps}/{total_steps} "
        f"loss={totals['loss'] / steps:.6f} "
        f"cls={totals['loss_cls'] / steps:.6f} "
        f"bbox={totals['loss_bbox'] / steps:.6f} "
        f"obj={totals['loss_obj'] / steps:.6f} "
        f"kpt={totals['loss_kpt'] / steps:.6f} "
        f"kpt_vis={totals['loss_kpt_vis'] / steps:.6f}"
    )
