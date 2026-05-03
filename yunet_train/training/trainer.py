from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch.utils.data import DataLoader

from yunet_train.data import FaceBatch


class Criterion(Protocol):
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
class TrainStats:
    loss: float
    loss_cls: float
    loss_bbox: float
    loss_obj: float
    loss_kps: float
    steps: int


def move_batch_to_device(batch: FaceBatch, device: torch.device | str) -> FaceBatch:
    return FaceBatch(
        images=batch.images.to(device, non_blocking=True),
        boxes=[boxes.to(device, non_blocking=True) for boxes in batch.boxes],
        labels=[labels.to(device, non_blocking=True) for labels in batch.labels],
        keypoints=[keypoints.to(device, non_blocking=True) for keypoints in batch.keypoints],
        ignored_boxes=[boxes.to(device, non_blocking=True) for boxes in batch.ignored_boxes],
        ignored_labels=[labels.to(device, non_blocking=True) for labels in batch.ignored_labels],
        metas=batch.metas,
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: Criterion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int = 1,
    lr_scheduler: LRScheduler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    logger: Callable[[str], None] | None = None,
    progress_suffix: Callable[[int], str] | None = None,
) -> TrainStats:
    model.train()
    totals = {
        "loss": 0.0,
        "loss_cls": 0.0,
        "loss_bbox": 0.0,
        "loss_obj": 0.0,
        "loss_kps": 0.0,
    }
    steps = 0

    for batch in data_loader:
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch.images)
        losses = criterion(
            preds,
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
        totals["loss"] += float(loss.detach().cpu())
        for name in ("loss_cls", "loss_bbox", "loss_obj", "loss_kps"):
            totals[name] += float(losses[name].detach().cpu())
        if log_interval > 0 and logger is not None and (steps == 1 or steps % log_interval == 0):
            suffix = f" {progress_suffix(steps)}" if progress_suffix is not None else ""
            logger(
                f"epoch={epoch} "
                f"step={steps}/{len(data_loader)} "
                f"loss={totals['loss'] / steps:.6f} "
                f"cls={totals['loss_cls'] / steps:.6f} "
                f"bbox={totals['loss_bbox'] / steps:.6f} "
                f"obj={totals['loss_obj'] / steps:.6f} "
                f"kps={totals['loss_kps'] / steps:.6f}"
                f"{suffix}"
            )

    if steps == 0:
        raise ValueError("data_loader yielded no batches")

    return TrainStats(
        loss=totals["loss"] / steps,
        loss_cls=totals["loss_cls"] / steps,
        loss_bbox=totals["loss_bbox"] / steps,
        loss_obj=totals["loss_obj"] / steps,
        loss_kps=totals["loss_kps"] / steps,
        steps=steps,
    )


@torch.no_grad()
def evaluate_loss(
    *,
    model: torch.nn.Module,
    criterion: Criterion,
    data_loader: DataLoader,
    device: torch.device | str,
) -> TrainStats:
    model.eval()
    totals = {
        "loss": 0.0,
        "loss_cls": 0.0,
        "loss_bbox": 0.0,
        "loss_obj": 0.0,
        "loss_kps": 0.0,
    }
    steps = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        preds = model(batch.images)
        losses = criterion(
            preds,
            boxes=batch.boxes,
            labels=batch.labels,
            keypoints=batch.keypoints,
        )
        loss = sum(losses.values())

        steps += 1
        totals["loss"] += float(loss.detach().cpu())
        for name in ("loss_cls", "loss_bbox", "loss_obj", "loss_kps"):
            totals[name] += float(losses[name].detach().cpu())

    if steps == 0:
        raise ValueError("data_loader yielded no batches")

    return TrainStats(
        loss=totals["loss"] / steps,
        loss_cls=totals["loss_cls"] / steps,
        loss_bbox=totals["loss_bbox"] / steps,
        loss_obj=totals["loss_obj"] / steps,
        loss_kps=totals["loss_kps"] / steps,
        steps=steps,
    )
