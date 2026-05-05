from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from yunet_train.pose import YOLOPoseDataset, build_pose_train_transforms, collate_pose_samples


def _coco8_pose_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "coco8-pose"


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_yolo_pose_dataset_loads_coco8_pose_labels() -> None:
    dataset = YOLOPoseDataset(_coco8_pose_root(), split="train")

    assert len(dataset) == 4
    sample = dataset[0]
    height, width = sample.image.shape[:2]
    assert sample.labels.dtype.name == "int64"
    assert sample.boxes.shape[1] == 4
    assert sample.keypoints.shape[1:] == (17, 3)
    assert (sample.boxes[:, 0::2] >= 0).all()
    assert (sample.boxes[:, 0::2] <= width).all()
    assert (sample.boxes[:, 1::2] >= 0).all()
    assert (sample.boxes[:, 1::2] <= height).all()
    assert set(sample.labels.tolist()) <= {0}
    assert set(sample.keypoints[..., 2].reshape(-1).tolist()) <= {0.0, 1.0, 2.0}


@pytest.mark.skipif(not _coco8_pose_root().exists(), reason="data/coco8-pose is not available")
def test_yolo_pose_dataset_collates_training_batch() -> None:
    dataset = YOLOPoseDataset(_coco8_pose_root(), split="train", transform=build_pose_train_transforms(image_size=128))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_pose_samples)
    batch = next(iter(loader))

    assert isinstance(batch.images, torch.Tensor)
    assert tuple(batch.images.shape) == (2, 3, 128, 128)
    assert len(batch.boxes) == 2
    assert len(batch.keypoints) == 2
    assert batch.keypoints[0].shape[1:] == (17, 3)
    assert batch.metas[0]["kpt_shape"] == (17, 3)
