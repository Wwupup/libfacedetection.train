from __future__ import annotations

import numpy as np
import torch

from yunet_train.pose import COCO17_FLIP_IDX, Pad, PoseSample, RandomHorizontalFlip, Resize, ToTensor, collate_pose_samples


def _sample() -> PoseSample:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    boxes = np.array([[2, 1, 12, 9]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, :, 0] = np.arange(17, dtype=np.float32)
    keypoints[0, :, 1] = 1
    keypoints[0, :, 2] = 2
    keypoints[0, 16] = [21, 5, 2]
    return PoseSample(
        image=image,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        filename="synthetic.jpg",
        original_shape=image.shape,
        image_shape=image.shape,
        pad_shape=image.shape,
        kpt_shape=(17, 3),
    )


def test_pose_resize_scales_boxes_and_keypoints_and_masks_outside_points() -> None:
    sample = Resize((40, 20), keep_ratio=False)(_sample())

    assert sample.image.shape == (20, 40, 3)
    np.testing.assert_allclose(sample.boxes, [[4, 2, 24, 18]])
    np.testing.assert_allclose(sample.keypoints[0, 0, :2], [0, 2])
    np.testing.assert_allclose(sample.keypoints[0, 15, :2], [30, 2])
    np.testing.assert_allclose(sample.keypoints[0, 16, :2], [40, 10])
    assert sample.keypoints[0, 16, 2] == 0


def test_pose_horizontal_flip_uses_coco17_left_right_mapping() -> None:
    sample = RandomHorizontalFlip(1.0)(_sample())
    expected_x = 20 - _sample().keypoints[0, list(COCO17_FLIP_IDX), 0]

    np.testing.assert_allclose(sample.boxes, [[8, 1, 18, 9]])
    np.testing.assert_allclose(sample.keypoints[0, :, 0], expected_x)
    assert sample.keypoints[0, 1, 0] == 18
    assert sample.keypoints[0, 2, 0] == 19
    assert sample.flip is True
    assert sample.flip_direction == "horizontal"


def test_pose_to_tensor_and_collate_samples() -> None:
    sample = ToTensor()(Pad(size=(24, 16))(_sample()))
    batch = collate_pose_samples([sample, sample])

    assert isinstance(batch.images, torch.Tensor)
    assert tuple(batch.images.shape) == (2, 3, 16, 24)
    assert batch.boxes[0].shape == (1, 4)
    assert batch.keypoints[0].shape == (1, 17, 3)
