from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

from .config import COCO17_FLIP_IDX
from .types import PoseSample


class Compose:
    def __init__(self, transforms: Sequence[Callable[[PoseSample], PoseSample]]):
        self.transforms = tuple(transforms)

    def __call__(self, sample: PoseSample) -> PoseSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize:
    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        keep_ratio: bool = False,
        clip_border: bool = True,
    ):
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.clip_border = clip_border

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        old_h, old_w = image.shape[:2]
        target_w, target_h = self.image_size
        if self.keep_ratio:
            scale = min(target_w / old_w, target_h / old_h)
            new_w = int(old_w * scale + 0.5)
            new_h = int(old_h * scale + 0.5)
        else:
            new_w, new_h = target_w, target_h

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        w_scale = new_w / old_w
        h_scale = new_h / old_h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        sample.image = resized
        sample.image_shape = resized.shape
        sample.pad_shape = resized.shape
        sample.scale_factor = scale_factor
        sample.boxes = _scale_boxes(_ensure_numpy_array(sample.boxes), scale_factor, resized.shape, self.clip_border)
        sample.keypoints = _scale_keypoints(
            _ensure_numpy_array(sample.keypoints),
            w_scale,
            h_scale,
            resized.shape,
            self.clip_border,
        )
        return sample


class RandomHorizontalFlip:
    def __init__(
        self,
        probability: float = 0.5,
        *,
        flip_idx: Sequence[int] = COCO17_FLIP_IDX,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("flip probability must be in [0, 1]")
        self.probability = probability
        self.flip_idx = tuple(flip_idx)

    def __call__(self, sample: PoseSample) -> PoseSample:
        if np.random.random() >= self.probability:
            sample.flip = False
            sample.flip_direction = None
            return sample

        image = _ensure_numpy_image(sample.image)
        width = image.shape[1]
        sample.image = np.flip(image, axis=1).copy()
        sample.boxes = _flip_boxes(_ensure_numpy_array(sample.boxes), width)
        sample.keypoints = _flip_keypoints(_ensure_numpy_array(sample.keypoints), width, self.flip_idx)
        sample.flip = True
        sample.flip_direction = "horizontal"
        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float], *, to_rgb: bool = False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image).astype(np.float32)
        if self.to_rgb:
            image = image[..., ::-1]
        sample.image = (image - self.mean) / self.std
        sample.image_norm = {
            "mean": self.mean,
            "std": self.std,
            "to_rgb": self.to_rgb,
        }
        return sample


class Pad:
    def __init__(
        self,
        *,
        size: tuple[int, int] | None = None,
        size_divisor: int | None = None,
        pad_value: int | float = 114,
    ):
        if (size is None) == (size_divisor is None):
            raise ValueError("exactly one of size or size_divisor must be set")
        self.size = size
        self.size_divisor = size_divisor
        self.pad_value = pad_value

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        height, width = image.shape[:2]
        if self.size is not None:
            target_w, target_h = self.size
        else:
            assert self.size_divisor is not None
            target_h = int(np.ceil(height / self.size_divisor)) * self.size_divisor
            target_w = int(np.ceil(width / self.size_divisor)) * self.size_divisor

        if target_h < height or target_w < width:
            raise ValueError(f"pad target {(target_w, target_h)} is smaller than image {(width, height)}")

        padded = np.full((target_h, target_w, image.shape[2]), self.pad_value, dtype=image.dtype)
        padded[:height, :width] = image
        sample.image = padded
        sample.pad_shape = padded.shape
        return sample


class ToTensor:
    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        sample.image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        sample.boxes = torch.from_numpy(np.ascontiguousarray(sample.boxes)).float()
        sample.labels = torch.from_numpy(np.ascontiguousarray(sample.labels)).long()
        sample.keypoints = torch.from_numpy(np.ascontiguousarray(sample.keypoints)).float()
        return sample


def build_pose_train_transforms(
    image_size: int = 640,
    *,
    flip_idx: Sequence[int] = COCO17_FLIP_IDX,
) -> Compose:
    return Compose(
        (
            Resize((image_size, image_size), keep_ratio=True),
            Pad(size=(image_size, image_size), pad_value=114),
            RandomHorizontalFlip(0.5, flip_idx=flip_idx),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
            ToTensor(),
        )
    )


def build_pose_eval_transforms(image_size: int = 640) -> Compose:
    return Compose(
        (
            Resize((image_size, image_size), keep_ratio=True),
            Pad(size=(image_size, image_size), pad_value=114),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
            ToTensor(),
        )
    )


def _ensure_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        raise TypeError("image is already a tensor; ToTensor should be the final transform")
    return image


def _ensure_numpy_array(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _scale_boxes(boxes: np.ndarray, scale_factor: np.ndarray, image_shape: tuple[int, int, int], clip: bool) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes *= scale_factor
    if clip:
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_shape[1])
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_shape[0])
    return boxes


def _scale_keypoints(
    keypoints: np.ndarray,
    w_scale: float,
    h_scale: float,
    image_shape: tuple[int, int, int],
    clip: bool,
) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)
    if keypoints.size == 0:
        return keypoints.reshape(0, keypoints.shape[1] if keypoints.ndim == 3 else 0, keypoints.shape[-1] if keypoints.ndim else 3)
    keypoints[..., 0] *= w_scale
    keypoints[..., 1] *= h_scale
    if clip:
        keypoints = _clip_keypoints(keypoints, image_shape[1], image_shape[0])
    return keypoints


def _clip_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    outside = (keypoints[..., 0] < 0) | (keypoints[..., 0] > width) | (keypoints[..., 1] < 0) | (keypoints[..., 1] > height)
    if keypoints.shape[-1] >= 3:
        keypoints[..., 2] = np.where(outside, 0, keypoints[..., 2])
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, width)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, height)
    return keypoints


def _flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    flipped = boxes.copy()
    flipped[..., 0::4] = width - boxes[..., 2::4]
    flipped[..., 2::4] = width - boxes[..., 0::4]
    return flipped


def _flip_keypoints(keypoints: np.ndarray, width: int, flip_idx: Sequence[int]) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)
    if keypoints.size == 0:
        return keypoints.reshape(0, len(flip_idx), keypoints.shape[-1] if keypoints.ndim else 3)
    if len(flip_idx) != keypoints.shape[1]:
        raise ValueError(f"flip_idx length {len(flip_idx)} does not match keypoints {keypoints.shape[1]}")
    flipped = keypoints[:, flip_idx, :].copy()
    flipped[..., 0] = width - flipped[..., 0]
    return flipped
