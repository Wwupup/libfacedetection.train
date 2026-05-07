from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
COCO8_POSE_ROOT = DATA_ROOT / "coco8-pose"
COCO_ROOT = DATA_ROOT / "coco-pose"
COCO_VAL2017_IMAGE_DIR = COCO_ROOT / "val2017"
COCO_PERSON_KEYPOINTS_VAL2017 = COCO_ROOT / "annotations" / "person_keypoints_val2017.json"
