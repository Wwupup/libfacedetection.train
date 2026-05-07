# YuNet Pose Adaptation Plan

## Status

This document is the implementation plan for adding a human pose detection task to this repository. The pose path is being added in parallel with the existing face detection path, not by modifying the face workflow in place.

Current implementation status:

- Added `yunet_train.tasks.pose` as the pose task namespace.
- Added YOLO pose dataset parsing for `data/coco8-pose`.
- Added pose dataclasses, collate, resize/pad/flip/normalize/tensor transforms.
- Added COCO17 flip index and OKS sigmas.
- Added `YuNetPose` and `YuNetPoseHead` with `17 x 3` keypoint output support.
- Added pose keypoint encode/decode helpers.
- Added pose postprocessor with score filtering and NMS.
- Added OKS-style keypoint location loss and keypoint visibility BCE.
- Added `train_pose.py` for CPU/GPU smoke training, resume-ready checkpoints, best checkpoint, logs, and metrics CSV.
- Added pose augmentation visualization output.
- Added pose tiny-overfit check for `coco8-pose`.
- Added pose validation-loss evaluation CLI with optional prediction visualization.
- Added official COCO keypoint AP evaluation through optional `pycocotools`.
- Added pose ONNX export CLI with ONNX Runtime parity verification.
- Added unit tests for parser, transforms, collate, model output shape, codec, postprocess, pose losses, pose visualization, pose training CLI smoke/resume, and tiny overfit.

Verified locally:

```shell
conda run -n yunet python -m pytest tests\test_pose_dataset.py tests\test_pose_transforms.py tests\test_pose_model.py tests\test_pose_losses.py -q
conda run -n yunet python -m ruff check yunet_train tests
```

The current YuNet face detection workflow must remain stable:

- WIDER Face training and evaluation continue to work.
- Legacy `weights/yunet_n.pth` and `weights/yunet_s.pth` loading remains unchanged.
- Existing ONNX, C++ and TFLite export behavior for face detection remains unchanged.

## Goal

Add a lightweight human pose detection task that reuses the YuNet backbone and feature pyramid, while following the useful pose-task ideas from Ultralytics YOLO:

- A detection-style pose model that predicts person boxes and keypoints in one pass.
- Configurable keypoint shape, for example COCO pose `17 x 3`.
- Keypoint flip index for horizontal flip augmentation.
- OKS-style keypoint location loss.
- Keypoint visibility/objectness loss.
- COCO-style pose evaluation when the implementation is mature enough.

The target output is:

```text
image -> person bbox + class/objectness score + K keypoints
```

For COCO human pose:

```text
K = 17
keypoint dim = 3  # x, y, visibility
```

## Non-Goals

- Do not copy Ultralytics source code into this BSD-licensed project. Ultralytics is AGPL-licensed, so its implementation should be treated as a reference only.
- Do not replace the current YuNet face detector.
- Do not add a full YOLO framework, YAML model parser, trainer abstraction, or task registry.
- Do not implement heatmap-based top-down pose estimation in the first version.
- Do not add heavy dependencies to the default training requirements unless they are needed by the existing face task.

## Design Summary

The recommended design is a parallel task implementation:

```text
face task: existing YuNet face detector
pose task: new YuNetPose detector
```

The two tasks can share low-level model blocks, backbone, neck, checkpoint utilities, logging, schedulers, and some transforms. The pose-specific pieces should live in a separate namespace so that face training does not become harder to reason about.

Proposed layout:

```text
yunet_train/
  pose/
    __init__.py
    config.py
    dataset.py
    transforms.py
    collate.py
    model.py
    losses.py
    codec.py
    criterion.py
    postprocess.py
    trainer.py
    evaluation.py
  cli/
    train_pose.py
    eval_pose.py
    export_pose_onnx.py
tests/
  test_pose_dataset.py
  test_pose_transforms.py
  test_pose_model.py
  test_pose_losses.py
  test_pose_codec.py
  test_pose_train_cli.py
```

If later more tasks are added, this can be promoted to a generic `tasks/` layout. For now, a focused `pose/` package is simpler and safer.

## Data Format

Start with YOLO pose format because it is compact and easy to inspect:

```text
class cx cy w h x1 y1 v1 x2 y2 v2 ... xK yK vK
```

Coordinates are normalized to image width and height.

Dataset config should use a small YAML file:

```yaml
path: data/coco8-pose
train: images/train
val: images/val

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
```

Initial parser responsibilities:

- Resolve dataset root, image directories, and label directories.
- Parse normalized boxes into absolute `xyxy`.
- Parse keypoints into shape `(N, K, D)`.
- Preserve visibility flags.
- Validate that label length matches `5 + K * D`.
- Support empty labels.

Later, add COCO JSON support if official COCO keypoint evaluation becomes a priority.

## Augmentation Requirements

Pose augmentation must transform boxes and keypoints together. It cannot reuse face transforms blindly.

Minimum first version:

- Letterbox or resize with padding.
- Horizontal flip with `flip_idx`.
- Small-object filtering by person bbox size.
- Normalize and tensor conversion.

Important details:

- Keypoints outside the image should have visibility set to zero or be masked out consistently.
- Horizontal flip must both mirror x-coordinates and reorder left/right keypoints.
- All transform tests should use deterministic inputs with known expected coordinates.

Recommended first tests:

- Resize scales boxes and all keypoint x/y values.
- Padding offsets boxes and all keypoint x/y values.
- Horizontal flip swaps left/right keypoints according to `flip_idx`.
- Invisible keypoints do not contribute to keypoint location loss.

## Model Architecture

Reuse:

```text
YuNetBackbone -> TFPN -> P3/P4/P5 features
```

Add a pose-specific head:

```text
YuNetPoseHead
  cls branch
  bbox branch
  obj branch
  keypoint branch
```

For the first version, keep the current bbox/objectness style close to the face detector. Add a separate pose branch per feature level:

```text
keypoint output channels = kpt_num * kpt_dim
```

For COCO:

```text
17 * 3 = 51 channels
```

Suggested output contract during training:

```python
(
    cls_scores,     # list[Tensor], each (B, 1, H, W)
    bbox_preds,     # list[Tensor], each (B, 4, H, W)
    objectnesses,   # list[Tensor], each (B, 1, H, W)
    kpt_preds,      # list[Tensor], each (B, K * D, H, W)
)
```

Suggested keypoint decode:

```text
x = prior_x + pred_x * stride
y = prior_y + pred_y * stride
visibility_logit = pred_visibility
```

This is close to the current YuNet keypoint decode and avoids introducing YOLO-specific decode behavior in the first version.

## Assignment and Targets

The pose task should reuse the current positive sample assignment where possible, but the criterion must know which GT instance each positive point matched.

Required target information per positive point:

```text
matched bbox
matched class
matched keypoints
matched keypoint visibility
matched GT index
```

If the current assigner does not expose all of this cleanly, add a small structured return type instead of threading more parallel tensors through the loss code.

Do not couple pose targets to WIDER Face-specific assumptions.

## Loss Design

Use detection losses plus pose-specific losses:

```text
total_loss =
  cls_loss * cls_weight
  + obj_loss * obj_weight
  + bbox_loss * bbox_weight
  + keypoint_location_loss * pose_weight
  + keypoint_visibility_loss * kobj_weight
```

Keypoint location loss should be OKS-style:

```text
d = (pred_x - gt_x)^2 + (pred_y - gt_y)^2
e = d / ((2 * sigma)^2 * area * 2)
loss = mean((1 - exp(-e)) * visibility_mask)
```

Use COCO sigmas for `kpt_shape == [17, 3]`:

```text
[0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89] / 10
```

For non-COCO keypoint layouts, default to equal sigmas:

```text
sigma = 1 / kpt_num
```

Visibility loss:

```text
BCEWithLogitsLoss(pred_visibility, gt_visibility > 0)
```

Only positive samples should contribute to keypoint losses.

## Inference and Postprocess

Pose inference should:

1. Decode bbox predictions.
2. Decode keypoint x/y predictions.
3. Apply sigmoid to objectness/class scores.
4. Apply sigmoid to keypoint visibility logits.
5. Run NMS on person boxes.
6. Return boxes, scores, labels, and keypoints.

Suggested result type:

```python
@dataclass(frozen=True)
class PoseDetectionResult:
    boxes: torch.Tensor       # (N, 4), xyxy
    scores: torch.Tensor      # (N,)
    labels: torch.Tensor      # (N,)
    keypoints: torch.Tensor   # (N, K, D)
```

Do not reuse the face `DetectionResult` if doing so forces pose-specific shape assumptions into face inference.

## Evaluation

First version:

- Provide visual validation output.
- Provide deterministic unit tests for OKS calculation.
- Provide smoke validation on a tiny dataset.

Mature version:

- COCO keypoint AP via an optional dependency. `done`
- Keep COCO evaluation dependencies outside the core face-training requirements if they are heavy.

Potential optional dependency file:

```text
requirements-pose.txt
```

Install it only when official COCO AP is needed:

```shell
python -m pip install -r requirements-pose.txt
```

The current `requirements-pose.txt` contains `pycocotools==2.0.11`.

## CLI Plan

Add separate pose commands:

```shell
python -m yunet_train.cli.train_pose --data-root data/coco8-pose --variant yunet_n
python -m yunet_train.tools.check_pose_overfit --data-root data/coco8-pose --samples 4 --epochs 120 --image-size 160
python -m yunet_train.cli.eval_pose work_dirs/yunet_pose_n/best_loss.pth --data-root data/coco8-pose
python -m yunet_train.cli.eval_pose_coco work_dirs/yunet_pose_n/best_loss.pth --ann-file data/coco/annotations/person_keypoints_val2017.json --image-dir data/coco/val2017
python -m yunet_train.cli.export_pose_onnx work_dirs/yunet_pose_n/best_loss.pth --output-file work_dirs/export/yunet_pose_n.onnx
```

Do not add pose-specific options to `yunet_train.cli.train` until there is a clear shared abstraction. Keeping separate commands protects the stable face workflow.

## Implementation Phases

### Phase 0: Quality Baseline

- Confirm current face tests pass.
- Add the pose design document.
- Add no runtime dependency on Ultralytics.
- Add no copied AGPL implementation.

Exit criteria:

```text
python -m pytest -q
python -m ruff check yunet_train tests
```

### Phase 1: Pose Data Pipeline

Implement:

- Pose YAML config loader.
- YOLO pose label parser. `done`
- Pose sample dataclass. `done`
- Pose collate. `done`
- Resize/letterbox/flip transforms. `done`
- Pose visualization tool. `done`

Tests:

- Parser shape validation.
- Empty label handling.
- Flip index correctness. `done`
- Transform coordinate correctness. `done`
- Collate output shape. `done`

Exit criteria:

- A tiny pose dataset can be loaded and visualized.
- No face tests break.

### Phase 2: Pose Model and Decode

Implement:

- `PoseModelConfig`.
- `YuNetPose`. `done`
- `YuNetPoseHead`. `done`
- Pose keypoint encode/decode helpers. `done`
- Pose postprocessor. `done`

Tests:

- Forward shapes for `kpt_shape=(17, 3)`. `done`
- Decode known tensors to expected coordinates. `done`
- NMS keeps aligned boxes/keypoints. `done`
- Export shape smoke if ONNX export is added in this phase.

Exit criteria:

- Model forward works on CPU.
- Postprocess returns boxes and keypoints with stable shapes.

### Phase 3: Pose Criterion

Implement:

- Structured assignment output if needed.
- OKS-style keypoint location loss. `done`
- Keypoint visibility loss. `done`
- Pose criterion.

Tests:

- Invisible keypoints do not contribute to location loss. `done`
- Perfect prediction has near-zero keypoint location loss. `done`
- Larger person area reduces the same absolute keypoint error.
- Criterion handles images with no persons.
- Synthetic batch runs forward, loss, and backward. `done`

Exit criteria:

- One synthetic batch can run forward, loss, backward.

### Phase 4: Pose Training CLI

Implement:

- `train_pose.py`. `done`
- Pose checkpoint save/resume. `done`
- Pose best checkpoint. `done`
- Pose metrics CSV/logging. `done`
- Tiny overfit test on a small dataset. `done`

Tests:

- CPU smoke training with `--limit-samples`. `done`
- Resume from `latest.pth`. `done`
- Logging file exists and has expected fields. `done`

Exit criteria:

- Model can overfit a tiny pose dataset without NaN. `done`

### Phase 5: Pose Evaluation and Export

Implement:

- `eval_pose.py`. `done: validation loss and optional visualization`
- Optional COCO keypoint AP. `done`
- `export_pose_onnx.py`. `done`
- Single-image pose visualization.

Tests:

- Eval smoke on tiny dataset. `done`
- COCO AP smoke on tiny synthetic COCO data. `done`
- ONNX export smoke. `done`
- Optional ONNX Runtime parity test. `done`

Exit criteria:

- Pose inference, official COCO AP evaluation, and export are usable end to end.

## Engineering Quality Rules

- Keep face and pose code paths isolated unless a shared abstraction is clearly beneficial.
- Add tests before or alongside each behavior change.
- Prefer small dataclasses and typed function boundaries over loosely shaped dictionaries.
- Do not introduce global task registries.
- Do not add heavy optional dependencies to default installation.
- Do not copy AGPL code from Ultralytics; reimplement ideas in this repository's style.
- Keep all CLI commands runnable on CPU with tiny samples for CI.
- Preserve existing public commands and checkpoint compatibility.

## Main Risks

### Direct Regression May Underperform Heatmaps

YuNet-style direct keypoint regression is simple and fast, but human pose has long limbs, occlusion, and fine keypoints. It may not match top-down heatmap models.

Mitigation:

- Treat the first version as a lightweight detector-style baseline.
- Add OKS loss and visibility handling early.
- Use visual debugging and tiny overfit checks before full training.

### Backbone May Need More High-Resolution Detail

YuNet is optimized for face detection. Person keypoints like wrists and ankles may need richer high-resolution features.

Mitigation:

- Start with P3/P4/P5.
- Consider adding P2/4 later only if evidence shows small keypoints are poor.
- Keep this as a measured follow-up, not a first implementation requirement.

### Evaluation Dependencies Can Bloat the Project

Official COCO keypoint AP may require extra dependencies.

Mitigation:

- Keep official evaluation optional.
- Maintain lightweight smoke metrics for development.

## Initial Acceptance Criteria

The first useful pose milestone is complete when:

- `python -m pytest -q` passes.
- `python -m ruff check yunet_train tests` passes.
- Face detection tests and legacy checkpoint loading still pass.
- A tiny YOLO-pose dataset can be parsed, augmented, visualized, trained for one CPU epoch, and checkpointed.
- A synthetic pose batch can run forward, loss, and backward without NaN.
- A pose model produces decoded boxes and `(N, 17, 3)` keypoints at inference time.

Only after this milestone should full COCO training and official pose AP be considered.
