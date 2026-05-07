# YuNet Multi-Task Architecture

## Status

This repository is now structured as a multi-task YuNet training and export library.

The supported tasks are:

- `face`: WIDER Face detection with five-point face landmarks.
- `pose`: YOLO-pose / COCO-style person box and keypoint detection.

The public command-line workflows live in `yunet_train.cli`. Task implementation details live under `yunet_train.tasks`.

## Architecture

```text
yunet_train/
  engine/
    assigners.py
    checkpoint.py
    codec.py
    loop.py
    losses.py
    nms.py
    onnx_export.py
    priors.py
    scheduler.py

  models/
    backbone.py
    config.py
    init.py
    layers.py
    neck.py

  tasks/
    face/
      config.py
      dataset.py
      transforms.py
      collate.py
      types.py
      model.py
      head.py
      criterion.py
      postprocess.py
      evaluation.py
      trainer.py
      widerface.py

    pose/
      config.py
      dataset.py
      transforms.py
      collate.py
      types.py
      model.py
      criterion.py
      losses.py
      codec.py
      postprocess.py
      coco_eval.py
      visualize.py
      trainer.py

  cli/
  tools/
```

The dependency direction is:

```text
cli/tools -> tasks -> engine/models
```

`engine` and shared `models` do not depend on task packages.

## Module Responsibilities

### `engine`

`engine` contains task-independent mechanisms:

- checkpoint save/load
- learning-rate scheduling
- train/eval loss loops
- ONNX export mechanics
- bbox and keypoint codec helpers
- NMS
- multi-level prior generation
- assignment
- bbox overlap and bbox loss functions

`engine` must not contain WIDER Face parsing, YOLO-pose parsing, COCO evaluation, task-specific dataclasses, or task-specific postprocessors.

### `models`

`models` contains shared YuNet network structure:

- model variant configuration
- backbone
- neck
- reusable convolution blocks
- initialization

Task-specific heads and assembled task models live under `tasks/*`.

### `tasks.face`

The face task owns:

- WIDER Face label parsing
- face samples and batches
- face-specific transforms
- YuNet face head and assembled face model
- face criterion
- face postprocessor
- WIDER Face evaluation
- face training adapter

### `tasks.pose`

The pose task owns:

- YOLO-pose label parsing
- pose samples and batches
- pose-specific transforms and flip index
- YuNet pose head and assembled pose model
- OKS and visibility losses
- pose criterion
- pose postprocessor
- COCO keypoint evaluation
- pose visualization
- pose training adapter

## Extension Rules

When adding a new YuNet task:

1. Put task-specific code under `yunet_train/tasks/<task_name>`.
2. Reuse `yunet_train.models` for shared YuNet blocks.
3. Reuse `yunet_train.engine` for checkpointing, priors, assignment, bbox losses, NMS, training loop mechanics, and ONNX export mechanics.
4. Keep task-specific dataclasses, datasets, transforms, losses, postprocessors, and evaluations inside the task package.
5. Add CLI files under `yunet_train/cli` that depend on the task package and `engine`.
6. Do not make `engine` or `models` import the new task.

## Validation

After architecture changes, run:

```shell
python -m pytest -q
python -m ruff check yunet_train tests
```

Useful dependency checks:

```shell
rg "yunet_train\\.tasks" yunet_train/engine yunet_train/models/backbone.py yunet_train/models/neck.py yunet_train/models/layers.py yunet_train/models/init.py yunet_train/models/config.py
rg "yunet_train\\.tasks\\.face" yunet_train/tasks/pose
rg "yunet_train\\.tasks\\.pose" yunet_train/tasks/face
```

Expected result for all three checks:

```text
no matches
```
