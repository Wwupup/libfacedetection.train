# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

This repository provides the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). It includes a lightweight PyTorch implementation of YuNet, WIDER Face data loading, training, evaluation, and model export tools for ONNX, C++ source code, and TFLite.

The training code is implemented directly with PyTorch and does not require MMDetection or MMCV.

Visualization of the YuNet network architecture: [[netron]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_320_320.onnx).

## Contents

- [Installation](#installation)
- [Preparation](#preparation)
- [Training](#training)
- [Evaluation on WIDER Face](#evaluation-on-wider-face)
- [Export CPP source code](#export-cpp-source-code)
- [Export to ONNX model](#export-to-onnx-model)
- [Export to TFLite model](#export-to-tflite-model)
- [Compare ONNX model with other works](#compare-onnx-model-with-other-works)
- [Testing](#testing)
- [Citation](#citation)

## Installation

1. Create and activate a conda environment.

   ```shell
   conda create -n yunet python=3.11
   conda activate yunet
   ```

2. Install [PyTorch](https://pytorch.org/) following the official instructions for your CUDA version.

   This codebase has been tested with PyTorch 2.11.0 and CUDA 12.6.

3. Clone this repository. We will call the cloned directory `$TRAIN_ROOT`.

   ```shell
   git clone https://github.com/ShiqiYu/libfacedetection.train.git
   cd libfacedetection.train
   ```

4. Install the Python dependencies.

   ```shell
   python -m pip install -e ".[dev]"
   ```

If you only want the runtime dependencies without editable package metadata, you can also use:

```shell
python -m pip install -r requirements.txt
```

Note: neither `pyproject.toml` nor `requirements.txt` installs `torch` or `torchvision`, so pip will not replace the PyTorch package in your conda environment.

## Preparation

1. Download the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) dataset.
2. Extract the dataset under `$TRAIN_ROOT/data/widerface` as follows:

   ```shell
   data/widerface
   |-- WIDER_train
   |   `-- images
   |-- WIDER_val
   |   `-- images
   `-- labelv2
       |-- train
       |   `-- labelv2.txt
       `-- val
           |-- gt
           `-- labelv2.txt
   ```

The `labelv2` annotations come from [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd).

You can check the dataset layout with:

```shell
python -m yunet_train.tools.check_widerface --split train --check-images 10
python -m yunet_train.tools.check_widerface --split val --check-images 10
```

## Training

Run a short smoke test first:

```shell
python -m yunet_train.cli.train --variant yunet_s --epochs 1 --batch-size 1 --workers 0 --device cpu --image-size 64 --limit-samples 1 --no-tensorboard
```

Train YuNet_n:

```shell
python -m yunet_train.cli.train --variant yunet_n --epochs 640 --batch-size 16 --workers 2 --device cuda --checkpoint-interval 80 --eval-interval 100 --work-dir work_dirs/yunet_n
```

Resume training from the latest checkpoint:

```shell
python -m yunet_train.cli.train --variant yunet_n --epochs 640 --batch-size 16 --workers 2 --device cuda --checkpoint-interval 80 --eval-interval 100 --resume work_dirs/yunet_n/latest.pth --work-dir work_dirs/yunet_n
```

Useful outputs:

```shell
work_dirs/yunet_n/latest.pth
work_dirs/yunet_n/best_loss.pth
work_dirs/yunet_n/metrics.csv
work_dirs/yunet_n/train.log
work_dirs/yunet_n/tensorboard
```

The default schedule follows the original YuNet training setup:

```text
base lr: 0.01
warmup: 1500 iterations
step decay: epoch 400 and epoch 544
max epochs: 640
```

TensorBoard can be launched with:

```shell
tensorboard --logdir work_dirs/yunet_n/tensorboard
```

## Evaluation on WIDER Face

Evaluate a trained checkpoint on WIDER Face val:

```shell
python -m yunet_train.cli.eval_widerface work_dirs/yunet_n/best_loss.pth --variant yunet_n --device cuda --batch-size 1 --workers 4 --output-dir work_dirs/yunet_n_widerface_eval --save-preds
```

Evaluate the released YuNet_n checkpoint:

```shell
python -m yunet_train.cli.eval_widerface weights/yunet_n.pth --variant yunet_n --device cuda --batch-size 1 --workers 4 --mode origin --output-dir work_dirs/legacy_yunet_n_eval --save-preds
```

Default WIDER Face evaluation settings:

```text
mode: origin size
confidence threshold: 0.02
nms threshold: 0.45
iou threshold: 0.5
```

Performance of the released YuNet_n checkpoint on WIDER Face val:

```text
AP_easy=0.892, AP_medium=0.883, AP_hard=0.811
```

## Export CPP source code

Export C++ weight data for [libfacedetection](https://github.com/ShiqiYu/libfacedetection):

```shell
python -m yunet_train.cli.export_cpp work_dirs/yunet_n/best_loss.pth --variant yunet_n --output-file work_dirs/export/facedetectcnn-data.cpp
```

The exporter fuses `Conv + BN` and generates the `ConvInfoStruct param_pConvInfo` data used by libfacedetection.

## Export to ONNX model

Export an ONNX model:

```shell
python -m yunet_train.cli.export_onnx work_dirs/yunet_n/best_loss.pth --variant yunet_n --shape 640 640 --output-file work_dirs/export/yunet_n_640_640.onnx
```

Export a dynamic-shape ONNX model:

```shell
python -m yunet_train.cli.export_onnx work_dirs/yunet_n/best_loss.pth --variant yunet_n --dynamic-export --output-file work_dirs/export/yunet_n_dynamic.onnx
```

Export and verify the ONNX output with ONNX Runtime:

```shell
python -m yunet_train.cli.export_onnx work_dirs/yunet_n/best_loss.pth --variant yunet_n --shape 640 640 --verify --output-file work_dirs/export/yunet_n_640_640.onnx
```

The ONNX outputs follow the original YuNet order:

```text
cls_8, cls_16, cls_32
obj_8, obj_16, obj_32
bbox_8, bbox_16, bbox_32
kps_8, kps_16, kps_32
```

## Export to TFLite model

TFLite export is optional and kept outside the main training dependencies.

Install the optional conversion dependencies:

```shell
python -m pip install -r requirements-tflite.txt
```

Export TFLite:

```shell
python -m yunet_train.cli.export_tflite work_dirs/yunet_n/best_loss.pth --variant yunet_n --shape 640 640 --output-file work_dirs/export/yunet_n_640_640.tflite
```

The conversion path is:

```text
PyTorch checkpoint -> ONNX -> TFLite
```

## Compare ONNX model with other works

Inference on exported ONNX models using ONNX Runtime:

```shell
python -m yunet_train.cli.compare_inference work_dirs/export/yunet_n_640_640.onnx --mode AUTO --eval --score-thresh 0.02 --nms-thresh 0.45 --out-dir work_dirs/compare_yunet_n
```

Single-image inference and visualization:

```shell
python -m yunet_train.cli.compare_inference work_dirs/export/yunet_n_640_640.onnx --mode AUTO --image image.jpg --out-dir work_dirs/sample
```

The compare CLI selects the detector type from the ONNX filename prefix. Supported prefixes are `yunet`, `scrfd`, `yolo5face`, and `retinaface`.

With Intel i7-12700K and `input_size = origin size, score_thresh = 0.02, nms_thresh = 0.45`, reference results are listed as follows:

| Model                   | AP_easy | AP_medium | AP_hard | #Params | Params Ratio | MFlops (320x320) | FPS (320x320) |
| ----------------------- | ------- | --------- | ------- | ------- | ------------ | ---------------- | ------------- |
| SCRFD0.5 (ICLR2022)     | 0.892   | 0.885     | 0.819   | 631,410 | 8.32x        | 184              | 284           |
| Retinaface0.5 (CVPR2020) | 0.907   | 0.883     | 0.742   | 426,608 | 5.62x        | 245              | 235           |
| YuNet_n (Ours)          | 0.892   | 0.883     | 0.811   | 75,856  | 1.00x        | 149              | 456           |
| YuNet_s (Ours)          | 0.887   | 0.871     | 0.768   | 54,608  | 0.72x        | 96               | 537           |

The compared models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PmnX0LPkQxGali2dvRqABr0VnE8OJ7FA?usp=sharing).

## Testing

Run tests and lint checks:

```shell
python -m pytest -q
python -m ruff check yunet_train tests
```

## Citation

We published a paper for the main idea of this repository:

```text
@article{yunet,
  title={YuNet: A Tiny Millisecond-level Face Detector},
  author={Wu, Wei and Peng, Hanyang and Yu, Shiqi},
  journal={Machine Intelligence Research},
  pages={1--10},
  year={2023},
  doi={10.1007/s11633-023-1423-y},
  publisher={Springer}
}
```

The paper can be open accessed at https://link.springer.com/article/10.1007/s11633-023-1423-y.

The loss used in training is EIoU. More details can be found in:

```text
@article{eiou,
 author={Peng, Hanyang and Yu, Shiqi},
 journal={IEEE Transactions on Image Processing},
 title={A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization},
 year={2021},
 volume={30},
 pages={5032-5044},
 doi={10.1109/TIP.2021.3077144}
}
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9429909.

We also published a paper on face detection to evaluate different methods.

```text
@article{facedetect-yu,
  author={Feng, Yuantao and Yu, Shiqi and Peng, Hanyang and Li, Yan-Ran and Zhang, Jianguo},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  title={Detect Faces Efficiently: A Survey and Evaluations},
  year={2022},
  volume={4},
  number={1},
  pages={1-18},
  doi={10.1109/TBIOM.2021.3120412}
}
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9580485.
