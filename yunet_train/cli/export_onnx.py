from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from yunet_train.models import build_yunet
from yunet_train.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight YuNet checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--shape", type=int, nargs="+", default=[640, 640])
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--dynamic-export", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = export_onnx(args)
    print(f"Successfully exported ONNX model: {output_file}")


def export_onnx(args: argparse.Namespace) -> Path:
    input_shape = _parse_input_shape(args.shape)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_yunet(variant)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(args.device).eval()

    output_file = args.output_file or _default_output_file(args.checkpoint, variant, input_shape, args.dynamic_export)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    example_input = torch.randn(input_shape, dtype=torch.float32, device=args.device)
    output_names = _output_names()
    dynamic_axes = None
    if args.dynamic_export:
        dynamic_axes = {name: {0: "batch", 1: "dim"} for name in output_names}
        dynamic_axes["input"] = {0: "batch", 2: "height", 3: "width"}

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_file),
            input_names=["input"],
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    _check_onnx(output_file)
    if args.verify:
        _verify_onnx(model, example_input, output_file)
    return output_file


def _parse_input_shape(shape: list[int]) -> tuple[int, int, int, int]:
    if len(shape) == 1:
        height = width = shape[0]
    elif len(shape) == 2:
        height, width = shape
    else:
        raise ValueError("--shape expects one int or two ints")
    return (1, 3, height, width)


def _default_output_file(
    checkpoint: Path,
    variant: str,
    input_shape: tuple[int, int, int, int],
    dynamic_export: bool,
) -> Path:
    tag = "dynamic" if dynamic_export else f"{input_shape[-2]}_{input_shape[-1]}"
    return Path("work_dirs") / "export" / f"{checkpoint.stem}_{variant}_{tag}.onnx"


def _output_names() -> list[str]:
    names: list[str] = []
    for head in ("cls", "obj", "bbox", "kps"):
        names.extend([f"{head}_{stride}" for stride in (8, 16, 32)])
    return names


def _check_onnx(output_file: Path) -> None:
    import onnx

    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)

    inputs = model.graph.input
    name_to_input = {graph_input.name: graph_input for graph_input in inputs}
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(model, str(output_file))


def _verify_onnx(model: torch.nn.Module, example_input: torch.Tensor, output_file: Path) -> None:
    import onnxruntime

    with torch.no_grad():
        torch_outputs = [output.detach().cpu().numpy() for output in _flatten_export_outputs(model, example_input)]
    session = onnxruntime.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: example_input.detach().cpu().numpy()})
    if len(torch_outputs) != len(onnx_outputs):
        raise AssertionError(f"ONNX output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}")
    for idx, (torch_output, onnx_output) in enumerate(zip(torch_outputs, onnx_outputs)):
        np.testing.assert_allclose(
            onnx_output,
            torch_output,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"ONNX output {idx} differs from PyTorch",
        )
    print("The numerical values are close between PyTorch and ONNX")


def _flatten_export_outputs(model: torch.nn.Module, image: torch.Tensor) -> list[torch.Tensor]:
    cls_scores, bbox_preds, objectnesses, kps_preds = model(image)
    batch_size = image.shape[0]
    cls = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in cls_scores
    ]
    obj = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in objectnesses
    ]
    bbox = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for pred in bbox_preds
    ]
    kps = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 10)
        for pred in kps_preds
    ]
    return cls + obj + bbox + kps


if __name__ == "__main__":
    main()
