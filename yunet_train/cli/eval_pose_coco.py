from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from yunet_train.tasks.pose import COCO_PERSON_KEYPOINTS_VAL2017, COCO_VAL_IMAGE_DIR, build_yunet_pose
from yunet_train.tasks.pose.coco_eval import COCOPoseEvalDataset, collect_coco_keypoint_predictions, evaluate_coco_keypoints
from yunet_train.engine import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YuNet pose with official COCO keypoint AP.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ann-file", type=Path, default=COCO_PERSON_KEYPOINTS_VAL2017)
    parser.add_argument("--image-dir", type=Path, default=COCO_VAL_IMAGE_DIR)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/pose_coco_eval"))
    return parser.parse_args()


def main() -> None:
    result = eval_pose_coco(parse_args())
    metrics = " ".join(f"{key}={value:.6f}" for key, value in result.metrics.items())
    print(f"COCO keypoint AP {metrics}")


def eval_pose_coco(args: argparse.Namespace):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    device = torch.device(args.device)
    model = build_yunet_pose(variant, kpt_shape=(17, 3))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(device).eval()

    dataset = COCOPoseEvalDataset(
        ann_file=args.ann_file,
        image_dir=args.image_dir,
        image_size=args.image_size,
        limit_samples=args.limit_samples,
    )
    predictions = collect_coco_keypoint_predictions(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        category_id=args.category_id,
    )
    result = evaluate_coco_keypoints(
        ann_file=args.ann_file,
        predictions=predictions,
        results_file=args.out_dir / "pose_coco_results.json",
    )
    _write_metrics(args.out_dir / "pose_coco_metrics.csv", result.metrics, result.num_predictions)
    return result


def _write_metrics(path: Path, metrics: dict[str, float], num_predictions: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ("num_predictions", *metrics.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"num_predictions": num_predictions, **metrics})


if __name__ == "__main__":
    main()
