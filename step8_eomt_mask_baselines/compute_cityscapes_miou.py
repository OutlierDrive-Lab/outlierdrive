#!/usr/bin/env python3
"""Compute Cityscapes mIoU from saved prediction masks.

Use this helper when EoMT predictions have already been exported as PNG masks.
It mirrors the Step 5 notebook metric logic and writes a small CSV that can be
joined with the Step 8 anomaly results table.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def load_mask(path: Path) -> np.ndarray:
    import numpy as np
    from PIL import Image

    mask = np.array(Image.open(path), dtype=np.int64)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    import numpy as np

    valid = (
        (gt != ignore_index)
        & (gt >= 0)
        & (gt < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    return np.bincount(
        num_classes * gt[valid] + pred[valid],
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def prediction_path_for_gt(pred_dir: Path, gt_path: Path) -> Path:
    gt_filename = gt_path.name
    candidates = [
        gt_filename.replace("_gtFine_labelTrainIds.png", "_predTrainIds.png"),
        gt_filename.replace("_gtFine_labelTrainIds.png", ".png"),
        gt_filename,
    ]
    for filename in candidates:
        candidate = pred_dir / filename
        if candidate.exists():
            return candidate
    return pred_dir / candidates[0]


def compute_miou(
    *,
    gt_glob: str,
    pred_dir: Path,
    num_classes: int,
    ignore_index: int,
) -> tuple[float, float, np.ndarray, list[Path]]:
    import numpy as np

    gt_paths = sorted(Path(path) for path in glob.glob(gt_glob) if Path(path).is_file())
    if not gt_paths:
        raise FileNotFoundError(f"No ground-truth masks found for glob: {gt_glob}")

    total_hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    missing_predictions: list[Path] = []

    for gt_path in gt_paths:
        pred_path = prediction_path_for_gt(pred_dir, gt_path)
        if not pred_path.exists():
            missing_predictions.append(pred_path)
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {pred_path}: pred {pred.shape}, gt {gt.shape}")
        total_hist += confusion_matrix(gt, pred, num_classes, ignore_index)

    intersection = np.diag(total_hist)
    union = total_hist.sum(axis=1) + total_hist.sum(axis=0) - intersection
    class_iou = intersection / np.maximum(union, 1)
    mean_iou = float(np.mean(class_iou) * 100.0)
    pixel_accuracy = float(intersection.sum() / np.maximum(total_hist.sum(), 1) * 100.0)
    return mean_iou, pixel_accuracy, class_iou * 100.0, missing_predictions


def write_summary(
    *,
    output_csv: Path,
    checkpoint: str,
    config: str,
    miou: float,
    pixel_accuracy: float,
    missing_predictions: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint",
                "config",
                "miou",
                "pixel_accuracy",
                "missing_predictions",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "checkpoint": checkpoint,
                "config": config,
                "miou": miou,
                "pixel_accuracy": pixel_accuracy,
                "missing_predictions": missing_predictions,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--gt-glob", required=True, help="Glob for Cityscapes val labelTrainIds masks.")
    parser.add_argument("--pred-dir", required=True, type=Path, help="Directory containing prediction PNG masks.")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--output-csv", type=Path, default=Path("step8_eomt_mask_baselines/eomt_miou_results.csv"))
    parser.add_argument("--per-class-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    miou, pixel_accuracy, class_iou, missing = compute_miou(
        gt_glob=args.gt_glob,
        pred_dir=args.pred_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )
    write_summary(
        output_csv=args.output_csv,
        checkpoint=args.checkpoint_name,
        config=args.config,
        miou=miou,
        pixel_accuracy=pixel_accuracy,
        missing_predictions=len(missing),
    )

    if args.per_class_csv:
        args.per_class_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_class_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["class_id", "class_name", "iou"])
            for class_id, iou in enumerate(class_iou):
                class_name = CITYSCAPES_CLASSES[class_id] if class_id < len(CITYSCAPES_CLASSES) else str(class_id)
                writer.writerow([class_id, class_name, iou])

    print(
        f"{args.checkpoint_name}: mIoU={miou:.4f}, "
        f"pixel_accuracy={pixel_accuracy:.4f}, missing_predictions={len(missing)}"
    )


if __name__ == "__main__":
    main()
