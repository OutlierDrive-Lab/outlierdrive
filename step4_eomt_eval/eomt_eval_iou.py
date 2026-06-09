# ------------------------------------------------------------
# EoMT semantic segmentation IoU evaluation on Cityscapes
#
# Adapted from the original eval_iou.py idea:
# - load model
# - run inference on Cityscapes val set
# - compute per-class IoU and mIoU
# - optionally save prediction masks
#
# ------------------------------------------------------------

import os
import sys
import csv
import yaml
import torch
import argparse
import warnings
import importlib
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from contextlib import nullcontext
from torch.amp import autocast


CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

NUM_CLASSES = 19
IGNORE_INDEX = 255


def setup_repo_path(repo_root: str):
    """
    Makes sure local EoMT modules like datasets.cityscapes_semantic
    are imported from the repo, not from external packages.
    """
    os.chdir(repo_root)

    if repo_root in sys.path:
        sys.path.remove(repo_root)

    sys.path.insert(0, repo_root)

    # Avoid conflict with HuggingFace datasets package.
    for name in list(sys.modules):
        if name == "datasets" or name.startswith("datasets."):
            del sys.modules[name]


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config, data_path, batch_size=1, num_workers=0, img_size=None):
    """
    Dynamically loads the datamodule from the YAML config.
    """
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module_cls = getattr(importlib.import_module(data_module_name), class_name)

    data_module_kwargs = config["data"].get("init_args", {})

    if img_size is not None:
        data_module_kwargs["img_size"] = list(img_size)

    data = data_module_cls(
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        check_empty_targets=False,
        **data_module_kwargs,
    ).setup()

    # Force the same image size used during training/inference.
    if img_size is not None:
        data.img_size = tuple(img_size)

    return data


def build_model(
    config,
    device,
    data=None,
    reference_img_size=None,
    reference_num_classes=None,
    num_q=None,
):
    """
    Dynamically builds the EoMT semantic model from config.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    if data is not None:
        img_size = reference_img_size if reference_img_size is not None else data.img_size
        num_classes = reference_num_classes if reference_num_classes is not None else data.num_classes
    else:
        img_size = reference_img_size
        num_classes = reference_num_classes

    if img_size is None:
        raise ValueError("img_size is None. Pass --img-size or provide data.img_size.")

    if num_classes is None:
        raise ValueError("num_classes is None. Provide data or reference_num_classes.")

    # Encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

    encoder = encoder_cls(
        img_size=tuple(img_size),
        **encoder_cfg.get("init_args", {}),
    )

    # Network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)

    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items()
        if k != "encoder"
    }

    # Must match training override, for example --model.network.num_q 200
    if num_q is not None:
        network_kwargs["num_q"] = num_q

    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)

    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items()
        if k != "network"
    }

    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(
        img_size=tuple(img_size),
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )

    return model.eval().to(device)


def load_weights(model, checkpoint_path, device):
    """
    Loads .ckpt or .bin weights 
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("First missing keys:", missing[:20])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:20])

    return model.eval().to(device)


def infer_semantic(img, target, model, data, device):
    """
    Runs EoMT semantic inference on one Cityscapes image.

    Returns:
        pred_array: predicted trainId mask, shape H x W
        target_array: GT trainId mask, shape H x W
    """
    model.eval()

    use_cuda_amp = str(device).startswith("cuda")
    amp_context = autocast(dtype=torch.float16, device_type="cuda") if use_cuda_amp else nullcontext()

    with torch.inference_mode(), amp_context:
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1],
            data.img_size,
            mode="bilinear",
            align_corners=False,
        )

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits,
            class_logits_per_layer[-1],
        )

        logits = model.revert_window_logits_semantic(
            crop_logits,
            origins,
            img_sizes,
        )

        preds = logits[0].argmax(0).cpu()

    pred_array = preds.numpy().astype(np.uint8)

    target_array = model.to_per_pixel_targets_semantic(
        [target],
        IGNORE_INDEX,
    )[0].cpu().numpy().astype(np.uint8)

    return pred_array, target_array


def confusion_matrix(gt, pred, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """
    Build confusion matrix for one image.

    Rows: ground truth classes
    Columns: predicted classes
    """
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    valid_mask = (
        (gt != ignore_index) &
        (gt >= 0) & (gt < num_classes) &
        (pred >= 0) & (pred < num_classes)
    )

    hist = np.bincount(
        num_classes * gt[valid_mask] + pred[valid_mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return hist


def compute_iou_from_hist(hist):
    intersection = np.diag(hist)
    gt_pixels = hist.sum(axis=1)
    pred_pixels = hist.sum(axis=0)
    union = gt_pixels + pred_pixels - intersection

    class_iou = intersection / np.maximum(union, 1)
    mean_iou = class_iou.mean()

    pixel_accuracy = intersection.sum() / np.maximum(hist.sum(), 1)

    return class_iou, mean_iou, pixel_accuracy


def save_prediction_png(pred_array, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(pred_array.astype(np.uint8)).save(save_path)


def save_results_csv(class_iou, mean_iou, pixel_accuracy, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Class ID", "Class Name", "IoU (%)"])

        for class_id, class_name in enumerate(CITYSCAPES_CLASSES):
            writer.writerow([
                class_id,
                class_name,
                class_iou[class_id] * 100,
            ])

        writer.writerow([])
        writer.writerow(["mIoU (%)", mean_iou * 100])
        writer.writerow(["Pixel Accuracy (%)", pixel_accuracy * 100])


def evaluate(args):
    setup_repo_path(args.repo_root)

    config = load_yaml_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = tuple(args.img_size)

    print("Repo root:", args.repo_root)
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Data path:", args.data_path)
    print("Device:", device)
    print("Image size:", img_size)
    print("num_q:", args.num_q)

    data = load_data(
        config=config,
        data_path=args.data_path,
        batch_size=1,
        num_workers=args.num_workers,
        img_size=img_size,
    )

    print("Data module:", type(data))
    print("Data img_size:", data.img_size)
    print("Data num_classes:", data.num_classes)

    model = build_model(
        config=config,
        device=device,
        data=data,
        reference_img_size=img_size,
        num_q=args.num_q,
    )

    model = load_weights(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    val_dataset = data.val_dataloader().dataset

    if args.max_images is not None:
        num_images = min(args.max_images, len(val_dataset))
    else:
        num_images = len(val_dataset)

    print("Validation images:", len(val_dataset))
    print("Evaluating images:", num_images)

    total_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for idx in tqdm(range(num_images), desc="Evaluating"):
        img, target = val_dataset[idx]

        pred_array, target_array = infer_semantic(
            img=img,
            target=target,
            model=model,
            data=data,
            device=device,
        )

        if pred_array.shape != target_array.shape:
            raise ValueError(
                f"Shape mismatch at idx={idx}: "
                f"pred={pred_array.shape}, target={target_array.shape}"
            )

        total_hist += confusion_matrix(
            gt=target_array,
            pred=pred_array,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
        )

        if args.save_preds:
            pred_name = f"cityscapes_val_{idx:06d}_predTrainIds.png"
            pred_path = os.path.join(args.output_dir, "predictions", pred_name)
            save_prediction_png(pred_array, pred_path)

    class_iou, mean_iou, pixel_accuracy = compute_iou_from_hist(total_hist)

    print("\n=======================================")
    print("Per-Class IoU:")
    for class_id, class_name in enumerate(CITYSCAPES_CLASSES):
        print(f"{class_id:2d} {class_name:15s}: {class_iou[class_id] * 100:6.2f}%")

    print("=======================================")
    print(f"MEAN IoU:          {mean_iou * 100:.2f}%")
    print(f"Pixel Accuracy:    {pixel_accuracy * 100:.2f}%")
    print("=======================================\n")

    if args.output_dir is not None:
        csv_path = os.path.join(args.output_dir, "iou_results.csv")
        save_results_csv(
            class_iou=class_iou,
            mean_iou=mean_iou,
            pixel_accuracy=pixel_accuracy,
            csv_path=csv_path,
        )
        print("Saved CSV results to:", csv_path)

        hist_path = os.path.join(args.output_dir, "confusion_matrix.npy")
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(hist_path, total_hist)
        print("Saved confusion matrix to:", hist_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EoMT semantic segmentation IoU on Cityscapes."
    )

    parser.add_argument(
        "--repo-root",
        type=str,
        default="/content/outlierdrive/eomt",
        help="Path to EoMT repo root. This folder should contain main.py, datasets/, training/, configs/.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path to Cityscapes semantic EoMT config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned .ckpt or .bin checkpoint.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Cityscapes dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eomt_cityscapes_eval",
        help="Directory where CSV, confusion matrix, and predictions are saved.",
    )

    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="If set, saves predicted trainId PNG masks.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, for example cuda:0 or cpu.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Model image size. Must match training, e.g. --img-size 640 640.",
    )

    parser.add_argument(
        "--num-q",
        type=int,
        default=200,
        help="Number of queries. Must match training override --model.network.num_q.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging, e.g. --max-images 10.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)


