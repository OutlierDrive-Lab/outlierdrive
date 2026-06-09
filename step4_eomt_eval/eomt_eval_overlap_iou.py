# ------------------------------------------------------------
# EoMT overlap IoU evaluation on Cityscapes validation set
#
# It runs inference on the mapped class space,
# and evaluates IoU on the Cityscapes.
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
from contextlib import nullcontext
from torch.amp import autocast


CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

NUM_CITYSCAPES_CLASSES = 19
IGNORE_INDEX = 255


# EoMT COCO internal output id -> Cityscapes trainId
#
# We investigated that the COCO-trained model outputs EoMT's internal COCO/panoptic ids,
# not raw COCO category ids.
DEFAULT_COCO_TO_CITYSCAPES_TRAINID = {
    0: 11,     # person -> person
    1: 18,     # bicycle -> bicycle
    2: 13,     # car -> car
    3: 17,     # motorcycle -> motorcycle
    5: 15,     # bus -> bus
    6: 16,     # train -> train
    7: 14,     # truck -> truck
    9: 6,      # traffic light -> traffic light
    11: 7,     # stop sign -> traffic sign, approximate

    100: 0,    # road -> road
    123: 1,    # pavement-merged -> sidewalk, approximate
    129: 2,    # building-other-merged -> building, approximate

    109: 3,    # wall-brick -> wall
    110: 3,    # wall-stone -> wall
    111: 3,    # wall-tile -> wall
    112: 3,    # wall-wood -> wall
    131: 3,    # wall-other-merged -> wall

    117: 4,    # fence-merged -> fence

    116: 8,    # tree-merged -> vegetation
    125: 8,    # grass-merged -> vegetation, approximate

    119: 10,   # sky-other-merged -> sky
}


def setup_repo_path(repo_root: str):
    
    os.chdir(repo_root)

    if repo_root in sys.path:
        sys.path.remove(repo_root)

    sys.path.insert(0, repo_root)

    for name in list(sys.modules):
        if name == "datasets" or name.startswith("datasets."):
            del sys.modules[name]


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cityscapes_data(city_config, data_path, batch_size=1, num_workers=0, img_size=None):
    
    data_module_name, class_name = city_config["data"]["class_path"].rsplit(".", 1)
    data_module_cls = getattr(importlib.import_module(data_module_name), class_name)

    data_module_kwargs = dict(city_config["data"].get("init_args", {}))

    if img_size is not None:
        data_module_kwargs["img_size"] = list(img_size)

    data = data_module_cls(
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        check_empty_targets=False,
        **data_module_kwargs,
    ).setup()

    if img_size is not None:
        data.img_size = tuple(img_size)

    return data


def build_model(config, device, img_size, num_classes, num_q=None):
    
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

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

    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(
        img_size=tuple(img_size),
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )

    return model.eval().to(device)


def load_weights(model, checkpoint_path, device):
    
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Checkpoint loaded:", checkpoint_path)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("First missing keys:", missing[:20])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:20])

    return model.eval().to(device)


def infer_semantic(img, target, model, city_data, device):
    
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
            city_data.img_size,
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

        pred_array = logits[0].argmax(0).cpu().numpy().astype(np.uint8)

    # This converts the Cityscapes dataset target to a Cityscapes trainId mask.
    target_array = model.to_per_pixel_targets_semantic(
        [target],
        IGNORE_INDEX,
    )[0].cpu().numpy().astype(np.uint8)

    return pred_array, target_array


def map_coco_prediction_to_cityscapes(pred_array, mapping):
    """
    Map COCO/EoMT prediction ids to Cityscapes trainIds.
    Unmapped prediction classes become IGNORE_INDEX.

    During IoU computation, pred=255 is NOT ignored if the GT class is one of the
    evaluated overlap classes. It counts as a wrong prediction.
    """
    mapped = np.full(pred_array.shape, IGNORE_INDEX, dtype=np.uint8)

    for coco_id, city_train_id in mapping.items():
        mapped[pred_array == int(coco_id)] = int(city_train_id)

    return mapped


def compute_overlap_stats(gt_city, pred_city, eval_class_ids):
    """
    Compute per-class intersection/union on selected Cityscapes classes.

    Pixels with GT outside eval_class_ids are ignored.
    Pixels with GT inside eval_class_ids and pred=IGNORE_INDEX are counted as wrong.
    """
    gt_city = gt_city.astype(np.int64)
    pred_city = pred_city.astype(np.int64)

    eval_gt_mask = np.isin(gt_city, eval_class_ids)

    intersections = {}
    unions = {}

    for cid in eval_class_ids:
        gt_c = (gt_city == cid) & eval_gt_mask
        pred_c = (pred_city == cid) & eval_gt_mask

        intersections[cid] = float(np.logical_and(gt_c, pred_c).sum())
        unions[cid] = float(np.logical_or(gt_c, pred_c).sum())

    correct = float(((gt_city == pred_city) & eval_gt_mask).sum())
    valid = float(eval_gt_mask.sum())

    return intersections, unions, correct, valid


def finalize_metrics(total_intersections, total_unions, total_correct, total_valid, eval_class_ids):
    class_iou = {}

    for cid in eval_class_ids:
        if total_unions[cid] > 0:
            class_iou[cid] = total_intersections[cid] / total_unions[cid]
        else:
            class_iou[cid] = np.nan

    valid_ious = [v for v in class_iou.values() if not np.isnan(v)]
    mean_iou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else np.nan
    pixel_accuracy = float(total_correct / total_valid) if total_valid > 0 else np.nan

    return class_iou, mean_iou, pixel_accuracy


def save_single_model_csv(model_name, eval_class_ids, class_iou, mean_iou, pixel_accuracy, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["model", "class_id", "class_name", "iou_percent"])

        for cid in eval_class_ids:
            writer.writerow([
                model_name,
                cid,
                CITYSCAPES_CLASSES[cid],
                "" if np.isnan(class_iou[cid]) else class_iou[cid] * 100.0,
            ])

        writer.writerow([])
        writer.writerow(["model", "metric", "value_percent"])
        writer.writerow([model_name, "mIoU", mean_iou * 100.0])
        writer.writerow([model_name, "Pixel Accuracy", pixel_accuracy * 100.0])



def parse_class_ids(text):
    if text is None:
        return None

    text = text.strip()

    if text == "":
        return None

    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def evaluate_model(
    model_name,
    config_path,
    checkpoint_path,
    num_classes,
    img_size,
    num_q,
    model_kind,
    city_data,
    val_dataset,
    device,
    eval_class_ids,
    max_images,
    debug_unique_preds,
):
    
    print("\n=======================================")
    print("Evaluating:", model_name)
    print("Model kind:", model_kind)
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("num_classes:", num_classes)
    print("img_size:", img_size)
    print("num_q:", num_q)
    print("=======================================")

    config = load_yaml_config(config_path)

    model = build_model(
        config=config,
        device=device,
        img_size=tuple(img_size),
        num_classes=num_classes,
        num_q=num_q,
    )

    model = load_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if max_images is not None:
        num_images = min(max_images, len(val_dataset))
    else:
        num_images = len(val_dataset)

    total_intersections = {cid: 0.0 for cid in eval_class_ids}
    total_unions = {cid: 0.0 for cid in eval_class_ids}
    total_correct = 0.0
    total_valid = 0.0

    printed_unique = False

    for idx in tqdm(range(num_images), desc=f"Evaluating {model_name}"):
        img, target = val_dataset[idx]

        pred_array, target_array = infer_semantic(
            img=img,
            target=target,
            model=model,
            city_data=city_data,
            device=device,
        )

        if debug_unique_preds and not printed_unique:
            unique_before = np.unique(pred_array)
            print(f"\nUnique predicted ids before mapping for {model_name}:")
            print(unique_before[:200])
            printed_unique = True

        if model_kind == "coco":
            pred_array = map_coco_prediction_to_cityscapes(
                pred_array=pred_array,
                mapping=DEFAULT_COCO_TO_CITYSCAPES_TRAINID,
            )
        elif model_kind == "cityscapes":
            # Already Cityscapes trainIds.
            pass
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        if pred_array.shape != target_array.shape:
            raise ValueError(
                f"Shape mismatch at idx={idx}: "
                f"pred={pred_array.shape}, target={target_array.shape}"
            )

        intersections, unions, correct, valid = compute_overlap_stats(
            gt_city=target_array,
            pred_city=pred_array,
            eval_class_ids=eval_class_ids,
        )

        for cid in eval_class_ids:
            total_intersections[cid] += intersections[cid]
            total_unions[cid] += unions[cid]

        total_correct += correct
        total_valid += valid

    class_iou, mean_iou, pixel_accuracy = finalize_metrics(
        total_intersections=total_intersections,
        total_unions=total_unions,
        total_correct=total_correct,
        total_valid=total_valid,
        eval_class_ids=eval_class_ids,
    )

    print("\nPer-class IoU on evaluated overlap classes:")
    for cid in eval_class_ids:
        value = class_iou[cid]
        text = "nan" if np.isnan(value) else f"{value * 100.0:6.2f}%"
        print(f"{cid:2d} {CITYSCAPES_CLASSES[cid]:15s}: {text}")

    print("---------------------------------------")
    print(f"{model_name} overlap mIoU:    {mean_iou * 100.0:.2f}%")
    print(f"{model_name} pixel accuracy:  {pixel_accuracy * 100.0:.2f}%")
    print("---------------------------------------")

    return {
        "class_iou": class_iou,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
    }


def evaluate(args):
    setup_repo_path(args.repo_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    city_config = load_yaml_config(args.city_config)

    city_data = load_cityscapes_data(
        city_config=city_config,
        data_path=args.data_path,
        batch_size=1,
        num_workers=args.num_workers,
        img_size=tuple(args.data_img_size),
    )

    val_dataset = city_data.val_dataloader().dataset

    eval_class_ids = parse_class_ids(args.eval_class_ids)

    if eval_class_ids is None:
        eval_class_ids = sorted(set(DEFAULT_COCO_TO_CITYSCAPES_TRAINID.values()))

    for cid in eval_class_ids:
        if cid < 0 or cid >= NUM_CITYSCAPES_CLASSES:
            raise ValueError(f"Invalid Cityscapes trainId: {cid}")

    print("Cityscapes data path:", args.data_path)
    print("Cityscapes validation images:", len(val_dataset))
    print("Evaluated Cityscapes overlap classes:")
    for cid in eval_class_ids:
        print(f"  {cid:2d}: {CITYSCAPES_CLASSES[cid]}")

    results = {}

    if args.coco_checkpoint is not None:
        if args.coco_config is None:
            raise ValueError("--coco-config is required when --coco-checkpoint is used.")

        results[args.coco_model_name] = evaluate_model(
            model_name=args.coco_model_name,
            config_path=args.coco_config,
            checkpoint_path=args.coco_checkpoint,
            num_classes=args.coco_num_classes,
            img_size=tuple(args.coco_img_size),
            num_q=args.coco_num_q,
            model_kind="coco",
            city_data=city_data,
            val_dataset=val_dataset,
            device=device,
            eval_class_ids=eval_class_ids,
            max_images=args.max_images,
            debug_unique_preds=args.debug_unique_preds,
        )

    if args.city_checkpoint is not None:
        results[args.city_model_name] = evaluate_model(
            model_name=args.city_model_name,
            config_path=args.city_config,
            checkpoint_path=args.city_checkpoint,
            num_classes=args.city_num_classes,
            img_size=tuple(args.city_img_size),
            num_q=args.city_num_q,
            model_kind="cityscapes",
            city_data=city_data,
            val_dataset=val_dataset,
            device=device,
            eval_class_ids=eval_class_ids,
            max_images=args.max_images,
            debug_unique_preds=args.debug_unique_preds,
        )

    if len(results) == 0:
        raise ValueError("No model checkpoint was provided. Use --coco-checkpoint and/or --city-checkpoint.")

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, result in results.items():
        csv_path = os.path.join(args.output_dir, f"{model_name}_overlap_iou.csv")
        save_single_model_csv(
            model_name=model_name,
            eval_class_ids=eval_class_ids,
            class_iou=result["class_iou"],
            mean_iou=result["mean_iou"],
            pixel_accuracy=result["pixel_accuracy"],
            csv_path=csv_path,
        )
        print("Saved:", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO-trained and/or Cityscapes-trained EoMT models on Cityscapes overlap classes."
    )

    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument(
        "--data-img-size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Image size used by the Cityscapes datamodule during evaluation.",
    )

    parser.add_argument(
        "--eval-class-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated Cityscapes trainIds to evaluate. "
            "Default: all Cityscapes classes appearing in DEFAULT_COCO_TO_CITYSCAPES_TRAINID."
        ),
    )

    parser.add_argument(
        "--debug-unique-preds",
        action="store_true",
        help="Print unique predicted ids before mapping for the first evaluated image of each model.",
    )

    # Cityscapes model
    parser.add_argument("--city-config", type=str, required=True)
    parser.add_argument("--city-checkpoint", type=str, default=None)
    parser.add_argument("--city-model-name", type=str, default="cityscapes_trained")
    parser.add_argument("--city-num-classes", type=int, default=19)
    parser.add_argument("--city-img-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--city-num-q", type=int, default=100)

    # COCO model
    parser.add_argument("--coco-config", type=str, default=None)
    parser.add_argument("--coco-checkpoint", type=str, default=None)
    parser.add_argument("--coco-model-name", type=str, default="coco_trained")
    parser.add_argument("--coco-num-classes", type=int, default=133)
    parser.add_argument("--coco-img-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--coco-num-q", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
