#!/usr/bin/env python3
"""Run Step 8 EoMT mask-based anomaly segmentation baselines.

This script adapts the Step 7 anomaly evaluation pattern to EoMT. It keeps
dataset handling, binary anomaly masks, AuPRC/FPR95, and CSV output consistent
with the ERFNet baselines while replacing ERFNet inference with EoMT mask-query
inference.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import inspect
import sys
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASETS = [
    "RoadAnomaly21",
    "RoadObsticle21",
    "RoadAnomaly",
    "fs_static",
    "FS_LostFound_full",
]

DEFAULT_METHODS = ["msp", "maxlogit", "entropy", "rba"]
METHOD_CHOICES = DEFAULT_METHODS + ["all"]
IGNORE_LABEL = 255


@dataclass(frozen=True)
class DatasetResult:
    checkpoint: str
    dataset: str
    method: str
    temperature: float
    auprc: float
    fpr95: float
    num_images: int
    num_ood_pixels: int
    num_ind_pixels: int
    miou: str | float = ""


def parse_img_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected image size as HxW, for example 640x640.")
    try:
        height, width = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Image size must contain integers.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Image size values must be positive.")
    return height, width


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Temperature values must be positive.")
    return parsed


def ensure_eomt_on_path(repo_root: Path) -> None:
    eomt_root = repo_root / "eomt"
    if str(eomt_root) not in sys.path:
        sys.path.insert(0, str(eomt_root))


def import_from_class_path(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read EoMT config files.") from exc

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def infer_num_classes(config: dict[str, Any], explicit_num_classes: int | None) -> int:
    if explicit_num_classes is not None:
        return explicit_num_classes

    data_args = config.get("data", {}).get("init_args", {})
    stuff_classes = data_args.get("stuff_classes")
    if stuff_classes:
        return max(stuff_classes) + 1

    raise ValueError(
        "Could not infer num_classes from the config. Pass --num-classes explicitly "
        "(19 for Cityscapes semantic, usually 133 for COCO panoptic)."
    )


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    class_head = state_dict.get("network.class_head.weight")
    if class_head is None:
        return None
    return int(class_head.shape[0] - 1)


def infer_num_q_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    query_weights = state_dict.get("network.q.weight")
    if query_weights is None:
        return None
    return int(query_weights.shape[0])


def infer_img_size_from_state_dict(state_dict: dict[str, Any]) -> tuple[int, int] | None:
    import math

    pos_embed = state_dict.get("network.encoder.backbone.pos_embed")
    patch_embed = state_dict.get("network.encoder.backbone.patch_embed.proj.weight")
    if pos_embed is None or patch_embed is None:
        return None

    num_patches = int(pos_embed.shape[1])
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        return None

    patch_size = int(patch_embed.shape[-1])
    image_size = grid_size * patch_size
    return image_size, image_size


def select_device(device_name: str):
    import torch

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def load_checkpoint_state(path: Path, device):
    import torch

    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint did not contain a state dict: {path}")
    return state


def build_eomt_model(
    *,
    repo_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
    img_size: tuple[int, int] | None,
    num_classes: int | None,
    stuff_classes: list[int] | None,
    load_ckpt_class_head: bool,
):
    import torch

    ensure_eomt_on_path(repo_root)
    config = load_yaml(config_path)
    device = select_device(device_name)
    state_dict = load_checkpoint_state(checkpoint_path, device)
    resolved_num_classes = (
        num_classes
        or infer_num_classes_from_state_dict(state_dict)
        or infer_num_classes(config, None)
    )
    resolved_img_size = img_size or infer_img_size_from_state_dict(state_dict) or (640, 640)
    checkpoint_num_q = infer_num_q_from_state_dict(state_dict)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_cls = import_from_class_path(encoder_cfg["class_path"])
    encoder_kwargs = dict(encoder_cfg.get("init_args", {}))
    # The local ViT wrapper uses ckpt_path only to decide whether timm should
    # auto-download pretrained backbone weights. We load the full EoMT checkpoint
    # below, so keep this non-null to make local smoke tests work offline.
    encoder_kwargs.setdefault("ckpt_path", str(checkpoint_path))
    encoder = encoder_cls(img_size=resolved_img_size, **encoder_kwargs)

    network_cfg = config["model"]["init_args"]["network"]
    network_cls = import_from_class_path(network_cfg["class_path"])
    network_kwargs = {
        key: value
        for key, value in network_cfg.get("init_args", {}).items()
        if key != "encoder"
    }
    network_kwargs["masked_attn_enabled"] = False
    if checkpoint_num_q is not None:
        network_kwargs["num_q"] = checkpoint_num_q
    network = network_cls(
        encoder=encoder,
        num_classes=resolved_num_classes,
        **filter_kwargs(network_cls, network_kwargs),
    )

    model_cfg = config["model"]
    model_cls = import_from_class_path(model_cfg["class_path"])
    model_kwargs = {
        key: value
        for key, value in model_cfg.get("init_args", {}).items()
        if key != "network"
    }

    config_stuff_classes = config.get("data", {}).get("init_args", {}).get("stuff_classes")
    if stuff_classes is not None:
        model_kwargs["stuff_classes"] = stuff_classes
    elif config_stuff_classes is not None:
        model_kwargs["stuff_classes"] = config_stuff_classes

    model_kwargs["load_ckpt_class_head"] = load_ckpt_class_head
    model_kwargs = filter_kwargs(model_cls, model_kwargs)
    model = model_cls(
        network=network,
        img_size=resolved_img_size,
        num_classes=resolved_num_classes,
        **model_kwargs,
    )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: missing checkpoint keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Warning: unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")

    return model.eval().to(device), device


def dataset_image_paths(dataset_root: Path, dataset: str, input_glob: str | None) -> list[Path]:
    if input_glob:
        return sorted(Path(path) for path in glob.glob(input_glob) if Path(path).is_file())

    image_dir = dataset_root / dataset / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in allowed_suffixes)


def infer_ground_truth_path(image_path: Path) -> Path:
    path = str(image_path).replace("images", "labels_masks")
    if "RoadObsticle21" in path:
        path = path.replace(".webp", ".png")
    if "fs_static" in path:
        path = path.replace(".jpg", ".png").replace(".jpeg", ".png")
    if "RoadAnomaly" in path:
        path = path.replace(".jpg", ".png").replace(".jpeg", ".png")
    return Path(path)


def load_rgb_image(path: Path):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1)


def resize_mask_if_needed(mask, shape: tuple[int, int]):
    import numpy as np
    from PIL import Image

    if mask.shape == shape:
        return mask

    pil_mask = Image.fromarray(mask.astype(np.uint8))
    resized = pil_mask.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(resized)


def prepare_ground_truth_mask(mask_path: Path, target_shape: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if target_shape is not None:
        mask = resize_mask_if_needed(mask, target_shape)

    path_text = str(mask_path)
    if "RoadAnomaly" in path_text:
        mask = np.where(mask == 2, 1, mask)

    if "FS_LostFound_full" in path_text:
        return mask

    if "LostAndFound" in path_text:
        mask = np.where(mask == 0, IGNORE_LABEL, mask)
        mask = np.where(mask == 1, 0, mask)
        mask = np.where((mask > 1) & (mask < 201), 1, mask)

    if "Streethazard" in path_text:
        mask = np.where(mask == 14, IGNORE_LABEL, mask)
        mask = np.where(mask < 20, 0, mask)
        mask = np.where(mask == IGNORE_LABEL, 1, mask)

    return mask


def fpr_at_95_tpr(scores, labels) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    if np.max(tpr) < 0.95:
        return 1.0
    return float(fpr[np.argmax(tpr >= 0.95)])


def compute_anomaly_score(pixel_logits, method: str, temperature: float):
    import torch

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if pixel_logits.dim() != 3:
        raise ValueError(f"Expected pixel logits with shape [C,H,W], got {tuple(pixel_logits.shape)}")

    if method == "msp":
        probs = torch.softmax(pixel_logits / temperature, dim=0)
        return 1.0 - probs.max(dim=0).values

    if method == "maxlogit":
        return -pixel_logits.max(dim=0).values

    if method == "entropy":
        probs = torch.softmax(pixel_logits / temperature, dim=0)
        return -(probs * torch.log(probs + 1e-12)).sum(dim=0)

    raise ValueError(f"Unsupported pixel-logit anomaly method: {method}")


def compute_rba_crop_scores(mask_logits, class_logits):
    import torch

    mask_probs = mask_logits.sigmoid()
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    known_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs).clamp(0.0, 1.0)
    return torch.exp(torch.log1p(-known_scores.clamp(max=1.0 - 1e-6)).sum(dim=1, keepdim=True))


def infer_eomt_outputs(model, image, method: str, device, temperature: float):
    import torch
    import torch.nn.functional as functional

    autocast_context = (
        torch.amp.autocast(device_type="cuda", enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        imgs = [image.to(device)]
        img_sizes = [image.shape[-2:] for image in imgs]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = functional.interpolate(
            mask_logits_per_layer[-1],
            model.img_size,
            mode="bilinear",
        )
        class_logits = class_logits_per_layer[-1]

        if method == "rba":
            crop_scores = compute_rba_crop_scores(mask_logits, class_logits)
            scores = model.revert_window_logits_semantic(crop_scores, origins, img_sizes)[0][0]
            return scores.float().cpu(), None

        crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]
        scores = compute_anomaly_score(logits, method, temperature)
        return scores.float().cpu(), logits.float().cpu()


def infer_eomt_all_scores(model, image, device, temperature: float):
    import torch
    import torch.nn.functional as functional

    autocast_context = (
        torch.amp.autocast(device_type="cuda", enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        imgs = [image.to(device)]
        img_sizes = [image.shape[-2:] for image in imgs]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = functional.interpolate(
            mask_logits_per_layer[-1],
            model.img_size,
            mode="bilinear",
        )
        class_logits = class_logits_per_layer[-1]

        crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]

        rba_crop_scores = compute_rba_crop_scores(mask_logits, class_logits)
        rba_scores = model.revert_window_logits_semantic(
            rba_crop_scores,
            origins,
            img_sizes,
        )[0][0]

        return {
            "msp": compute_anomaly_score(logits, "msp", temperature).float().cpu(),
            "maxlogit": compute_anomaly_score(logits, "maxlogit", temperature).float().cpu(),
            "entropy": compute_anomaly_score(logits, "entropy", temperature).float().cpu(),
            "rba": rba_scores.float().cpu(),
        }, logits.float().cpu()


def save_logits(path: Path, image_path: Path, logits) -> None:
    if logits is None:
        return
    import numpy as np

    path.mkdir(parents=True, exist_ok=True)
    safe_name = image_path.stem.replace(" ", "_")
    np.savez_compressed(path / f"{safe_name}_pixel_logits.npz", pixel_logits=logits.numpy())


def collect_scores_for_dataset(
    *,
    model,
    device,
    image_paths: Iterable[Path],
    method: str,
    temperature: float,
    max_images: int | None,
    logits_dir: Path | None,
):
    import numpy as np

    score_parts = []
    label_parts = []
    processed = 0
    num_ood_pixels = 0
    num_ind_pixels = 0

    for image_path in image_paths:
        if max_images is not None and processed >= max_images:
            break

        gt_path = infer_ground_truth_path(image_path)
        if not gt_path.exists():
            print(f"Warning: missing ground-truth mask, skipping: {gt_path}")
            continue

        image = load_rgb_image(image_path)
        scores, logits = infer_eomt_outputs(model, image, method, device, temperature)
        score_array = scores.numpy()
        gt = prepare_ground_truth_mask(gt_path, target_shape=score_array.shape)

        if 1 not in np.unique(gt):
            continue

        ind_mask = gt == 0
        ood_mask = gt == 1
        ind_scores = score_array[ind_mask]
        ood_scores = score_array[ood_mask]

        if ind_scores.size == 0 or ood_scores.size == 0:
            continue

        score_parts.extend([ind_scores, ood_scores])
        label_parts.extend(
            [
                np.zeros(ind_scores.shape[0], dtype=np.uint8),
                np.ones(ood_scores.shape[0], dtype=np.uint8),
            ]
        )
        num_ind_pixels += int(ind_scores.shape[0])
        num_ood_pixels += int(ood_scores.shape[0])
        processed += 1

        if logits_dir is not None:
            save_logits(logits_dir, image_path, logits)

        print(f"[{processed}] processed {image_path.name}")

    if not score_parts:
        raise RuntimeError("No valid images with anomaly and in-distribution pixels were processed.")

    return (
        np.concatenate(score_parts),
        np.concatenate(label_parts),
        processed,
        num_ood_pixels,
        num_ind_pixels,
    )


def collect_all_scores_for_dataset(
    *,
    model,
    device,
    image_paths: Iterable[Path],
    temperature: float,
    max_images: int | None,
    logits_dir: Path | None,
):
    import numpy as np

    score_parts = {method: [] for method in DEFAULT_METHODS}
    label_parts = []
    processed = 0
    num_ood_pixels = 0
    num_ind_pixels = 0

    for image_path in image_paths:
        if max_images is not None and processed >= max_images:
            break

        gt_path = infer_ground_truth_path(image_path)
        if not gt_path.exists():
            print(f"Warning: missing ground-truth mask, skipping: {gt_path}")
            continue

        image = load_rgb_image(image_path)
        scores_by_method, logits = infer_eomt_all_scores(model, image, device, temperature)
        reference_shape = scores_by_method["maxlogit"].shape
        gt = prepare_ground_truth_mask(gt_path, target_shape=reference_shape)

        if 1 not in np.unique(gt):
            continue

        ind_mask = gt == 0
        ood_mask = gt == 1
        if not np.any(ind_mask) or not np.any(ood_mask):
            continue

        for method, scores in scores_by_method.items():
            score_array = scores.numpy()
            score_parts[method].extend([score_array[ind_mask], score_array[ood_mask]])

        ind_count = int(np.count_nonzero(ind_mask))
        ood_count = int(np.count_nonzero(ood_mask))
        label_parts.extend(
            [
                np.zeros(ind_count, dtype=np.uint8),
                np.ones(ood_count, dtype=np.uint8),
            ]
        )
        num_ind_pixels += ind_count
        num_ood_pixels += ood_count
        processed += 1

        if logits_dir is not None:
            save_logits(logits_dir, image_path, logits)

        print(f"[{processed}] processed {image_path.name}")

    if not label_parts:
        raise RuntimeError("No valid images with anomaly and in-distribution pixels were processed.")

    labels = np.concatenate(label_parts)
    return (
        {method: np.concatenate(parts) for method, parts in score_parts.items()},
        labels,
        processed,
        num_ood_pixels,
        num_ind_pixels,
    )


def evaluate_dataset(
    *,
    model,
    device,
    dataset_root: Path,
    dataset: str,
    input_glob: str | None,
    method: str,
    temperature: float,
    checkpoint_name: str,
    miou: str | float,
    max_images: int | None,
    logits_dir: Path | None,
) -> DatasetResult:
    from sklearn.metrics import average_precision_score

    image_paths = dataset_image_paths(dataset_root, dataset, input_glob)
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset {dataset}.")

    scores, labels, processed, num_ood, num_ind = collect_scores_for_dataset(
        model=model,
        device=device,
        image_paths=image_paths,
        method=method,
        temperature=temperature,
        max_images=max_images,
        logits_dir=logits_dir,
    )

    auprc = float(average_precision_score(labels, scores) * 100.0)
    fpr95 = float(fpr_at_95_tpr(scores, labels) * 100.0)
    method_label = method
    if method in {"msp", "entropy"} and temperature != 1.0:
        method_label = f"{method}_t{temperature:g}"

    return DatasetResult(
        checkpoint=checkpoint_name,
        dataset=dataset,
        method=method_label,
        temperature=temperature,
        auprc=auprc,
        fpr95=fpr95,
        num_images=processed,
        num_ood_pixels=num_ood,
        num_ind_pixels=num_ind,
        miou=miou,
    )


def evaluate_dataset_all_methods(
    *,
    model,
    device,
    dataset_root: Path,
    dataset: str,
    input_glob: str | None,
    temperature: float,
    checkpoint_name: str,
    miou: str | float,
    max_images: int | None,
    logits_dir: Path | None,
) -> list[DatasetResult]:
    from sklearn.metrics import average_precision_score

    image_paths = dataset_image_paths(dataset_root, dataset, input_glob)
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset {dataset}.")

    scores_by_method, labels, processed, num_ood, num_ind = collect_all_scores_for_dataset(
        model=model,
        device=device,
        image_paths=image_paths,
        temperature=temperature,
        max_images=max_images,
        logits_dir=logits_dir,
    )

    results = []
    for method in DEFAULT_METHODS:
        scores = scores_by_method[method]
        method_label = method
        if method in {"msp", "entropy"} and temperature != 1.0:
            method_label = f"{method}_t{temperature:g}"
        results.append(
            DatasetResult(
                checkpoint=checkpoint_name,
                dataset=dataset,
                method=method_label,
                temperature=temperature,
                auprc=float(average_precision_score(labels, scores) * 100.0),
                fpr95=float(fpr_at_95_tpr(scores, labels) * 100.0),
                num_images=processed,
                num_ood_pixels=num_ood,
                num_ind_pixels=num_ind,
                miou=miou,
            )
        )

    return results


def collect_temperature_scores_for_dataset(
    *,
    model,
    device,
    image_paths: Iterable[Path],
    method: str,
    temperatures: list[float],
    max_images: int | None,
    logits_dir: Path | None,
):
    import numpy as np

    if method not in {"msp", "entropy"}:
        raise ValueError("--temperatures is only supported for MSP or entropy scores.")

    score_parts = {temperature: [] for temperature in temperatures}
    label_parts = []
    processed = 0
    num_ood_pixels = 0
    num_ind_pixels = 0

    for image_path in image_paths:
        if max_images is not None and processed >= max_images:
            break

        gt_path = infer_ground_truth_path(image_path)
        if not gt_path.exists():
            print(f"Warning: missing ground-truth mask, skipping: {gt_path}")
            continue

        image = load_rgb_image(image_path)
        _, logits = infer_eomt_outputs(model, image, method, device, temperature=1.0)
        if logits is None:
            raise RuntimeError("Temperature scaling requires saved pixel logits.")

        reference_shape = logits.shape[-2:]
        gt = prepare_ground_truth_mask(gt_path, target_shape=reference_shape)

        if 1 not in np.unique(gt):
            continue

        ind_mask = gt == 0
        ood_mask = gt == 1
        if not np.any(ind_mask) or not np.any(ood_mask):
            continue

        for temperature in temperatures:
            score_array = compute_anomaly_score(logits, method, temperature).numpy()
            score_parts[temperature].extend([score_array[ind_mask], score_array[ood_mask]])

        ind_count = int(np.count_nonzero(ind_mask))
        ood_count = int(np.count_nonzero(ood_mask))
        label_parts.extend(
            [
                np.zeros(ind_count, dtype=np.uint8),
                np.ones(ood_count, dtype=np.uint8),
            ]
        )
        num_ind_pixels += ind_count
        num_ood_pixels += ood_count
        processed += 1

        if logits_dir is not None:
            save_logits(logits_dir, image_path, logits)

        print(f"[{processed}] processed {image_path.name}")

    if not label_parts:
        raise RuntimeError("No valid images with anomaly and in-distribution pixels were processed.")

    labels = np.concatenate(label_parts)
    return (
        {temperature: np.concatenate(parts) for temperature, parts in score_parts.items()},
        labels,
        processed,
        num_ood_pixels,
        num_ind_pixels,
    )


def evaluate_dataset_temperatures(
    *,
    model,
    device,
    dataset_root: Path,
    dataset: str,
    input_glob: str | None,
    method: str,
    temperatures: list[float],
    checkpoint_name: str,
    miou: str | float,
    max_images: int | None,
    logits_dir: Path | None,
) -> list[DatasetResult]:
    from sklearn.metrics import average_precision_score

    image_paths = dataset_image_paths(dataset_root, dataset, input_glob)
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset {dataset}.")

    scores_by_temperature, labels, processed, num_ood, num_ind = collect_temperature_scores_for_dataset(
        model=model,
        device=device,
        image_paths=image_paths,
        method=method,
        temperatures=temperatures,
        max_images=max_images,
        logits_dir=logits_dir,
    )

    results = []
    for temperature in temperatures:
        scores = scores_by_temperature[temperature]
        method_label = method if temperature == 1.0 else f"{method}_t{temperature:g}"
        results.append(
            DatasetResult(
                checkpoint=checkpoint_name,
                dataset=dataset,
                method=method_label,
                temperature=temperature,
                auprc=float(average_precision_score(labels, scores) * 100.0),
                fpr95=float(fpr_at_95_tpr(scores, labels) * 100.0),
                num_images=processed,
                num_ood_pixels=num_ood,
                num_ind_pixels=num_ind,
                miou=miou,
            )
        )

    return results


def append_result(path: Path, result: DatasetResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint",
        "dataset",
        "method",
        "temperature",
        "miou",
        "auprc",
        "fpr95",
        "num_images",
        "num_ood_pixels",
        "num_ind_pixels",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "model": "EoMT",
                "checkpoint": result.checkpoint,
                "dataset": result.dataset,
                "method": result.method,
                "temperature": result.temperature,
                "miou": result.miou,
                "auprc": result.auprc,
                "fpr95": result.fpr95,
                "num_images": result.num_images,
                "num_ood_pixels": result.num_ood_pixels,
                "num_ind_pixels": result.num_ind_pixels,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--config", required=True, type=Path, help="EoMT YAML config path.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local EoMT checkpoint path.")
    parser.add_argument("--checkpoint-name", required=True, help="Short checkpoint label for CSV rows.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root containing anomaly dataset folders.")
    parser.add_argument("--dataset", required=True, choices=DEFAULT_DATASETS)
    parser.add_argument("--input-glob", help="Optional explicit image glob. Overrides --dataset-root/--dataset discovery.")
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperatures",
        type=parse_float_list,
        help="Comma-separated MSP/entropy temperatures to evaluate from one forward pass, for example 0.5,0.75,1.0,1.1.",
    )
    parser.add_argument("--miou", default="", help="Optional mIoU value to repeat in the output row.")
    parser.add_argument("--output-csv", default=Path("step8_eomt_mask_baselines/eomt_anomaly_results.csv"), type=Path)
    parser.add_argument("--img-size", type=parse_img_size, help="EoMT inference image size as HxW. Defaults to checkpoint shape.")
    parser.add_argument("--num-classes", type=int, help="Number of model classes. Defaults to checkpoint class head if present.")
    parser.add_argument("--stuff-classes", help="Comma-separated panoptic stuff classes if not present in config.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--max-images", type=int, help="Limit images for smoke tests.")
    parser.add_argument("--save-logits", action="store_true", help="Save per-image pixel logits for non-RbA methods.")
    parser.add_argument("--logits-dir", type=Path, default=Path("step8_eomt_mask_baselines/saved_logits"))
    parser.add_argument(
        "--load-ckpt-class-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load the checkpoint class head when constructing compatible EoMT modules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, device = build_eomt_model(
        repo_root=args.repo_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        img_size=args.img_size,
        num_classes=args.num_classes,
        stuff_classes=parse_int_list(args.stuff_classes),
        load_ckpt_class_head=args.load_ckpt_class_head,
    )

    logits_dir = args.logits_dir if args.save_logits else None
    if args.temperatures is not None:
        if args.method == "all":
            raise ValueError("--temperatures cannot be combined with --method all.")
        results = evaluate_dataset_temperatures(
            model=model,
            device=device,
            dataset_root=args.dataset_root,
            dataset=args.dataset,
            input_glob=args.input_glob,
            method=args.method,
            temperatures=args.temperatures,
            checkpoint_name=args.checkpoint_name,
            miou=args.miou,
            max_images=args.max_images,
            logits_dir=logits_dir,
        )
    elif args.method == "all":
        results = evaluate_dataset_all_methods(
            model=model,
            device=device,
            dataset_root=args.dataset_root,
            dataset=args.dataset,
            input_glob=args.input_glob,
            temperature=args.temperature,
            checkpoint_name=args.checkpoint_name,
            miou=args.miou,
            max_images=args.max_images,
            logits_dir=logits_dir,
        )
    else:
        results = [
            evaluate_dataset(
                model=model,
                device=device,
                dataset_root=args.dataset_root,
                dataset=args.dataset,
                input_glob=args.input_glob,
                method=args.method,
                temperature=args.temperature,
                checkpoint_name=args.checkpoint_name,
                miou=args.miou,
                max_images=args.max_images,
                logits_dir=logits_dir,
            )
        ]

    for result in results:
        append_result(args.output_csv, result)
        print(
            f"Saved {result.checkpoint} {result.dataset} {result.method}: "
            f"AuPRC={result.auprc:.4f}, FPR95={result.fpr95:.4f}"
        )


if __name__ == "__main__":
    main()
