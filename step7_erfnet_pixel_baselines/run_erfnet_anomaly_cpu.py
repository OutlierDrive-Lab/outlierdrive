# ERFNet anomaly segmentation evaluation runner.
#
# This script is based on the evaluation logic provided in eval/evalAnomaly.py,
# but it is placed under scripts/ as our own runner so that the original
# provided file remains unchanged.
#
# Main features:
# 1. CPU/Mac compatible execution.
# 2. No dependency on ood_metrics.
# 3. Local implementation of FPR@95TPR using sklearn.
# 4. Supports MSP, MaxLogit, and Max Entropy anomaly scoring.
# 5. Saves results to a CSV file.

import os
import sys
import glob
import torch
import random
from PIL import Image
import numpy as np
from argparse import ArgumentParser

# The original ERFNet implementation is inside eval/erfnet.py.
# Since this script is inside scripts/, we add eval/ to Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eval"))
from erfnet import ERFNet

from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def fpr_at_95_tpr(scores, labels):
    """
    Compute FPR when TPR reaches at least 95%.

    Args:
        scores: anomaly scores, higher means more anomalous.
        labels: binary labels, 1 = anomaly/OOD, 0 = in-distribution.
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    if np.max(tpr) < 0.95:
        return 1.0

    return float(fpr[np.argmax(tpr >= 0.95)])


# ERFNet returns per-class pixel scores.
# For anomaly segmentation, we do not train a new anomaly class.
# Instead, post-hoc methods convert the model output into one anomaly score per pixel.
# Higher anomaly score means the pixel is more likely to be out-of-distribution.
def compute_anomaly_score(logits, method, temperature=1.0):
    """
    Compute pixel-wise anomaly scores from ERFNet outputs.

    Args:
        logits:
            torch.Tensor with shape [1, C, H, W].
        method:
            One of: msp, maxlogit, entropy.
        temperature:
            Softmax temperature used for MSP and Entropy.
            Logits are divided by this value before softmax.

    Returns:
        np.ndarray with shape [H, W].
        Higher score means more anomalous.
    """
    if logits.dim() != 4:
        raise ValueError(f"Expected logits with shape [B, C, H, W], got {logits.shape}")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    scaled_logits = logits / temperature

    if method == "msp":
        # MSP uses calibrated softmax probabilities.
        # Lower confidence means higher anomaly score.
        probs = torch.softmax(scaled_logits, dim=1)
        score = 1.0 - torch.max(probs, dim=1).values

    elif method == "maxlogit":
        # MaxLogit works directly on raw logits, so we keep it independent
        # from temperature scaling.
        score = -torch.max(logits, dim=1).values

    elif method == "entropy":
        # Entropy measures uncertainty in the softmax distribution.
        # Higher entropy means the model is more uncertain.
        probs = torch.softmax(scaled_logits, dim=1)
        score = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

    else:
        raise ValueError(f"Unknown anomaly scoring method: {method}")

    return score.squeeze(0).detach().cpu().numpy()


def load_my_state_dict(model, state_dict):
    """
    Load ERFNet weights even when some keys contain the 'module.' prefix
    from DataParallel training.
    """
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                clean_name = name.split("module.")[-1]
                if clean_name in own_state:
                    own_state[clean_name].copy_(param)
                else:
                    print(name, "not loaded")
            else:
                print(name, "not loaded")
                continue
        else:
            own_state[name].copy_(param)

    return model


# The anomaly datasets do not all use the same label encoding.
# This helper normalizes the masks into one common binary convention:
# 1 = anomaly/OOD pixel, 0 = normal in-distribution pixel, 255 = ignored pixel.
def prepare_ground_truth_mask(path_gt):
    """
    Load and convert the anomaly ground-truth mask into binary format.

    Output convention:
        1 = anomaly/OOD
        0 = in-distribution
        255 = ignored pixels
    """
    mask = Image.open(path_gt)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)

    if "LostAndFound" in path_gt:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

    if "Streethazard" in path_gt:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts


# Each anomaly image has a corresponding mask in labels_masks/.
# Some datasets store images as jpg/webp while the masks are png, so we fix
# the extension here before loading the ground-truth mask.
def infer_ground_truth_path(image_path):
    """
    Infer the corresponding ground-truth mask path from the image path.

    The anomaly datasets use:
        images/
        labels_masks/
    """
    path_gt = image_path.replace("images", "labels_masks")

    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")

    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")

    if "RoadAnomaly" in path_gt:
        path_gt = path_gt.replace("jpg", "png")

    return path_gt


# Store every run as a CSV row so results from multiple datasets and methods
# can be directly used in the final project table.
def append_result_to_csv(output_csv, dataset_name, method, auprc, fpr95):
    """
    Append one experiment result to a CSV file.
    """
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a") as csv_file:
        if write_header:
            csv_file.write("model,dataset,method,auprc,fpr95\n")

        csv_file.write(
            f"ERFNet,{dataset_name},{method},{auprc},{fpr95}\n"
        )


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input image glob pattern, e.g. data/anomaly/.../images/*.png",
    )

    parser.add_argument("--loadDir", default="trained_models/")
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--loadModel", default="erfnet.py")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--method",
        default="maxlogit",
        choices=["msp", "maxlogit", "entropy"],
        help="Post-hoc anomaly scoring method.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used for MSP/Entropy confidence calibration.",
    )

    parser.add_argument(
        "--dataset-name",
        default="unknown",
        help="Name of the evaluated anomaly dataset.",
    )

    parser.add_argument(
        "--output-csv",
        default="step7_erfnet_pixel_baselines/erfnet_anomaly_results.csv",
        help="CSV file where results will be appended.",
    )

    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    print(f"Using anomaly scoring method: {args.method}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Using temperature: {args.temperature}")

    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print("Loading model:", modelpath)
    print("Loading weights:", weightspath)

    model = ERFNet(NUM_CLASSES)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    state_dict = torch.load(weightspath, map_location=device)
    model = load_my_state_dict(model, state_dict)

    print("Model and weights LOADED successfully")

    model.eval()

    input_paths = sorted(glob.glob(os.path.expanduser(str(args.input[0]))))

    if len(input_paths) == 0:
        raise FileNotFoundError(f"No input images found for pattern: {args.input[0]}")

    print(f"Found {len(input_paths)} input images.")

    for path in input_paths:
        print(path)

        image = Image.open(path).convert("RGB")
        images = input_transform(image).unsqueeze(0).float().to(device)

        with torch.no_grad():
            result = model(images)

        anomaly_result = compute_anomaly_score(
            result,
            args.method,
            temperature=args.temperature,
        )

        path_gt = infer_ground_truth_path(path)

        if not os.path.exists(path_gt):
            print(f"Warning: ground-truth mask not found, skipping: {path_gt}")
            continue

        ood_gts = prepare_ground_truth_mask(path_gt)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        del result, anomaly_result, ood_gts

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        raise RuntimeError("No valid anomaly ground-truth masks were collected.")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # OOD pixels are positives, normal pixels are negatives.
    # Ignore pixels are excluded because they are neither OOD nor in-distribution.
    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    # AuPRC is important for anomaly segmentation because anomaly pixels are rare.
    # FPR95 measures the false-positive rate when 95% of anomaly pixels are detected.
    prc_auc = average_precision_score(val_label, val_out)
    fpr95 = fpr_at_95_tpr(val_out, val_label)

    auprc_percent = prc_auc * 100.0
    fpr95_percent = fpr95 * 100.0

    print(f"AUPRC score: {auprc_percent}")
    print(f"FPR@TPR95: {fpr95_percent}")

    # Store every run as a CSV row so results from multiple datasets and methods
    # can be directly used in the final project table.
    method_label = args.method

    if args.method in ["msp", "entropy"] and args.temperature != 1.0:
        method_label = f"{args.method}_t{args.temperature}"

    # Store every run as a CSV row so results from multiple datasets and methods
    # can be directly used in the final project table.
    append_result_to_csv(
        output_csv=args.output_csv,
        dataset_name=args.dataset_name,
        method=method_label,
        auprc=auprc_percent,
        fpr95=fpr95_percent,
    )

    print(f"Result appended to: {args.output_csv}")


if __name__ == "__main__":
    main()
