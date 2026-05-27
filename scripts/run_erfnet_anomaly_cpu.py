# ERFNet anomaly segmentation evaluation runner.
#
# This script is based on the evaluation logic provided in eval/evalAnomaly.py,
# but it is placed under scripts/ as our own runner so that the original
# provided file remains unchanged.
#
# Main changes compared to the provided evalAnomaly.py:
# 1. CPU/Mac compatible execution.
# 2. No direct dependency on ood_metrics.
# 3. Local implementation of FPR@95TPR using sklearn.
# 4. Correct tensor handling for torchvision ToTensor().
# 5. Safer device handling.

import os
import sys
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser

# The original ERFNet implementation is inside eval/erfnet.py.
# Since this script is inside scripts/, we add eval/ to Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eval"))
from erfnet import ERFNet

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


seed = 42

# General reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

# These settings are relevant only when CUDA is available.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # The original provided script does not normalize the input.
        # We keep the same behavior for consistency.
        # Normalize([.485, .456, .406], [.229, .224, .225]),
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
        scores:
            1D numpy array of anomaly scores.
            Higher score means more anomalous.
        labels:
            1D numpy array of binary labels.
            1 = anomaly/OOD, 0 = in-distribution.

    Returns:
        float:
            False Positive Rate at 95% True Positive Rate.
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    if np.max(tpr) < 0.95:
        return 1.0

    return float(fpr[np.argmax(tpr >= 0.95)])


def load_my_state_dict(model, state_dict):
    """
    Custom function to load model weights even when some keys
    contain the 'module.' prefix from DataParallel training.
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


def prepare_ground_truth_mask(path_gt):
    """
    Load and convert the anomaly ground-truth mask into a binary format.

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


def infer_ground_truth_path(image_path):
    """
    Infer the corresponding ground-truth mask path from the image path.

    The provided anomaly datasets use:
        images/
        labels_masks/

    Some datasets use different image extensions, while labels are PNG.
    """
    path_gt = image_path.replace("images", "labels_masks")

    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")

    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")

    if "RoadAnomaly" in path_gt:
        path_gt = path_gt.replace("jpg", "png")

    return path_gt


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        default="../data/anomaly/Validation_Dataset/RoadAnomaly21/images/*.png",
        nargs="+",
        help=(
            "A list of space separated input images, or a single glob pattern "
            "such as '../data/anomaly/Validation_Dataset/RoadAnomaly21/images/*.png'"
        ),
    )

    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--loadModel", default="erfnet.py")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

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

        # Original scoring logic from the provided script:
        # anomaly score = 1 - max model output over classes.
        #
        # Note:
        # In the next step, we will extend this script to support:
        # MSP, MaxLogit, and Max Entropy explicitly.
        anomaly_result = 1.0 - np.max(result.squeeze(0).detach().cpu().numpy(), axis=0)

        path_gt = infer_ground_truth_path(path)

        if not os.path.exists(path_gt):
            print(f"Warning: ground-truth mask not found, skipping: {path_gt}")
            continue

        ood_gts = prepare_ground_truth_mask(path_gt)

        # Keep only images containing at least one anomaly pixel.
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

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"AUPRC score: {prc_auc * 100.0}")
    print(f"FPR@TPR95: {fpr * 100.0}")

    with open("results.txt", "a") as file:
        file.write("\n")
        file.write(
            "AUPRC score: "
            + str(prc_auc * 100.0)
            + "   FPR@TPR95: "
            + str(fpr * 100.0)
        )


if __name__ == "__main__":
    main()