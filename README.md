# OutlierDrive: Open-World Road Anomaly Segmentation

OutlierDrive studies anomaly segmentation for autonomous-driving scenes. The
project compares pixel-based and mask-based models and evaluates how well they
detect objects that are outside the training distribution.

## Project Scope

- ERFNet semantic segmentation and pixel-based anomaly scores
- EoMT mask-based segmentation
- Fine-tuning on Cityscapes
- Evaluation on road-anomaly datasets
- MSP, MaxLogit, Max Entropy, RbA, and temperature scaling

## This Branch

`feature/eomt-mask-baselines` is intended for the Step 8 EoMT anomaly
experiments. The goal is to evaluate mask-based predictions from COCO-trained,
Cityscapes-trained, and fine-tuned EoMT checkpoints on the same datasets used
for the ERFNet baselines.

The expected methods are:

- MSP
- MaxLogit
- Max Entropy
- RbA-style mask rejection

AuPRC and FPR95 are used for anomaly segmentation, while Cityscapes mIoU is
reported separately for the semantic quality of each checkpoint.

## Branch Status

This branch currently contains the base project code but does not contain the
Step 8 runner or result tables. Those files are available in
`origin/feature/step8-eomt-mask-baselines` under:

```text
step8_eomt_mask_baselines/
```

The Step 8 implementation should be merged or cherry-picked before documenting
commands that refer to that folder.

## Main Folders

- [`eval`](eval): ERFNet evaluation and anomaly-segmentation code
- [`trained_models`](trained_models): ERFNet checkpoint files
- [`eomt`](eomt): EoMT training and inference code

## Required Data

- Cityscapes
- RoadAnomaly21
- RoadAnomaly
- RoadObsticle21
- Fishyscapes Static
- Fishyscapes Lost & Found

Datasets and EoMT checkpoint files are kept locally and are not committed to the
repository.
