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

`feature/erfnet-baselines` contains the Step 7 ERFNet experiments. It evaluates
a pretrained ERFNet checkpoint with MSP, MaxLogit, and Max Entropy and reports
AuPRC and FPR95.

The runner, usage instructions, and saved results are in
[`step7_erfnet_pixel_baselines`](step7_erfnet_pixel_baselines).

## Main Folders

- [`eval`](eval): original ERFNet evaluation code
- [`trained_models`](trained_models): ERFNet checkpoint files
- [`eomt`](eomt): EoMT training and inference code
- [`step7_erfnet_pixel_baselines`](step7_erfnet_pixel_baselines): ERFNet anomaly
  baseline runner and result table

## Datasets

- Cityscapes
- RoadAnomaly21
- RoadAnomaly
- RoadObsticle21
- Fishyscapes Static
- Fishyscapes Lost & Found

## Metrics

- Semantic segmentation: mIoU
- Anomaly segmentation: AuPRC and FPR95
