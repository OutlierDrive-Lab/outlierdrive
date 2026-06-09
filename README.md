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

`feature/erfnet-temperature-scaling` extends the Step 7 ERFNet baselines with
temperature scaling. The runner applies the selected temperature before the
softmax used by MSP and Max Entropy.

Instructions and result tables are in
[`step7_erfnet_pixel_baselines`](step7_erfnet_pixel_baselines).

## Main Folders

- [`eval`](eval): original ERFNet evaluation code
- [`trained_models`](trained_models): ERFNet checkpoint files
- [`eomt`](eomt): EoMT training and inference code
- [`step7_erfnet_pixel_baselines`](step7_erfnet_pixel_baselines): ERFNet anomaly
  baselines and temperature-scaling experiments

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
