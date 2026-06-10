# Anomaly Segmentation

This project studies anomaly segmentation for autonomous-driving scenes. It
compares a pixel-based ERFNet model with the mask-based EoMT architecture and
evaluates their ability to detect objects that are outside the training
distribution.

## Project Stages

- **Step 4 - EoMT evaluation:** compare COCO-trained and Cityscapes-trained EoMT
  checkpoints on Cityscapes.
- **Step 5 - EoMT fine-tuning:** fine-tune the COCO checkpoint on Cityscapes and
  evaluate the resulting semantic segmentation models.
- **Step 7 - ERFNet baselines:** evaluate MSP, MaxLogit, and Max Entropy on the
  anomaly validation datasets.
- **Step 8 - EoMT baselines:** evaluate MSP, MaxLogit, Max Entropy, and RbA for
  three EoMT checkpoints.

## Repository Structure

- [`eomt`](eomt): EoMT model, training code, configurations, and dependencies
- [`eval`](eval): original ERFNet evaluation code
- [`step4_eomt_eval`](step4_eomt_eval): EoMT visualization and Cityscapes
  evaluation pipeline
- [`step5_fine_tuning`](step5_fine_tuning): fine-tuning notebook and mIoU
  results
- [`step7_erfnet_pixel_baselines`](step7_erfnet_pixel_baselines): ERFNet anomaly
  baselines and temperature-scaling results
- [`step8_eomt_mask_baselines`](step8_eomt_mask_baselines): EoMT anomaly
  baselines, temperature scaling, and report-ready results

Each stage contains its own README with setup and execution details.

## Models

- ERFNet
- EoMT with a DINOv2 backbone
- COCO-trained EoMT
- Cityscapes-trained EoMT
- EoMT fine-tuned on Cityscapes

## Datasets

- Cityscapes
- SegmentMeIfYouCan RoadAnomaly21
- SegmentMeIfYouCan RoadObstacle21
- Fishyscapes Lost & Found
- Fishyscapes Static
- Road Anomaly

Datasets and EoMT checkpoints are kept locally and are not committed to the
repository.

## Evaluation

Semantic segmentation is evaluated with:

- mIoU
- Pixel accuracy

Anomaly segmentation is evaluated with:

- AuPRC
- FPR95

The anomaly scoring methods used in the project are MSP, MaxLogit, Max Entropy,
RbA, and temperature-scaled MSP.

## Environment

The root dependencies are listed in `requirements.txt` and `environment.yml`.
EoMT-specific dependencies are listed in `eomt/requirements.txt`.
