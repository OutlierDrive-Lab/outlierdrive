# OutlierDrive:  Road Anomaly Segmentation

OutlierDrive is a research oriented computer vision project focused on anomaly segmentation for autonomous driving scenes.  
The project compares pixel based and mask based segmentation models for detecting unknown objects in road environments.

## Goals

- Study semantic, instance, and panoptic segmentation
- Compare ERFNet and EoMT on road-scene understanding
- Evaluate post-hoc anomaly segmentation methods
- Fine-tune a COCO-pretrained EoMT model on Cityscapes
- Analyze MSP, MaxLogit, Max Entropy, RbA, and temperature scaling

## Models

- ERFNet
- EoMT
- MaskFormer / Mask2Former concepts
- DINOv2-based segmentation architecture

## Datasets

- Cityscapes
- SegmentMeIfYouCan
- Fishyscapes
- Road Anomaly

## Evaluation

Semantic segmentation:
- mIoU

Anomaly segmentation:
- AuPRC
- FPR95

## Team Workflow

Development is organized through feature branches and pull requests.

Main branches:
- `feature/data-evaluation-pipeline`
- `feature/erfnet-baselines`
- `feature/eomt-mask-baselines`
- `feature/finetuning-report`

