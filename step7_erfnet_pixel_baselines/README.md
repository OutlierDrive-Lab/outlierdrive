# Step 7 — ERFNet Pixel-Based Anomaly Baselines

This folder contains our implementation for Step 7 of the project: pixel-based anomaly segmentation baselines using a pretrained ERFNet model.

## Goal

The goal of this step is to evaluate a pretrained ERFNet semantic segmentation model on anomaly segmentation datasets using post-hoc anomaly scoring methods.

## Original files

The original provided file:

`eval/evalAnomaly.py`

is kept unchanged.

Our implementation is provided as a separate runner:

`step7_erfnet_pixel_baselines/run_erfnet_anomaly_cpu.py`

## Methods

The runner supports the three required post-hoc anomaly scoring methods:

- MSP
- MaxLogit
- Max Entropy

## Datasets

The evaluation was run on the anomaly validation datasets:

- RoadAnomaly21
- RoadAnomaly
- RoadObsticle21
- fs_static
- FS_LostFound_full

These correspond to the required anomaly benchmarks in the project.

## Metrics

The reported metrics are:

- AuPRC
- FPR95

## Results

The results are saved in:

`step7_erfnet_pixel_baselines/erfnet_anomaly_results.csv`

In the current ERFNet experiments, MaxLogit performs better than MSP and Max Entropy on most datasets.
