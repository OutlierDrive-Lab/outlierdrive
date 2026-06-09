# Step 7 - ERFNet Pixel-Based Anomaly Baselines

This folder contains the ERFNet anomaly-segmentation experiments for Step 7.
The original evaluator in `eval/evalAnomaly.py` is left unchanged; the
experiments use a separate runner that also works on CPU.

## Methods

The runner converts ERFNet class logits into a pixel-level anomaly score:

- **MSP:** `1 - max(softmax(logits))`
- **MaxLogit:** negative maximum class logit
- **Max Entropy:** entropy of the softmax distribution

Higher values indicate that a pixel is more likely to be anomalous.

## Setup

Install the project dependencies and place the pretrained ERFNet checkpoint in
`trained_models/`:

```text
trained_models/
└── erfnet_pretrained.pth
```

The anomaly datasets should contain matching image and mask folders:

```text
Validation_Dataset/
└── RoadAnomaly21/
    ├── images/
    └── labels_masks/
```

## Run an Experiment

From the repository root:

```bash
python step7_erfnet_pixel_baselines/run_erfnet_anomaly_cpu.py \
  --input "Validation_Dataset/RoadAnomaly21/images/*.png" \
  --loadDir trained_models \
  --loadModel erfnet.py \
  --loadWeights erfnet_pretrained.pth \
  --dataset-name RoadAnomaly21 \
  --method maxlogit \
  --cpu
```

Available methods are `msp`, `maxlogit`, and `entropy`. Remove `--cpu` to use
CUDA when it is available.

## Output

Each run appends one row to
`step7_erfnet_pixel_baselines/erfnet_anomaly_results.csv`:

```csv
model,dataset,method,auprc,fpr95
```

The committed table contains results for five validation datasets:

- RoadAnomaly21
- RoadAnomaly
- RoadObsticle21
- fs_static
- FS_LostFound_full

MaxLogit gives the best AuPRC in the current results for all five datasets.
The folder name `RoadObsticle21` follows the spelling used in the provided
dataset archive.
