# Step 7 - ERFNet Baselines and Temperature Scaling

This folder contains the ERFNet anomaly-segmentation experiments for Step 7.
The runner supports the three pixel-based baselines and optional temperature
scaling without changing the original evaluator in `eval/evalAnomaly.py`.

## Methods

ERFNet class logits are converted into a pixel-level anomaly score:

- **MSP:** `1 - max(softmax(logits / T))`
- **MaxLogit:** negative maximum class logit
- **Max Entropy:** entropy of `softmax(logits / T)`

`T` is the temperature. A value of `1.0` gives the original baseline. MaxLogit
uses raw logits and is therefore not affected by temperature scaling.

## Setup

Install the project dependencies and place the pretrained checkpoint at
`trained_models/erfnet_pretrained.pth`.

The anomaly datasets should contain matching image and mask folders:

```text
Validation_Dataset/
└── RoadAnomaly21/
    ├── images/
    └── labels_masks/
```

## Run a Baseline

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

## Run Temperature Scaling

Pass `--temperature` with MSP or entropy:

```bash
python step7_erfnet_pixel_baselines/run_erfnet_anomaly_cpu.py \
  --input "Validation_Dataset/RoadAnomaly21/images/*.png" \
  --loadDir trained_models \
  --loadModel erfnet.py \
  --loadWeights erfnet_pretrained.pth \
  --dataset-name RoadAnomaly21 \
  --method msp \
  --temperature 0.75 \
  --output-csv step7_erfnet_pixel_baselines/erfnet_temperature_results.csv \
  --cpu
```

The committed sweep uses `T = 0.5`, `0.75`, `1.0`, and `1.1` on five anomaly
datasets.

## Results

Baseline results are stored in:

```text
step7_erfnet_pixel_baselines/erfnet_anomaly_results.csv
```

Temperature-scaling results are stored in:

```text
step7_erfnet_pixel_baselines/erfnet_temperature_results.csv
```

Both files use the same columns:

```csv
model,dataset,method,auprc,fpr95
```

For scaled runs, the method name includes the temperature, for example
`msp_t0.75`. The folder name `RoadObsticle21` follows the spelling used in the
provided dataset archive.
