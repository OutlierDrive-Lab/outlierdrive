# Step 8 - EoMT Mask-Based Anomaly Baselines

This folder implements Step 8 from the project PDF: evaluate EoMT on the same
anomaly validation datasets used for the ERFNet pixel-based baselines.

The runner reports pixel-level anomaly segmentation metrics for:

- MSP
- MaxLogit
- Max Entropy
- RbA-style mask-query rejection

It is designed to evaluate all required EoMT checkpoints:

- COCO-trained EoMT
- Cityscapes-trained EoMT
- Fine-tuned EoMT from Step 5

## Required Local Inputs

Do not commit datasets or checkpoint files.

You need local paths for:

- `Anomaly_Validation_Datasets/`
- EoMT checkpoint `.bin` files
- The EoMT config matching each checkpoint

The current branch contains the COCO panoptic config:

```text
eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml
```

This branch currently has the COCO config, Cityscapes config, EoMT checkpoints,
and extracted anomaly validation datasets needed for smoke testing.

The model construction and sliding-window inference path follows the Step 4
notebooks on this branch, especially:

```text
step4_eomt_eval/predictions_eomt_city.ipynb
step4_eomt_eval/predictions_eomt_coco.ipynb
```

The runner does not instantiate EoMT dataset modules for anomaly evaluation.
This is intentional because Step 8 reads images and anomaly masks directly from
`Validation_Dataset/`. Pass `--img-size` and `--num-classes` explicitly only if
checkpoint-based inference is not possible.
By default the runner infers image size, number of classes, and query count from
the checkpoint tensors.

## Environment

Install EoMT dependencies first:

```bash
cd eomt
python -m pip install -r requirements.txt
cd ..
```

For local smoke tests on macOS, use `--device cpu` or `--device mps` if PyTorch
MPS is available. Full evaluation is much faster on CUDA.

## One Smoke Test

Run one method on two images:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint eomt_checkpoints/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root Validation_Dataset \
  --dataset RoadAnomaly21 \
  --method maxlogit \
  --max-images 2 \
  --device cpu
```

The output CSV defaults to:

```text
step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

CSV columns:

```csv
model,checkpoint,dataset,method,temperature,miou,auprc,fpr95,num_images,num_ood_pixels,num_ind_pixels
```

## Reporting Notes

The final anomaly benchmark is:

```text
step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

This CSV contains 60 anomaly-evaluation rows:

```text
3 checkpoints x 5 datasets x 4 methods
```

The `miou` column is intentionally empty in the anomaly CSV. If the report table
requires mIoU, fill it from the Step 4/5 semantic segmentation evaluation on
Cityscapes, not from the anomaly segmentation run.

The final anomaly CSV only uses `temperature=1.0`. Temperature scaling is
implemented and smoke-tested separately, but it is not part of
`eomt_anomaly_results.csv` unless the temperature sweep is run and reported as an
extra baseline.

The dataset folder is named `RoadObsticle21` in the provided validation archive.
The raw CSV keeps that exact folder name, but the report-ready table uses the
clean display name `RoadObstacle21`.

Some results can look surprisingly strong or weak across datasets. Treat these
as values to double-check in the report discussion, especially:

- `eomt_cityscapes` on `RoadObstacle21` (`RoadObsticle21` folder)
- `eomt_finetuned` on `fs_static`
- `eomt_finetuned` FPR95 on `RoadAnomaly`

## mIoU CSV

If checkpoint predictions are already exported as Cityscapes `labelTrainIds`
PNG masks, compute mIoU with:

```bash
python step8_eomt_mask_baselines/compute_cityscapes_miou.py \
  --checkpoint-name eomt_cityscapes \
  --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --gt-glob "/local/path/gtFine_trainIds/val/*/*_gtFine_labelTrainIds.png" \
  --pred-dir /local/path/eomt_cityscapes_predictions/png \
  --output-csv step8_eomt_mask_baselines/eomt_miou_results.csv
```

Pass the resulting mIoU value back to anomaly runs with `--miou`.

## All Runs

Copy the example wrapper and fill in the local checkpoint/config paths:

```bash
bash step8_eomt_mask_baselines/run_all_eomt_anomaly.sh
```

The full Step 8 anomaly table has:

```text
3 checkpoints x 5 datasets x 4 methods = 60 rows
```

Datasets:

- `RoadAnomaly21`
- `RoadObsticle21`
- `RoadAnomaly`
- `fs_static`
- `FS_LostFound_full`

Methods:

- `msp`
- `maxlogit`
- `entropy`
- `rba`

## Temperature Scaling

Temperature scaling is supported through `--temperature`. For the PDF table, use
MSP with values such as:

```text
0.5, 0.75, 1.0, 1.1
```

Example:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint eomt_checkpoints/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root Validation_Dataset \
  --dataset RoadAnomaly21 \
  --method msp \
  --temperature 0.75 \
  --output-csv step8_eomt_mask_baselines/eomt_temperature_results.csv
```

There is also an editable wrapper:

```bash
bash step8_eomt_mask_baselines/run_temperature_sweep.sh
```

Only claim temperature-scaling results if you run the sweep and use
`eomt_temperature_results.csv` or another final temperature-scaling CSV. The
small `eomt_temperature_smoke_results.csv` file is only a sanity check.

## Method Notes

MSP, MaxLogit, and Max Entropy use semantic per-pixel logits produced by EoMT's
existing `to_per_pixel_logits_semantic` helper.

RbA uses query-level mask and class predictions before the final semantic
aggregation. The implementation follows the mask-architecture idea of rejecting
pixels that are not confidently accepted by any known query/class. Before final
reporting, compare this formula against the official RbA implementation and
document any difference.

## Validation Checklist

- Run `--max-images 1` or `--max-images 2` before full evaluation.
- Verify every dataset resolves its `labels_masks/` files correctly.
- Confirm anomaly maps are not constant for MSP, MaxLogit, Entropy, and RbA.
- Exclude ignored pixels with label `255` from metrics.
- Fill the `--miou` value for each checkpoint once it is known.
- Do not commit checkpoints, datasets, or large saved logits.
