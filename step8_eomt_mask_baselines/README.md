# Step 8 - EoMT Mask-Based Anomaly Baselines

Implementation of  Step 8 : evaluate EoMT on the same
anomaly validation datasets used for the ERFNet pixel-based baselines.

 Pixel level anomaly segmentation metrics for:

- MSP
- MaxLogit
- Max Entropy
- RbA-style mask-query rejection

It is designed to evaluate all required EoMT checkpoints:

- COCO-trained EoMT
- Cityscapes-trained EoMT
- Fine-tuned EoMT from Step 5

## Required Local Inputs


- `Anomaly_Validation_Datasets/`
- EoMT checkpoint `.bin` files
- The EoMT config matching each checkpoint


The model construction and sliding-window inference path follows the Step 4
notebooks on this branch, especially:

```text
step4_eomt_eval/test_eval_pipeline/predictions_eomt_city.ipynb
step4_eomt_eval/test_eval_pipeline/predictions_eomt_coco.ipynb
```

The runner does not instantiate EoMT dataset modules for anomaly evaluation.
This is intentional because Step 8 reads images and anomaly masks directly from
`Validation_Dataset/`. Pass `--img-size` and `--num-classes` explicitly only if
checkpoint-based inference is not possible.
By default the runner infers image size, number of classes, and query count from
the checkpoint tensors.


## Reporting Notes

The final anomaly benchmark is:

This CSV contains 60 anomaly-evaluation rows:



The `miou` column is intentionally empty in the anomaly CSV. If the report table
requires mIoU, fill it from the Step 4/5 semantic segmentation evaluation on
Cityscapes, not from the anomaly segmentation run.

The final anomaly CSV only uses `temperature=1.0`. Temperature scaling is
implemented and smoke-tested separately, but it is not part of
`eomt_anomaly_results.csv` unless the temperature sweep is run and reported as an
extra baseline.

Additional result files:

`eomt_temperature_results.csv` contains the 60-row MSP temperature-scaling sweep:
3 checkpoints x 5 datasets x 4 temperatures. `eomt_all_results.csv` keeps both
the 60-row anomaly baseline table and the 60-row temperature-scaling table in one
file, with a `result_group` column to distinguish the original anomaly baselines
from the temperature-scaling rows.

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
By default it evaluates MSP temperature
scaling for all three checkpoints and all five anomaly datasets at
`0.5, 0.75, 1.0, 1.1`, producing 60 rows:


The final temperature-scaling rows should be reported from
`eomt_temperature_results.csv` or another full sweep output. The small
`eomt_temperature_smoke_results.csv` file is only a sanity check.

## Method Notes

MSP, MaxLogit, and Max Entropy use semantic per-pixel logits produced by EoMT's
existing `to_per_pixel_logits_semantic` helper.

RbA uses query-level mask and class predictions before the final semantic
aggregation. The implementation follows the mask-architecture idea of rejecting
pixels that are not confidently accepted by any known query/class. Before final
reporting, compare this formula against the official RbA implementation and
document any difference.

