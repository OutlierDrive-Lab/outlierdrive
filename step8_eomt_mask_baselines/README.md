# Step 8 - EoMT Mask-Based Anomaly Baselines.

This folder has the implementation for Step 8, which is used to evaluate EoMT.

Anomaly validation datasets used by the Step 7 ERFNet baselines.

The pipeline is set up using scripts that run in the terminal.

It has the ability to load EoMT models. sliding-window inference, mask-based and
pixel-based anomaly scores, metric calculation, CSV output, and MSP temperature scaling.

## Goal

This is the final part of the mask-based anomaly segmentation section of the project,
which we're now going to complete in Step 8.

The evaluation pipeline:

- it operates the Equation of Motion Transform on the validation datasets for anomaly
  segmentation
- computes MSP, MaxLogit, Max Entropy, and RbA-style scores
- reports AuPRC and FPR95 for each dataset and method
- evaluates three EoMT checkpoints
- supports Cityscapes mIoU calculation from exported prediction masks
- evaluates MSP with multiple temperatures from one model forward pass

The three required checkpoints are:

- COCO-trained EoMT
- Cityscapes-trained EoMT
- EoMT fine-tuned on Cityscapes in Step 5

## Current Status

The repository currently contains:

- `run_eomt_anomaly.py`: main anomaly evaluation runner
- `run_all_eomt_anomaly.sh`: wrapper for all checkpoints and datasets
- `run_temperature_sweep.sh`: wrapper for MSP temperature scaling
- `compute_cityscapes_miou.py`: mIoU calculation from saved Cityscapes masks
- `eomt_anomaly_results.csv`: 60 baseline rows
- `eomt_temperature_results.csv`: 60 temperature-scaling rows
- `eomt_anomaly_report_table.csv`: 12 report-ready baseline rows
- `eomt_all_results.csv`: combined baseline and temperature results
- `eomt_temperature_smoke_results.csv`: small preliminary validation output

The CSV files for anomalies and temperatures have the right number of unique rows as
expected.

Their fields for measuring intersection over union are currently empty because the exact
evaluation of semantic meaning.

The outcome needs to match the specific checkpoint that was used to evaluate any
anomalies.

Two checks remain important before final reporting:

- confirm the mIoU value for each exact checkpoint
- Compare the RbA-style formula with the official RbA implementation

## Repository Context

The implementation uses the following project components:

- `eval/evalAnomaly.py`: reference for dataset iteration, ground-truth handling, binary anomaly
  labels, AuPRC, and FPR95
- `eval/README.md`: original anomaly dataset documentation
- `eomt/README.md`: EoMT installation and checkpoint instructions
- `step4_eomt_eval/eomt_eval_iou.py`: Cityscapes evaluation on 19 classes
- `step4_eomt_eval/eomt_eval_overlap_iou.py`: COCO-to-Cityscapes overlap evaluation
- `step4_eomt_eval/test_eval_pipeline/predictions_eomt_city.ipynb`: reference Cityscapes EoMT inference
- `step4_eomt_eval/test_eval_pipeline/predictions_eomt_coco.ipynb`: reference COCO EoMT inference
- `eomt/training/lightning_module.py`: sliding-window and semantic-logit helpers

The Step 8 runner is a standalone component within this folder, and it doesn't make any
changes to other parts of the system.

The script `eval/evalAnomaly.py` was utilized, and for comparison purposes, the Step 7 runner
served as a reference point for behavioral analysis. not as a runtime dependency.

The main EoMT helpers used by the runner are:

```text
window_imgs_semantic
revert_window_logits_semantic
to_per_pixel_logits_semantic
```

## Required Local Inputs

Datasets and model checkpoints are intentionally not committed to Git.

A local The environment must provide:

- Python with the EoMT dependencies
- the anomaly validation datasets
- the EoMT checkpoint files
- the YAML configuration matching each checkpoint

The runner accepts these locations through command-line arguments.

Without the Even without local datasets and checkpoints, the scripts can still be
checked for syntax errors.

Full inference cannot be reproduced.

## Environment Setup

Install the EoMT requirements using Conda or a virtual environment.

### Conda

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python -m pip install -r eomt/requirements.txt
```

### Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r eomt/requirements.txt
```

Verify the main dependencies and available devices:

```bash
python - <<'PY'
import torch
import torchvision
import lightning
from PIL import Image
import sklearn

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print(
    "mps available:",
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
)
PY
```

The runner supports:

```text
auto
cpu
cuda
cuda:0
mps
```

For a thorough assessment, CUDA is the preferred choice.

However, if you're looking for a quicker option, CPU and MPS can be used for shorter
evaluations.

Running validation can be a slow process, especially when doing a full End-of-Month
(EoMT) evaluation on a CPU.

## Recommended Execution Strategy

1. Install the EoMT dependencies. Let's start by testing just one checkpoint and one
   dataset, and then we can try it out with one or two images to see how it works.
3. Confirm that the image and mask paths are resolved correctly.
4. Confirm that every anomaly method produces a non-constant score map.
5. Run all methods for one complete dataset. To get the best results, run the full
   checkpoint and dataset matrix on a computer with a GPU. This will help you process
   the information faster and more efficiently.
7. Validate the CSV row counts before using the values in the report.

To do a quick test, use `--max-images 1` or `--max-images 2`.

This will stop the process after looking at just a few images. argument for full
evaluation.

## Anomaly Validation Datasets

Download and extract `Anomaly_Validation_Datasets.zip` from the project

Don't add the archive or extracted datasets to Git, just the materials.

Expected dataset folders:

```text
RoadAnomaly21
RoadObsticle21
RoadAnomaly
fs_static
FS_LostFound_full
```

The name `RoadObstacle21` is spelled exactly as it appears in the given dataset
directory.

report-ready CSV uses the corrected display name `RoadObstacle21`.

Expected structure:

```text
Anomaly_Validation_Datasets/
├── RoadAnomaly21/
│   ├── images/
│   └── labels_masks/
├── RoadObsticle21/
│   ├── images/
│   └── labels_masks/
├── RoadAnomaly/
│   ├── images/
│   └── labels_masks/
├── fs_static/
│   ├── images/
│   └── labels_masks/
└── FS_LostFound_full/
    ├── images/
    └── labels_masks/
```

The supported input image extensions are:

```text
.png
.jpg
.jpeg
.webp
```

The program also takes an option called `--input-glob` which can be used to manually
specify the dataset instead of relying on the automatic one. discovery.

## Ground-Truth Handling

Step 8 uses the same binary convention as the ERFNet baseline:

```text
1   = anomaly or OOD pixel
0   = in-distribution pixel
255 = ignored pixel
```

To figure out the actual paths, we replace 'images' with 'labels_masks'.

Extension conversion is handled as follows:

- `RoadObsticle21`: `.webp` to `.png`
- `fs_static`: `.jpg` or `.jpeg` to `.png. `
- `RoadAnomaly`: `.jpg` or `.jpeg` to `.png. `

Dataset masks are normalized as follows:

The label for RoadAnomaly is changed from 2 to 1, indicating it's now considered an
anomaly.

- Lost and Found style masks use a specific labeling system, where label `0`
  is treated as an ignore label and label `1` is used for other purposes, and
  then there's also label `255`. it turns into a label that's already part of
  the group, which is `0`, and the other labels, which are from `2`
  to `200`, also become part of this group. anomaly label `1. `
- We're keeping the way we handle street hazards the same as it was before, so
  everything stays compatible with the original system. evaluator

The `FS_LostFound_full` masks are utilized in their given binary format.

Images without anomaly pixels are skipped.

During metric calculation, pixels with label `255` are excluded.

## EoMT Checkpoints

The complete evaluation requires three checkpoints:

```text
eomt_checkpoints/eomt_coco.bin
eomt_checkpoints/eomt_cityscapes.bin
eomt_checkpoints/eomt_finetuned.bin
```

The actual names of the files might be different, but the main program can handle
specific checkpoints. paths through `--checkpoint`.

The current shell wrappers use these example paths:

```text
eomt_checkpoints/eomt_coco.bin
eomt_checkpoints/eomt_cityscapes.bin
eomt_checkpoints/epoch=9-step=1850.ckpt
```

Update the wrapper paths before running them on another machine.

Checkpoint Files must remain outside Git.

## EoMT Configurations

Use the configuration to match each checkpoint.

COCO-trained checkpoint:

```text
eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml
```

Cityscapes-trained and fine-tuned checkpoints:

```text
eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml
```

The runner reads model classes and initialization arguments from YAML.

It then It tries to figure out certain values from special files called checkpoint
tensors when it can.

- image size
- number of classes
- number of object queries

Explicit fallback values can be supplied with:

```text
--img-size
--num-classes
--stuff-classes
```

Masked attention is disabled during inference.

The classification head can be included or excluded with:

```text
--load-ckpt-class-head
--no-load-ckpt-class-head
```

Checkpoint loading uses `strict=False` and prints the number of missing or Unexpected
keys when the checkpoint and configured model do not match exactly.

## Model Loading

The runner:

1. loads the YAML configuration
2. selects the requested device
3. reads the checkpoint state dictionary
4. resolves image size, class count, and query count
5. constructs the configured encoder It builds the EoMT network without using masked
   attention.
7. constructs the configured Lightning module
8. loads checkpoint weights
9. moves the model to the selected device
10. switches the model to evaluation mode

The checkpoint loader supports raw state dictionaries and dictionaries containing one of
these keys:

```text
state_dict
model
model_state_dict
```

## EoMT Forward Pass

ERFNet produces per-pixel class outputs directly.

EoMT produces query-level mask and class predictions:

```text
mask_logits:  [B, Q, H, W]
class_logits: [B, Q, C + 1]
```

The final class is the no-object class.

For full-size anomaly images, the runner uses EoMT sliding-window inference:

```Python
imgs = [image.to(device)]
img_sizes = [image.shape[-2:] for image in imgs]
crops, origins = model.window_imgs_semantic(imgs)
mask_logits_per_layer, class_logits_per_layer = model(crops)
mask_logits = F.interpolate(
    mask_logits_per_layer[-1],
    model.img_size,
    mode="bilinear",
)
class_logits = class_logits_per_layer[-1]
```

For MSP, MaxLogit, and Max Entropy, query predictions are converted to semantic. pixel
logits:

```Python
crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
logits = model.revert_window_logits_semantic(
    crop_logits,
    origins,
    img_sizes,
)[0]
```

The semantic conversion is equivalent to:

```python
pixel_logits = torch.einsum(
    "bqhw,bqc->bchw",
    mask_logits.sigmoid(),
    class_logits.softmax(dim=-1)[..., :-1],
)
```

Inference runs under `torch.no_grad()`.

CUDA execution uses autocast.

## Anomaly Scoring Methods

Every method produces a two-dimensional score map where a larger value means it's more
likely that a pixel will be unusual.

### MSP

Maximum Softmax Probability uses:

```Python
probs = torch.softmax(pixel_logits / temperature, dim=0)
score = 1.0 - probs.max(dim=0).values
```

Low maximum class confidence gives a high anomaly score.

### MaxLogit

MaxLogit uses:

```python
score = -pixel_logits.max(dim=0).values
```

MaxLogit uses raw logits and is not changed by temperature scaling.

### Max Entropy

Predictive entropy uses:

```Python
probs = torch.softmax(pixel_logits / temperature, dim=0)
score = -(probs * torch.log(probs + 1e-12)).sum(dim=0)
```

High uncertainty gives a high anomaly score.

### RbA-Style Rejection

RbA uses query-level mask and class predictions before semantic aggregation:

```python
mask_probs = mask_logits.sigmoid()
class_probs = class_logits.softmax(dim=-1)[..., :-1]
known_scores = torch.einsum(
    "bqhw,bqc->bchw",
    mask_probs,
    class_probs,
).clamp(0.0, 1.0)

rba_score = torch.exp(
    torch.log1p(
        -known_scores.clamp(max=1.0 - 1e-6)
    ).sum(dim=1, keepdim=True)
)
```

This score is really high when a pixel doesn't fit into any of the classes we already
know about.

It's all about the log-space.

Calculation is used for numerical stability.

The current implementation follows the RbA mask-query idea, but the exact The formula
needs to be checked against the official RbA implementation to make sure it's correct.

This comparison is necessary to ensure everything is working as it should.

Final Report:

```text
https://github.com/NazirNayal8/RbA
```

If something is different from the standard way of doing things, it should be written
down so everyone knows about it. report.

## Metrics

Anomaly segmentation is treated as binary pixel classification:

```text
positive class = anomaly or OOD pixel
negative class = in-distribution pixel
```

For each checkpoint, dataset, and method, valid pixel scores are concatenated and
evaluated with:

- AuPRC
- FPR95

The calculation follows:

```Python
ood_mask = ground_truth == 1
ind_mask = ground_truth == 0

scores = np.concatenate([
    anomaly_scores[ind_mask],
    anomaly_scores[ood_mask],
])
labels = np.concatenate([
    np.zeros(num_ind_pixels),
    np.ones(num_ood_pixels),
])

auprc = average_precision_score(labels, scores) * 100.0
fpr95 = fpr_at_95_tpr(scores, labels) * 100.0
```

FPR95 is the false-positive rate at the first threshold where true-positive The rate
reaches at least 95 percent.

## Run One Evaluation

Run one method on one dataset:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint /path/to/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method maxlogit \
  --max-images 2 \
  --device cpu \
  --output-csv step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

Run all four methods with one forward pass:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint /path/to/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method all \
  --device cuda \
  --output-csv step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

Remove `--max-images` for a full dataset run.

## Run All Baselines

Edit the checkpoint paths in:

```text
step8_eomt_mask_baselines/run_all_eomt_anomaly.sh
```

The wrapper can be configured with environment variables:

```text
DATASET_ROOT
OUTPUT_CSV
DEVICE
PYTHON
MAX_IMAGES
```

Example:

```bash
DATASET_ROOT=/path/to/Anomaly_Validation_Datasets \
DEVICE=cuda \
PYTHON=.venv/bin/python \
bash step8_eomt_mask_baselines/run_all_eomt_anomaly.sh
```

The complete matrix is:

```text
3 checkpoints x 5 datasets x 4 methods = 60 rows
```

Methods:

```text
msp
maxlogit
entropy
rba
```

Results are written to:

```text
step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

## Temperature Scaling

Temperature scaling divides semantic pixel logits by a positive temperature before
softmax.

The project sweep uses:

```text
0.5, 0.75, 1.0, 1.1
```

Run several MSP temperatures from one model forward pass:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint /path/to/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method msp \
  --temperatures 0.5,0.75,1.0,1.1 \
  --device cuda \
  --output-csv step8_eomt_mask_baselines/eomt_temperature_results.csv
```

The `--temperatures` option works with MSP and entropy, but that's about it. combined with
`--method all`.

For a complete review, you need to modify the paths where the checkpoints are stored.

`run_temperature_sweep.sh`, then run:

```bash
DATASET_ROOT=/path/to/Anomaly_Validation_Datasets \
DEVICE=cuda \
PYTHON=.venv/bin/python \
bash step8_eomt_mask_baselines/run_temperature_sweep.sh
```

Optional environment variables are:

```text
DATASET_ROOT
OUTPUT_CSV
DEVICE
PYTHON
TEMPERATURES
MAX_IMAGES
```

The complete temperature matrix is:

```text
3 checkpoints x 5 datasets x 4 temperatures = 60 rows
```

Results are written to:

```text
step8_eomt_mask_baselines/eomt_temperature_results.csv
```

## Saved Logits

Use `--save-logits` to save semantic pixel logits as compressed `.npz` files:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config /path/to/config.yaml \
  --checkpoint /path/to/checkpoint \
  --checkpoint-name checkpoint_name \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method msp \
  --save-logits \
  --logits-dir step8_eomt_mask_baselines/saved_logits
```

The current path for listing temperatures already covers all the temperatures we need to
look at. from one in-memory forward pass.

Saved `.npz` files are useful for inspection and for use in future analysis when
not connected to the internet, however the current system in place does not update them
as needed. input.

Don't save big log files and then add them to Git.

## Cityscapes mIoU

The project report requires mIoU for each EoMT checkpoint.

This value does not change when the post-hoc anomaly method changes. For

Cityscapes-trained and fine-tuned checkpoints, mIoU should be computed on the full
19-class Cityscapes validation set.

For the COCO checkpoint, use the We use the same class mapping from COCO to Cityscapes
and the evaluation strategy that was used in the previous step.

Do not compare values calculated on different class spaces as if they were the same same
metric.

If checkpoint predictions are already exported as Cityscapes `labelTrainIds. ` PNG masks,
run:

```bash
python step8_eomt_mask_baselines/compute_cityscapes_miou.py \
  --checkpoint-name eomt_cityscapes \
  --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --gt-glob "/path/to/gtFine/val/*/*_gtFine_labelTrainIds.png" \
  --pred-dir /path/to/eomt_cityscapes_predictions \
  --output-csv step8_eomt_mask_baselines/eomt_miou_results.csv \
  --per-class-csv step8_eomt_mask_baselines/eomt_cityscapes_class_iou.csv
```

The helper reports:

- mIoU
- pixel accuracy
- number of missing predictions
- optional per-class IoU

The output schema is:

```csv
checkpoint,config,miou,pixel_accuracy,missing_predictions
```

After verifying the exact checkpoint mapping, pass its mIoU into the anomaly evaluation
with:

```text
--miou <verified-value>
```

The current anomaly result files have a blank mIoU.

Just leave this field empty, don't fill it in. guessing or by mixing full-class and
overlap-class evaluations.

## Main Runner Options

Required arguments:

```text
--config
--checkpoint
--checkpoint-name
--dataset-root
--dataset
--method
```

Optional arguments:

```text
--repo-root
--input-glob
--temperature
--temperatures
--miou
--output-csv
--img-size
--num-classes
--stuff-classes
--device
--max-images
--save-logits
--logits-dir
--load-ckpt-class-head
--no-load-ckpt-class-head
```

Supported methods:

```text
msp
maxlogit
entropy
rba
all
```

## Result CSV Schema

The anomaly and temperature result files are used:

```csv
model,checkpoint,dataset,method,temperature,miou,auprc,fpr95,num_images,num_ood_pixels,num_ind_pixels
```

The fields record:

- model family
- checkpoint identifier
- dataset
- anomaly method
- temperature
- semantic mIoU when verified
- AuPRC
- FPR95
- number of evaluated images
- number of anomaly pixels
- number of in-distribution pixels

## Result Files

### Baseline Results

```text
step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

This file contains:

```text
3 checkpoints x 5 datasets x 4 methods = 60 rows
```

The baseline file uses temperature `1.0`.

### Temperature Results

```text
step8_eomt_mask_baselines/eomt_temperature_results.csv
```

This file contains:

```text
3 checkpoints x 5 datasets x 4 temperatures = 60 rows
```

Method labels include:

```text
msp
msp_t0.5
msp_t0.75
msp_t1.1
```

### Combined Results

```text
step8_eomt_mask_baselines/eomt_all_results.csv
```

This file combines the 60 baseline rows and 60 temperature rows.

Its The `result_group` column distinguishes the two groups.

### Report-Ready Table

```text
step8_eomt_mask_baselines/eomt_anomaly_report_table.csv
```

This document has a single line for each checkpoint and technique.

```text
3 checkpoints x 4 methods = 12 rows
```

It expands the five dataset metrics into columns for direct use in the report.

### Preliminary Temperature Output

```text
step8_eomt_mask_baselines/eomt_temperature_smoke_results.csv
```

This small three-row file is just a starting point, we need to use the complete data to
get a better understanding of what's going on.

`eomt_temperature_results.csv` for the final report.

## Final Report Table

The final anomaly table should follow this structure:

```text
Model/checkpoint
mIoU
Method
RA-21 AuPRC
RA-21 FPR95
RO-21 AuPRC
RO-21 FPR95
FS Lost & Found AuPRC
FS Lost & Found FPR95
FS Static AuPRC
FS Static FPR95
Road Anomaly AuPRC
Road Anomaly FPR95
```

Required EoMT rows:

```text
EoMT COCO - MSP
EoMT COCO - MaxLogit
EoMT COCO - Max Entropy
EoMT COCO - RbA
EoMT Cityscapes - MSP
EoMT Cityscapes - MaxLogit
EoMT Cityscapes - Max Entropy
EoMT Cityscapes - RbA
EoMT Fine-tuned - MSP
EoMT Fine-tuned - MaxLogit
EoMT Fine-tuned - Max Entropy
EoMT Fine-tuned - RbA
```

If the report has one big table, you should add the ERFNet Step 7 rows separately.

This will help keep things organized and make it easier to understand the information.

## Validation Checklist

Repository-level checks:

- [x] the main runner is present
- [x] all five required datasets are supported
- [x] MSP is implemented
- [x] MaxLogit is implemented
- [x] Max Entropy is implemented
- [x] RbA-style query rejection is implemented
- [x] ignored pixels are excluded from metrics

The baseline data set has 60 distinct entries.

The temperature file has 60 different rows of data.

- [x] temperature lists reuse one forward pass
- [x] checkpoint and dataset paths are configurable Python and shell scripts have been
  checked for syntax errors and they all pass, so we're good to go.

Checks requiring the local datasets or exact report checkpoint mapping:

- [ ] rerun one image for every checkpoint configuration
- [ ] verify every dataset resolves all expected masks
- [ ] confirm all anomaly maps are non-constant
- [ ] compare the RbA-style score with the official RbA implementation

Can you please tell me the specific fine-tuned checkpoint that was used in Step 8?

I need to know this to move forward.

- [ ] compute and verify mIoU for all three checkpoints

Fill in the columns for mean Intersection over Union (mIoU) with the values that match
each checkpoint.

- [ ] confirm the report discussion explains unusually strong or weak results

## Known Result Checks

Before finalizing the report, inspect these results carefully:

- `eomt_cityscapes` on RoadObstacle21
- `eomt_finetuned` on `fs_static`
- `eomt_finetuned` FPR95 on Road Anomaly

Unexpectedly strong or weak values are not automatically incorrect, but they should be
rerun or explained before final submission.

## Main Risks and Limitations

- COCO and Cityscapes use different class spaces.
  Their mIoU values require a consistent mapping and must be labeled clearly.
- EoMT preprocessing must match the checkpoint configuration.
  Incorrect image size or normalization makes the evaluation unreliable.

- The current RbA score is a query-level rejection implementation that still requires
  comparison with the official method.
- Full-size anomaly images can require significant memory.
  Use batch size one, sliding-window inference, and CUDA autocast.

- CPU evaluation can be very slow.
- Checkpoint loading with `strict=False` can hide architectural mismatches if warnings
  are ignored.
- It's worth noting that saved logits can be quite big, so they need to be kept separate
  from Git.

The current fields for mean intersection over union are blank and should not be
populated with data that doesn't match. semantic evaluation values.

## Reproducibility Notes

Don't store things like datasets, checkpoints, and saved logits in your Git repository.

- Record the exact checkpoint filename used for each CSV.
- Record the configuration file used with each checkpoint.
- Maintain the temperature at `1.0` for the standard baseline table.
- Use the same metric implementation for all checkpoints and methods.
- Exclude ignored pixels consistently.
- Avoid appending duplicate rows when rerunning experiments.
- Validate row counts and unique checkpoint-dataset-method combinations after every full
  run.

## Completion Order

The implementation was developed in this order:

1. Let's use the Step 7 dataset and follow the same metric conventions as before.
2. Implement EoMT checkpoint and configuration loading
3. Implement one-image sliding-window inference
4. convert mask-query outputs to semantic pixel logits
5. add MaxLogit, MSP, and Max Entropy
6. Add ground-truth mask normalization and metrics
7. run one complete dataset
8. Add the RbA-style query score
9. add all-checkpoint and all-dataset wrappers
10. Add mIoU calculation from saved Cityscapes masks
11. Add multi-temperature evaluation from one forward pass
12. generate baseline, temperature, combined, and report-ready CSVs

The remaining work to be done is to finish the checkpoint-matched mIoU insertion and
then complete the final tasks.

RbA verification.
