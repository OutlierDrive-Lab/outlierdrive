# Step 8 EoMT Mask-Based Anomaly Baselines Plan

This document describes the Step 8 evaluation plan for EoMT mask-based anomaly segmentation baselines on the same anomaly validation datasets used in Step 7.

## Goal

Complete the "Mask-based baselines" section from the project PDF.

The final result should be a reproducible evaluation pipeline that:

- Runs EoMT on the anomaly segmentation validation datasets.
- Computes anomaly scores using MSP, MaxLogit, Max Entropy, and RbA.
- Evaluates AuPRC and FPR@95 for each dataset and method.
- Repeats the evaluation for three EoMT checkpoints:
  - COCO-trained EoMT.
  - Cityscapes-trained EoMT.
  - Fine-tuned EoMT from Step 5.
- Reports mIoU for each checkpoint.
- Adds temperature scaling experiments for MSP if time allows.

## Existing Repo Context

Important files already in this repository:

- `eval/evalAnomaly.py`: Original ERFNet anomaly evaluation script. Use its dataset loop, ground-truth path handling, mask normalization, and metric logic as a reference.
- `eval/README.md`: Explains the anomaly datasets and the original eval command.
- `eomt/README.md`: Explains how to install EoMT requirements, load checkpoints, and run validation.
- `step4_eomt_eval/test_eval_pipeline/predictions_eomt_city.ipynb`: Shows Cityscapes EoMT inference and prediction export.
- `step4_eomt_eval/test_eval_pipeline/predictions_eomt_coco.ipynb`: Shows COCO EoMT inference and prediction export.
- `step4_eomt_eval/eomt_eval_iou.py`: Scripted Cityscapes mIoU evaluation on all 19 trainId classes.
- `step4_eomt_eval/eomt_eval_overlap_iou.py`: Scripted COCO-to-Cityscapes overlap evaluation with class mapping.
- `eomt/training/lightning_module.py`: Contains key EoMT inference helpers:
  - `window_imgs_semantic`
  - `revert_window_logits_semantic`
  - `to_per_pixel_logits_semantic`
- `eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml`: COCO panoptic EoMT config currently present on this branch.

Important branch policy:

- Do not merge, rebase, or cherry-pick colleague branches just to complete Step 8.
- Step 8 should be self-contained on this branch.
- Other branches may be inspected as read-only references only.
- If a useful implementation exists on another branch, reimplement the small needed utility in the Step 8 folder or copy only a tiny, reviewed, self-contained snippet with attribution in comments if appropriate.
- The old Step 7 implementation exists on `origin/feature/erfnet-baselines` under `step7_erfnet_pixel_baselines/`; treat it as a reference template only, not a dependency.
- That Step 7 runner is useful because it shows MSP, MaxLogit, Max Entropy, CSV output, binary anomaly masks, AuPRC, and FPR@95.
- The current branch may not contain the Cityscapes EoMT config. If missing, add the needed config file directly to this branch from official/project materials or reconstruct it from the known EoMT config pattern. Do not merge another branch just for the config:
  - `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml`
  - It was present on the data-evaluation work.

## Required Inputs

These inputs are required locally to run Step 8, but they should not be committed to git.

The repository currently does not include the anomaly dataset archive or EoMT checkpoint files. This is expected: `.gitignore` excludes dataset folders, zip files, model weights, checkpoints, `.bin` files, and cached logits. The implementation should accept paths to these local files through CLI arguments.

## Local Execution Feasibility

Step 8 should be implemented as normal Python scripts that can run from the terminal. JupyterLab is optional for visualization/debugging, but the final pipeline should not require notebooks.

The repository does not track datasets or checkpoint files. A local environment must provide:

- Python with the EoMT dependencies installed.
- The anomaly validation datasets.
- The EoMT checkpoint files.
- The matching EoMT configuration files.

Without those local inputs, the scripts can still be syntax-checked and reviewed, but full EoMT inference cannot be reproduced.

To run Step 8 locally, create an environment first. Since `conda` is not available right now, either install Miniconda as described in `eomt/README.md`, or create another stable Python environment that can install PyTorch and EoMT dependencies.

Recommended local setup:

```bash
# Option A: Conda/Miniconda, matching eomt/README.md
conda create -n eomt python==3.13.2
conda activate eomt
python -m pip install -r eomt/requirements.txt

# Option B: stable venv if a suitable Python is installed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r eomt/requirements.txt
```

After setup, verify:

```bash
python - <<'PY'
import torch
import torchvision
import lightning
from PIL import Image
import sklearn
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

On this local macOS machine, CUDA should not be assumed. If PyTorch MPS is available, the runner can optionally support `--device mps`; otherwise use CPU for smoke tests. Full EoMT anomaly evaluation on CPU may be very slow, especially for all datasets and all checkpoints.

Recommended execution strategy:

1. Develop the pipeline as terminal scripts.
2. Run small CPU/MPS smoke tests locally:
   - one checkpoint,
   - one dataset,
   - one or two images,
   - one method such as `maxlogit`.
3. If local runtime is too slow, run the same script on JupyterLab, Colab, or another GPU machine. Do not rewrite the pipeline as notebook-only code.
4. Keep notebooks only for visualization or exploratory debugging.

The README should show that Step 8 can be launched from the terminal, for example:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint /local/path/eomt_coco.bin \
  --dataset-root /local/path/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method maxlogit \
  --max-images 2 \
  --device cpu
```

Use `--max-images` for local smoke tests, then remove it for full evaluation.

### 1. Anomaly Validation Datasets

Download and prepare `Anomaly_Validation_Datasets.zip` from the project Drive link referenced in `eval/README.md`.

Do not add the zip file or extracted dataset to git. Keep it somewhere local, for example:

```text
/Users/sadaf/outlierdrive/Anomaly_Validation_Datasets/
```

or:

```text
/Users/sadaf/datasets/Anomaly_Validation_Datasets/
```

The runner should take this location as `--dataset-root`.

Expected dataset folders:

- `RoadAnomaly21`
- `RoadObsticle21`
- `RoadAnomaly`
- `fs_static`
- `FS_LostFound_full`

Expected structure per dataset:

```text
<dataset_root>/<dataset_name>/images/*
<dataset_root>/<dataset_name>/labels_masks/*
```

The exact image extensions differ by dataset:

- `RoadAnomaly21`: usually `.png`
- `RoadObsticle21`: usually `.webp`
- `RoadAnomaly`: usually `.jpg`
- `fs_static`: usually `.jpg`
- `FS_LostFound_full`: check actual folder contents

### 2. EoMT Checkpoints

The Step 8 table requires all three:

```text
eomt_checkpoints/eomt_coco.bin
eomt_checkpoints/eomt_cityscapes.bin
eomt_checkpoints/eomt_finetuned.bin
```

Names can differ, but the runner should accept explicit paths through CLI args.

Do not commit checkpoint files. They are large model artifacts and `.gitignore` excludes `*.bin`, `*.pth`, `*.pt`, and checkpoint directories.

### 3. EoMT Configs

Use the config that matches each checkpoint:

- COCO checkpoint:
  - `eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml`
- Cityscapes checkpoint:
  - `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml`
- Fine-tuned checkpoint:
  - Usually the same Cityscapes semantic config unless fine-tuning used a different config.

For inference, set:

```text
--model.network.masked_attn_enabled False
```

or instantiate the network with `masked_attn_enabled=False`, matching `eomt/README.md`.

## Deliverables

Create a dedicated Step 8 folder:

```text
step8_eomt_mask_baselines/
```

Recommended files:

```text
step8_eomt_mask_baselines/
  README.md
  run_eomt_anomaly.py
  run_all_eomt_anomaly.sh
  eomt_anomaly_results.csv
  eomt_temperature_results.csv
  saved_logits/                 # optional; keep out of git if large
```

Expected CSV format:

```csv
model,checkpoint,dataset,method,miou,auprc,fpr95
EoMT,eomt_coco,RoadAnomaly21,msp,,,
EoMT,eomt_coco,RoadAnomaly21,maxlogit,,,
EoMT,eomt_coco,RoadAnomaly21,entropy,,,
EoMT,eomt_coco,RoadAnomaly21,rba,,,
```

`miou` should be filled once per checkpoint and repeated across rows for convenience, or stored in a separate `eomt_miou_results.csv`.

## Implementation Plan

### Task 1: Recover the Step 7 Runner Pattern

Inspect the Step 7 runner from `origin/feature/erfnet-baselines` as a read-only template. Do not merge or cherry-pick it.

Reference command:

```bash
git show origin/feature/erfnet-baselines:step7_erfnet_pixel_baselines/run_erfnet_anomaly_cpu.py
```

Port these pieces into the Step 8 runner:

- `fpr_at_95_tpr`
- CSV appending
- anomaly image globbing
- ground-truth path inference
- ground-truth mask normalization
- `average_precision_score`
- excluding ignored pixels
- one row per dataset and method

Do not modify the original `eval/evalAnomaly.py`. Keep Step 8 code separate.

### Task 2: Implement Dataset Ground Truth Utilities

Reuse the same logic as Step 7.

The binary convention should be:

```text
1 = anomaly / OOD pixel
0 = in-distribution pixel
255 = ignore pixel
```

Ground-truth path inference:

- Replace `images` with `labels_masks`.
- For `RoadObsticle21`, replace `.webp` with `.png`.
- For `fs_static`, replace `.jpg` with `.png`.
- For `RoadAnomaly`, replace `.jpg` with `.png`.

Ground-truth normalization:

- `RoadAnomaly`: convert label `2` to anomaly `1`.
- `LostAndFound`: convert `0` to ignore `255`, `1` to inlier `0`, and labels between `2` and `200` to anomaly `1`.
- `Streethazard`: keep the existing handling from `eval/evalAnomaly.py` if this dataset appears.

Skip images that do not contain anomaly pixels, matching Step 7 behavior.

### Task 3: Implement EoMT Model Loading

Use the EoMT loading and sliding-window inference logic from the Step 4 notebooks and scripts rather than ERFNet.

The loader should:

- Read a YAML config.
- Instantiate the configured data/model classes as needed.
- Instantiate the encoder and EoMT network.
- Instantiate the Lightning module.
- Load checkpoint weights.
- Move model to `cuda` if available, else CPU.
- Set `model.eval()`.
- Disable masked attention during inference.

Keep the CLI flexible:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint eomt_checkpoints/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method maxlogit \
  --output-csv step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

The runner should also allow:

```text
--device cuda
--device cpu
--save-logits
--logits-dir step8_eomt_mask_baselines/saved_logits
--temperature 1.0
```

### Task 4: Implement EoMT Forward Pass

ERFNet returns per-pixel outputs directly. EoMT returns mask logits and class logits.

EoMT output shape:

```text
mask_logits:  [B, Q, H, W]
class_logits: [B, Q, C + 1]
```

The last class is the no-object class.

For semantic pixel logits, use EoMT's existing function:

```python
pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
```

This is equivalent to:

```python
pixel_logits = torch.einsum(
    "bqhw,bqc->bchw",
    mask_logits.sigmoid(),
    class_logits.softmax(dim=-1)[..., :-1],
)
```

For full-size anomaly images, follow the inference notebook pattern:

```python
imgs = [image_tensor.to(device)]
img_sizes = [image_tensor.shape[-2:]]
crops, origins = model.window_imgs_semantic(imgs)
mask_logits_per_layer, class_logits_per_layer = model(crops)
mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
pixel_logits = logits[0]
```

Use `torch.no_grad()`. Use AMP/autocast on CUDA if stable:

```python
with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
    ...
```

### Task 5: Implement Anomaly Scoring Methods

All scoring methods must return a `[H, W]` numpy array where higher score means more anomalous.

#### MSP

```python
probs = torch.softmax(pixel_logits / temperature, dim=0)
score = 1.0 - probs.max(dim=0).values
```

#### MaxLogit

```python
score = -pixel_logits.max(dim=0).values
```

If temperature is applied to MaxLogit, document it clearly. The required temperature experiment in the PDF is for MSP, so keep MaxLogit unscaled unless intentionally experimenting.

#### Max Entropy

```python
probs = torch.softmax(pixel_logits / temperature, dim=0)
score = -(probs * torch.log(probs + 1e-12)).sum(dim=0)
```

#### RbA

RbA is the mask-specific method. It should use query-level mask and class outputs, not only the final aggregated pixel logits.

Concept:

- Each object query acts like an independent one-vs-all classifier.
- Known pixels receive confident votes from at least one known-class query.
- Unknown pixels are "rejected by all" known classes.

Implementation direction:

```python
mask_probs = mask_logits.sigmoid()                       # [B, Q, H, W]
class_probs = class_logits.softmax(dim=-1)[..., :-1]     # [B, Q, C]
known_scores = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
known_scores = known_scores.clamp(0.0, 1.0)
rba_score = torch.prod(1.0 - known_scores, dim=1)        # [B, H, W]
```

Use log-space for numerical stability if needed:

```python
rba_score = torch.exp(torch.log1p(-known_scores.clamp(max=1 - 1e-6)).sum(dim=1))
```

Important: verify this against the official RbA repository before final reporting:

- https://github.com/NazirNayal8/RbA
- Check their `evaluate_ood.py` and scoring utilities.

If the exact RbA implementation differs, match the official implementation and document the formula in `step8_eomt_mask_baselines/README.md`.

### Task 6: Compute Metrics

After collecting all images for one dataset/method/checkpoint:

```python
ood_mask = gt == 1
ind_mask = gt == 0

scores = np.concatenate([anomaly_scores[ind_mask], anomaly_scores[ood_mask]])
labels = np.concatenate([
    np.zeros(num_ind_pixels),
    np.ones(num_ood_pixels),
])

auprc = average_precision_score(labels, scores) * 100.0
fpr95 = fpr_at_95_tpr(scores, labels) * 100.0
```

Ignore pixels with GT value `255`.

Use the same metric implementation for all methods and checkpoints.

### Task 7: Evaluate All Required Datasets

Run each method on each dataset:

```text
RoadAnomaly21
RoadObsticle21
RoadAnomaly
fs_static
FS_LostFound_full
```

Methods:

```text
msp
maxlogit
entropy
rba
```

Checkpoints:

```text
eomt_coco
eomt_cityscapes
eomt_finetuned
```

Total required anomaly runs:

```text
3 checkpoints x 5 datasets x 4 methods = 60 rows
```

### Task 8: Compute mIoU Per Checkpoint

The PDF table asks for mIoU for EoMT checkpoints. This mIoU does not change by post-hoc method.

Compute mIoU on Cityscapes validation for each checkpoint:

- COCO-trained EoMT.
- Cityscapes-trained EoMT.
- Fine-tuned EoMT.

Use the same semantic conversion pipeline:

```python
pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
prediction = pixel_logits.argmax(dim=0)
```

For COCO, be careful because the class space differs from Cityscapes. Use the same mapping/evaluation strategy already used in the earlier EoMT Cityscapes evaluation notebooks.

Save:

```text
step8_eomt_mask_baselines/eomt_miou_results.csv
```

Recommended columns:

```csv
checkpoint,config,miou,notes
```

### Task 9: Add Temperature Scaling

The PDF asks for a temperature scaling baseline.

Recommended approach:

1. Run EoMT once per image/checkpoint/dataset and save reusable logits.
2. Recompute MSP with different temperatures without running model forward again.
3. Try at least:

```text
T = 0.5
T = 0.75
T = 1.0
T = 1.1
```

Optional extended grid:

```text
T = 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0
```

MSP with temperature:

```python
probs = torch.softmax(pixel_logits / temperature, dim=0)
score = 1.0 - probs.max(dim=0).values
```

Save:

```text
step8_eomt_mask_baselines/eomt_temperature_results.csv
```

Recommended columns:

```csv
checkpoint,dataset,method,temperature,auprc,fpr95
```

### Task 10: Write the Step 8 README

Create `step8_eomt_mask_baselines/README.md` with:

- What Step 8 evaluates.
- How to install/run.
- Required checkpoints and datasets.
- Commands for one dataset and all datasets.
- Explanation of MSP, MaxLogit, Max Entropy, and RbA.
- Output CSV locations.
- Known limitations.

## Suggested Command Layout

One run:

```bash
python step8_eomt_mask_baselines/run_eomt_anomaly.py \
  --config eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --checkpoint eomt_checkpoints/eomt_coco.bin \
  --checkpoint-name eomt_coco \
  --dataset-root /path/to/Anomaly_Validation_Datasets \
  --dataset RoadAnomaly21 \
  --method rba \
  --output-csv step8_eomt_mask_baselines/eomt_anomaly_results.csv
```

All runs should be wrapped in:

```bash
bash step8_eomt_mask_baselines/run_all_eomt_anomaly.sh
```

The shell script should loop over:

- 3 checkpoints/configs.
- 5 datasets.
- 4 methods.

## Final Report Table Shape

The report should include a table like:

```text
Model / checkpoint | mIoU | Method | RA-21 AuPRC | RA-21 FPR95 | RO-21 AuPRC | RO-21 FPR95 | FS L&F AuPRC | FS L&F FPR95 | FS Static AuPRC | FS Static FPR95 | Road Anomaly AuPRC | Road Anomaly FPR95
```

Rows:

- `EoMT COCO - MSP`
- `EoMT COCO - MaxLogit`
- `EoMT COCO - Max Entropy`
- `EoMT COCO - RbA`
- `EoMT Cityscapes - MSP`
- `EoMT Cityscapes - MaxLogit`
- `EoMT Cityscapes - Max Entropy`
- `EoMT Cityscapes - RbA`
- `EoMT Fine-tuned - MSP`
- `EoMT Fine-tuned - MaxLogit`
- `EoMT Fine-tuned - Max Entropy`
- `EoMT Fine-tuned - RbA`

Add the ERFNet Step 7 rows separately if the final project table combines Step 7 and Step 8.

## Validation Checklist

Before considering Step 8 complete:

- [ ] `run_eomt_anomaly.py` runs on one image without crashing.
- [ ] The runner processes one full dataset with `maxlogit`.
- [ ] MSP, MaxLogit, and Max Entropy produce non-constant anomaly maps.
- [ ] RbA uses query-level mask/class outputs and has been checked against the official RbA implementation.
- [ ] Ignored pixels (`255`) are excluded from metrics.
- [ ] Ground-truth masks are correctly found for all five datasets.
- [ ] All 60 anomaly result rows are present.
- [ ] mIoU is computed once per checkpoint.
- [ ] Temperature scaling results are saved, if included.
- [ ] CSVs are deterministic enough to rerun and compare.
- [ ] Large saved logits/checkpoints/datasets are not committed to git.

## Main Risks

- Cityscapes config may be missing on the current branch. Add the needed config directly to this branch from official/project materials before Cityscapes and fine-tuned checkpoint evaluation.
- COCO and Cityscapes class spaces differ. Use the prior mapping/evaluation strategy for COCO mIoU.
- EoMT image preprocessing must match the notebook/config. Wrong resizing or normalization will make results meaningless.
- RbA should not be approximated from only final pixel logits unless explicitly documented as an approximation. It is intended for mask-query outputs.
- Full-size anomaly images can be memory heavy. Use batch size 1, sliding-window inference, and AMP on CUDA.
- Temperature scaling should reuse saved logits to avoid repeating expensive model inference.

## Recommended Implementation Order

1. Inspect the Step 7 runner from `origin/feature/erfnet-baselines` as read-only reference material.
2. Create `step8_eomt_mask_baselines/run_eomt_anomaly.py`.
3. Get EoMT loading and one-image inference working.
4. Add per-pixel logits conversion and `maxlogit`.
5. Add MSP and Max Entropy.
6. Add GT loading and metrics.
7. Run one full dataset for one checkpoint.
8. Implement and verify RbA.
9. Add all-dataset/all-checkpoint runner.
10. Add mIoU computation.
11. Add temperature scaling with saved logits.
12. Write `step8_eomt_mask_baselines/README.md`.
13. Generate final CSVs and report-ready tables.
