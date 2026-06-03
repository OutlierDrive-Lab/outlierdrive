#!/usr/bin/env bash
set -euo pipefail

# Keep datasets and checkpoints out of git. Override these env vars if your
# local paths differ.
DATASET_ROOT="${DATASET_ROOT:-Validation_Dataset}"
OUTPUT_CSV="${OUTPUT_CSV:-step8_eomt_mask_baselines/eomt_anomaly_results.csv}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-.venv/bin/python}"
MAX_IMAGES_ARG=()

if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi

DATASETS=(
  RoadAnomaly21
  RoadObsticle21
  RoadAnomaly
  fs_static
  FS_LostFound_full
)

# Format per line:
# checkpoint_name|config_path|checkpoint_path|miou
CHECKPOINTS=(
  "eomt_coco|eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml|eomt_checkpoints/eomt_coco.bin|"
  "eomt_cityscapes|eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml|eomt_checkpoints/eomt_cityscapes.bin|"
  "eomt_finetuned|eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml|eomt_checkpoints/epoch=9-step=1850.ckpt|"
)

for checkpoint_spec in "${CHECKPOINTS[@]}"; do
  IFS="|" read -r checkpoint_name config_path checkpoint_path miou <<< "${checkpoint_spec}"

  for dataset in "${DATASETS[@]}"; do
    echo "Running ${checkpoint_name} ${dataset} all methods"
    "${PYTHON}" step8_eomt_mask_baselines/run_eomt_anomaly.py \
      --config "${config_path}" \
      --checkpoint "${checkpoint_path}" \
      --checkpoint-name "${checkpoint_name}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --method all \
      --miou "${miou}" \
      --device "${DEVICE}" \
      --output-csv "${OUTPUT_CSV}" \
      ${MAX_IMAGES_ARG[@]+"${MAX_IMAGES_ARG[@]}"}
  done
done
