#!/usr/bin/env bash
set -euo pipefail

# Example MSP temperature sweep. Override these env vars for another checkpoint.
DATASET_ROOT="${DATASET_ROOT:-Validation_Dataset}"
OUTPUT_CSV="${OUTPUT_CSV:-step8_eomt_mask_baselines/eomt_temperature_results.csv}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-.venv/bin/python}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-eomt_coco}"
CONFIG_PATH="${CONFIG_PATH:-eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-eomt_checkpoints/eomt_coco.bin}"

DATASETS=(
  RoadAnomaly21
  RoadObsticle21
  RoadAnomaly
  fs_static
  FS_LostFound_full
)

TEMPERATURES=(
  0.5
  0.75
  1.0
  1.1
)

MAX_IMAGES_ARG=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi

for dataset in "${DATASETS[@]}"; do
  for temperature in "${TEMPERATURES[@]}"; do
    echo "Running ${CHECKPOINT_NAME} ${dataset} MSP temperature=${temperature}"
    "${PYTHON}" step8_eomt_mask_baselines/run_eomt_anomaly.py \
      --config "${CONFIG_PATH}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --checkpoint-name "${CHECKPOINT_NAME}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --method msp \
      --temperature "${temperature}" \
      --device "${DEVICE}" \
      --output-csv "${OUTPUT_CSV}" \
      ${MAX_IMAGES_ARG[@]+"${MAX_IMAGES_ARG[@]}"}
  done
done
