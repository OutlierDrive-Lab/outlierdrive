# Report Notes and Presentation Points

## Submission Constraints From The Project PDF

- Use the CVPR LaTeX template.
- Maximum 5 pages, excluding references.
- Include the public GitHub repository link at the end of the abstract.
- The report should be self-contained: introduce any referenced method before discussing it.
- Required structure:
  - Abstract
  - Introduction
  - Methodology
  - Experimental Results
  - Discussion
  - Conclusion
  - References

## Main Numbers To Report

Semantic segmentation:

- Cityscapes-trained EoMT on all 19 Cityscapes classes:
  - mIoU: 81.68%
  - Pixel accuracy: 96.72%
- COCO-trained EoMT mapped to Cityscapes overlap classes:
  - mIoU: 62.86%
  - Pixel accuracy: 90.68%
- Cityscapes-trained EoMT on the same overlap classes:
  - mIoU: 84.78%
  - Pixel accuracy: 97.13%

Important distinction:

- Do not say the COCO model reaches 62.86% on all 19 classes. It is overlap classes only.
- Do not say the fine-tuned model reaches 81.68% unless you have a separate CSV proving that exact fine-tuned checkpoint produced it.

Step 8 anomaly segmentation:

- `eomt_anomaly_results.csv`: 60 rows.
- `eomt_temperature_results.csv`: 60 rows.
- `eomt_all_results.csv`: 120 rows.

Best individual anomaly result:

- Checkpoint: `eomt_cityscapes`
- Dataset: `RoadObsticle21` / RoadObstacle21
- Method: Entropy
- AuPRC: 94.28
- FPR95: 0.35

Temperature scaling:

- Tested MSP with T = 0.5, 0.75, 1.0, 1.1.
- Best average temperature was T = 1.1 for all three checkpoints, but the improvement over T = 1.0 is very small.
- Explain this as a required baseline, not as a major improvement.

## How To Explain The Code

`run_eomt_anomaly.py`:

1. Loads the EoMT config and checkpoint.
2. Infers image size, number of classes, and query count from the checkpoint when possible.
3. Reads anomaly dataset images directly from `Validation_Dataset/<dataset>/images`.
4. Finds ground-truth masks in `labels_masks`.
5. Runs EoMT sliding-window semantic inference.
6. Converts mask logits and class logits into per-pixel logits.
7. Computes MSP, MaxLogit, Entropy, and RbA-style anomaly scores.
8. Collects all valid pixels, ignoring label 255.
9. Computes AuPRC and FPR95.
10. Writes one CSV row per checkpoint/dataset/method.

Temperature scaling:

- The script supports `--temperatures`.
- It computes the model logits once per image.
- It then recomputes MSP for each temperature from the same logits.
- This avoids repeating the expensive model forward pass.

`compute_cityscapes_miou.py`:

- This is only for semantic mIoU when prediction PNG masks already exist.
- It is separate from anomaly segmentation.
- It computes the confusion matrix over Cityscapes trainIds and returns mIoU and pixel accuracy.

## Suggested 5-Page Allocation

- Abstract: 1 paragraph.
- Introduction: half page.
- Methodology: 1.25 pages.
- Experimental Results: 1.5 pages.
- Discussion: 1 page.
- Conclusion: short paragraph.
- References: excluded from 5-page limit.

## What To Emphasize In Discussion

- Cityscapes-trained EoMT is strongest overall for anomaly segmentation because its training domain matches road scenes.
- COCO-trained EoMT is weaker because its class space and data distribution are not aligned with Cityscapes/anomaly road scenes.
- Fine-tuning helps some datasets but not all, so it should be discussed as dataset-dependent rather than universally better.
- Entropy works well because it captures uncertainty over the full class distribution.
- Temperature scaling changes the confidence calibration but does not strongly change the ranking of anomaly pixels in these results.

## Files To Cite In The Report

- `step4_eomt_eval/iou_results.csv`
- `step4_eomt_eval/coco_trained_overlap_iou.csv`
- `step4_eomt_eval/cityscapes_trained_overlap_iou.csv`
- `step8_eomt_mask_baselines/eomt_anomaly_results.csv`
- `step8_eomt_mask_baselines/eomt_temperature_results.csv`
- `step8_eomt_mask_baselines/eomt_all_results.csv`
