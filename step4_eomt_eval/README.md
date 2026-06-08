# Step 4 — EOMT Evaluation

This folder contains the evaluation pipeline for Cityscapes-trained EoMT model and COCO-trained EoMT model versions. It includes scripts for computing IoU scores, folders with CSV result files and test pipeline notebooks, and a notebook for visualization.

- **test_eval_pipeline** shows the process of building evaluation pipeline through careful exploration of the Cityscapes dataset. It considers both pretrained model versions and builds a map for a class space for COCO-trained EoMT model.

- **csv_results** contains the IoU scores and mIoU result of evaluation of both models on the Cityscapes validation dataset.

- the scripts **eomt_eval_iou.py** and **eomt_eval_overlap_iou.py** provide the standalone evaluation pipeline for evaluating Cityscapes-trained model on all 19 classes and COCO-trained model on the mapped class space.

- **visualization.ipynb** visualizes semantic segmentation and panoptic segmentation tasks on some validation images.

## Datasets folder structure

In order to execute the test notebooks and the scripts, use the following datasets folder organization:

```text
data/
├── test_eval_pipeline/
|   ├── gtFine_trainvaltest.zip
|   └── leftImg8bit_trainvaltest.zip
├── datasets_unzip/
├── eomt_pretrained_output/
|   ├── panoptic/
|   └── semantic/
└── eomt_valset_predictions/
    ├── cityscapes_model/
    └── coco_model/
```

Notice that for executing the evaluation pipeline you do not need to unzip any dataset folder. The folder **datasets_unzip** was used only for local testing and investigation. Folder **eomt_pretrained_output** saves visualizations on several validation images.

## Run evaluation pipeline

To evaluate EoMT Cityscapes trained model on all 19 classes, run the following command for **:

```bash
python /content/outlierdrive/step4_eomt_eval/eomt_eval_iou.py \
  --config /eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --data-path /path/to/dataset \
```
Replace `/path/to/dataset` with the directory containing the dataset zip files and set the directory for the tested checkpoints. If necessary, set the directory for saving the output. The script saves it in .csv format automatically.



