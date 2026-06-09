# Step 5 — EoMT Fine-Tuning Evaluation

This folder contains the training and evaluation pipeline for a fine-tuned **EoMT** model on semantic segmentation task.

The model was fine-tuned using several GPU environments:

- Google Colab T4 GPU
- Google Colab A100 GPU
- local NVIDIA GPU

## Fine-tuning

In order to fine-tune the model, run the following command:

```bash
python main.py fit \
    -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --data.path /path/to/dataset \
    --model.ckpt_path /eomt_coco.bin \
    --model.load_ckpt_class_head False \
    --model.network.num_q 200 \
    --data.img_size "[640,640]" \
    --data.num_workers 2 \
    --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
    --trainer.callbacks.filename "epoch={epoch}-step={step}" \
```

Notice that while resources allowed the model was trained on 16 batches which also could positively affect the results.

## Evaluation

For evaluating the fine-tuned model, use the pipeline built in the previous step:

```bash
python /content/outlierdrive/step4_eomt_eval/eomt_eval_iou.py \
  --config /eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --data-path /path/to/dataset \
  --img-size 640 640 \
  --num-q 200 \
```