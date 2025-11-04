## CottonBudYOLOv2 · Cotton Bud Detection/Localization (v2)

---

### Overview

- Purpose: robust cotton bud detection/localization in real-field scenarios (occlusion, small objects, complex backgrounds, single/multi-row), with practical engineering optimizations (speed, CUDA, visualization).
- Based on `ultralytics` families (YOLOv8/YOLOv11/RT-DETR) with customized training/inference/visualization.
- End-to-end pipeline: training, inference (image/video/dir/webcam), heatmap visualization, and domain-gap evaluation in pure inference mode.

---

### Features

- Cross-domain robustness enhancements.
- Unified inference entry `infer.py` with visualization and YOLO txt export.
- Heatmap overlay or standalone map via `utils/heatmap_generator.py`.
- Pure-inference evaluation (`gap_val_inference.py`) reporting Precision/Recall/mAP (approx.)/Fitness.
- Easily switch between `YOLO` and `RTDETR` configs and weights.

---

### Environment

```bash
conda create -n YOLOv11_UP python=3.10
pip install ultralytics==8.3.193
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Make sure your CUDA driver matches the installed PyTorch CUDA build.

---

### Datasets

- Standard YOLO layout: `images/{train,val,test}` and `labels/{train,val,test}`.
- Follow examples in `ultralytics/cfg/datasets/*.yaml` and update the `path/train/val/test` fields.
- For domain-gap evaluation, prepare two YAMLs (e.g., 2025 as source, 2024 as target).

---

### Quick Start

#### Inference

Unified entry for image/video/directory/webcam:

```bash
python infer.py --weights runs/train/your/weights/best.pt --source images_test/test_one/cotton.jpg --heatmap --name exp
python infer.py --weights runs/train/your/weights/best.pt --source images_test/test_one/ --save-txt --save-conf --name batch
python infer.py --weights best.pt --source 0 --view-img --name webcam
```

Key args:
- `--conf-thres` confidence threshold (default 0.25)
- `--iou-thres` NMS IOU threshold (default 0.45)
- `--heatmap` enable heatmap; `--heatmap-colormap/decay/alpha` to style
- `--save-txt/--save-conf` export YOLO-format predictions

#### Training

Both `YOLO` and `RTDETR` APIs are supported. Example:

```python
from ultralytics import RTDETR
model = RTDETR('rtdetr-resnet50.yaml')
model.train(data='path/to/dataset.yaml', optimizer='auto', seed=0,
            epochs=100, imgsz=640, device='0', batch=4, workers=2,
            patience=60, project='runs/train', name='your-exp')
```

#### Domain Gap Validation (Pure Inference)

```bash
python gap_val_inference.py \
  --weights runs/train/your-exp/weights/best.pt \
  --source-data path/to/source.yaml \
  --target-data path/to/target.yaml \
  --imgsz 640 --conf 0.25 --iou 0.25 --iou-thres 0.5 --device 0 \
  --name gap_eval
```

Outputs include Precision / Recall / mAP@0.5 (F1 approx.) / mAP@0.5:0.95 (approx.) / Fitness, saved under `runs/gap_val/{name}`.

---

### Project Structure

```text
CottonBudYOLOv2/
├─ infer.py                     # Inference & visualization entry
├─ gap_val_inference.py         # Domain-gap evaluation (pure inference)
├─ train.py / train_domain.py   # Training samples/variants
├─ utils/
│  ├─ inference_visualizer.py   # Inference/visualization wrapper
│  └─ heatmap_generator.py      # Heatmap generation & blending
├─ tools/                       # Visualization & analysis tools
└─ ultralytics/                 # Upstream framework snapshot/adaptation
```

---

### Visualization

- Bounding boxes/labels with FPS and detection stats.
- With `--heatmap`, export overlay and standalone heatmap (`*_heatmap`).

---

### Roadmap / FAQ / License / Citation

- Roadmap: cross-domain learning, quantization/pruning/RT-optimizations, ensemble/auto-selection.
- FAQ: lower `--imgsz`/batch for OOM, check video path/codec, ensure `--heatmap` is set and alpha not too small.
- License: follow this repo and upstream `ultralytics` licenses.
- Citation:

```bibtex
@misc{CottonBudYOLOv2,
  title  = {CottonBudYOLOv2: Cotton Bud Detection and Localization},
  year   = {2025},
  howpublished = {GitHub},
  note   = {https://github.com/your/repo}
}
```


