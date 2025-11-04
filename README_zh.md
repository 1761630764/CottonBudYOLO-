## CottonBudYOLOv2 · 棉花顶芽检测/定位（v2）

---

### 项目简介 · Overview

- **目标**：面向田间环境的棉花顶芽检测与定位，强化跨域鲁棒性（遮挡、小目标、复杂背景、单/多行场景），并面向工程应用优化推理速度、CUDA 加速与可视化能力。
- **基座**：基于 `ultralytics` 系列（YOLOv8/YOLOv11/RT-DETR 等）进行二次开发与工程化适配。
- **本仓库**：包含训练、推理（图片/视频/文件夹/摄像头）、热力图可视化、跨域评估（纯推理模式）等完整流程。

---

### 特性 · Features

- **跨域检测强化**：针对不同采集年份/光照/场景的稳健性优化。
- **工程化推理**：统一入口 `infer.py`，支持图片/视频/文件夹/摄像头，保存可视化与 YOLO txt 结果。
- **热力图可视化**：可选启用叠加或纯热力图导出（`utils/heatmap_generator.py`）。
- **纯推理评估**：`gap_val_inference.py` 进行源/目标域数据集的快速度量（Precision/Recall/mAP 近似/Fitness）。
- **可扩展模型**：支持 `YOLO` 与 `RTDETR` 配置与权重，便于对比实验。

---

### 运行环境 · Environment

```bash
conda create -n YOLOv11_UP python=3.10
python==3.10
pip install ultralytics==8.3.193
pip show ultralytics  # 验证版本

# NVIDIA 环境（示例）
# Driver 576.88 · CUDA 12.8 · cuDNN 8.9.7.29

# PyTorch（CUDA 12.8 对应示例）
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

> 提示：确保本地 CUDA/驱动与安装的 PyTorch CUDA 版本匹配，否则请参照 PyTorch 官方说明选择合适的索引与版本。

---

### 数据集 · Datasets

- 使用标准 YOLO 数据格式组织：`images/{train,val,test}` 与 `labels/{train,val,test}`；
- 数据集配置示例参考 `ultralytics/cfg/datasets/*.yaml`；将路径字段 `path/train/val/test` 指向本地数据；
- 跨域评估建议准备两份 `.yaml`：如 `cotton_gap_2025.yaml`（源域）与 `cotton_gap_2024.yaml`（目标域）。

---

### 快速开始 · Quick Start

#### 推理 · Inference

图片/视频/文件夹/摄像头统一入口：

```bash
# 单张图片 + 热力图
python infer.py \
  --weights runs/train/yolov11n_DN_ALL_cotton/weights/best.pt \
  --source images_test/test_one/cotton.jpg \
  --heatmap --name exp_cotton

# 文件夹批处理 + 保存 txt 标签
python infer.py \
  --weights runs/train/yolov11n_DN_ALL_cotton/weights/best.pt \
  --source images_test/test_one/ \
  --save-txt --save-conf --name batch_exp

# 视频/摄像头（传入设备编号，如 0）
python infer.py --weights best.pt --source 0 --view-img --name webcam
```

关键参数：
- `--conf-thres` 置信度阈值（默认 0.25）
- `--iou-thres` NMS IOU 阈值（默认 0.45）
- `--heatmap` 是否启用热力图；`--heatmap-colormap/decay/alpha` 控制风格
- `--save-txt/--save-conf` 导出 YOLO 格式预测

#### 训练 · Training

两种风格均可（YOLO / RT-DETR），示例见代码注释：

```bash
# 示例：使用 RT-DETR 训练（见 gap_val_inference.py 中的格式对齐）
python - <<'PY'
from ultralytics import RTDETR
model = RTDETR('rtdetr-resnet50.yaml')
model.train(
  data='path/to/your_dataset.yaml',
  optimizer='auto', seed=0, epochs=100, imgsz=640, device='0',
  batch=4, workers=2, patience=60,
  project='runs/train', name='your-exp-name'
)
PY
```

> 也可按 `ultralytics` 官方 CLI 方式训练，或将 `train.py`/Notebook 自行整理为脚本入口。

#### 跨域评估（纯推理）· Domain Gap Validation (Pure Inference)

```bash
python gap_val_inference.py \
  --weights runs/train/your-exp/weights/best.pt \
  --source-data path/to/source.yaml \
  --target-data path/to/target.yaml \
  --imgsz 640 --conf 0.25 --iou 0.25 --iou-thres 0.5 --device 0 \
  --name gap_eval
```

输出包括 Precision / Recall / mAP@0.5（F1 近似）/ mAP@0.5:0.95（近似）/ Fitness 等统计，并保存到 `runs/gap_val/{name}`。

---

### 目录结构 · Project Structure

```text
CottonBudYOLOv2/
├─ infer.py                     # 推理与可视化入口（图片/视频/文件夹/摄像头）
├─ gap_val_inference.py         # 跨域评估（纯推理）
├─ train.py / train_domain.py   # 训练脚本样例/变体
├─ utils/
│  ├─ inference_visualizer.py   # 推理/可视化封装（叠加热力图、保存 txt 等）
│  └─ heatmap_generator.py      # 热力图生成/融合
├─ tools/                       # 可视化与分析工具
└─ ultralytics/                 # 上游框架（本地快照/适配）
```

---

### 结果可视化 · Visualization

- 叠加边框与标签，显示 FPS/检测数量等统计。
- 启用 `--heatmap` 输出叠加图与纯热力图（文件名带 `_heatmap`）。

---

### 路线图 · Roadmap

- [ ] 更系统的跨域自适应/稳健训练策略（数据增广/损失/蒸馏/重标定）。
- [ ] 量化/裁剪/张量 RT 优化以进一步提升部署性能。
- [ ] 模型集成与自动化模型选择（依据场景元数据）。

> 参考：`项目说明.md` 中 v1→v2 的阶段性目标描述。

---

### 常见问题 · FAQ

- **推理显存溢出？** 调低 `--imgsz`、`--batch`（视频流），或使用更小模型权重；确认 CUDA 与驱动匹配。
- **打开视频失败？** 确认路径与编码，或传入摄像头编号（如 `--source 0`）。
- **看不到热力图？** 确认传入 `--heatmap`，并检查 `--heatmap-alpha` 是否过小。

---

### 许可与致谢 · License & Acknowledgements

- 许可：遵循本仓库与上游 `ultralytics` 的相应 License（详见上游仓库）。
- 致谢：感谢 `ultralytics` 团队与社区、以及相关数据集的贡献者。

---

### 引用 · Citation

如果本项目对您的研究或产品有帮助，欢迎引用/标注本仓库与上游 `ultralytics`：

```bibtex
@misc{CottonBudYOLOv2,
  title  = {CottonBudYOLOv2: Cotton Bud Detection and Localization},
  year   = {2025},
  howpublished = {GitHub},
  note   = {https://github.com/your/repo}
}
```


