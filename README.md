# 🚀 YOLO 统一训练平台

> **YOLOv5 / YOLOv8 / YOLOv10 / YOLOv11 / YOLO26 — 全系列统一框架**

一个集成化的 YOLO 模型训练、部署、分析平台，支持全系列 YOLO 模型的统一接口调用。

## 🎬 Demo

<video src="https://private-user-images.githubusercontent.com/147730960/616556862-38df9014-9dde-415d-b437-59d925adb672.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODMwMzQ5ODgsIm5iZiI6MTc4MzAzNDY4OCwicGF0aCI6Ii8xNDc3MzA5NjAvNjE2NTU2ODYyLTM4ZGY5MDE0LTlkZGUtNDE1ZC1iNDM3LTU5ZDkyNWFkYjY3Mi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNzAyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDcwMlQyMzI0NDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kZWE1M2Q1N2Q5NTBmN2ZkYjI4Yzc4YmIxY2ZlN2NmZDA5NzVjZDBjZWFlYjhhMzczMjBkZWEyNTEzYTc3NzRmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.4LMsEkp9m8rXZ-HRlv4yWIQdCAiLppL2zHRTgbbYgAI" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>

---

## ✨ 功能特性

| 模块 | 功能 | 状态 |
|------|------|------|
| 🧠 统一模型接口 | YOLOv5/v8/v10/v11/26 全版本支持 | ✅ |
| 📦 数据集工具 | VOC/COCO/LabelMe/YOLO 格式互转、清洗、自动划分 | ✅ |
| 🏋️ 训练引擎 | 单卡/多卡训练、实时训练曲线、Force Stop | ✅ |
| 📉 自定义损失 | Focal/CIoU/DIoU/SIoU/Wasserstein 等 12 种 | ✅ |
| ✂️ 模型剪枝量化 | 结构化/非结构化剪枝、INT8/FP16 量化 | ✅ |
| 📦 模型部署 | ONNX / TensorRT 导出与推理 | ✅ |
| ⚡ 速度测速 | PyTorch/ONNX/TRT 多后端对比 | ✅ |
| 🖥️ 桌面 GUI | PySide6 桌面应用（检测/训练/导出/测速） | ✅ |
| 🌐 Web 界面 | Gradio Web UI（中英文双语） | ✅ |
| 📊 日志分析 | 训练曲线、过拟合检测、收敛分析 | ✅ |
| 🛠️ CLI 工具 | 命令行一键操作 | ✅ |

---

## 📁 项目结构

```
yolo-unified-platform/
├── gui/                         # PySide6 桌面 GUI
│   ├── main_window.py          # 主窗口（4 个 Tab）
│   ├── workers.py              # 后台线程（推理/训练/导出/视频/测速）
│   ├── styles.py               # QSS 样式
│   └── settings.py             # 可调配置（loss更新模式/视频帧率等）
├── web/                         # Gradio Web UI
│   └── app.py                  # Web 界面（中英文双语）
├── core/                        # 核心框架
│   ├── registry.py             # 全局注册器（9 个注册表）
│   ├── config.py               # YAML 配置管理
│   ├── model_factory.py        # 模型工厂
│   ├── base_model.py           # 抽象基类
│   └── models/                 # 5 个模型包装器
├── data/                        # 数据集工具
│   ├── cleaner.py              # 数据集清洗（自动修复）
│   ├── converter.py            # 格式转换（VOC/COCO/LabelMe/YOLO）
│   ├── splitter.py             # 数据集划分 + YAML 生成
│   └── augmentor.py            # 数据增强（Mosaic/MixUp/CopyPaste）
├── trainers/                    # 训练引擎
│   ├── unified_trainer.py      # 统一训练器
│   ├── multi_gpu_trainer.py    # 多卡训练（DDP/DP）
│   └── callbacks.py            # 回调系统
├── losses/                      # 自定义损失函数
│   ├── focal_loss.py           # Focal / QualityFocal / Varifocal / DFL
│   ├── ciou_loss.py            # GIoU / DIoU / CIoU / SIoU
│   ├── wasserstein_loss.py     # Wasserstein / NWD
│   └── combined_loss.py        # 组合损失 + EIoU
├── pruning/                     # 模型剪枝量化
│   ├── pruner.py               # 剪枝（magnitude/random/BN-scale）
│   └── quantizer.py            # 量化（INT8/FP16）
├── deployment/                  # 模型部署
│   ├── onnx_deployer.py        # ONNX 导出 + 推理
│   └── tensorrt_deployer.py    # TensorRT 引擎构建 + 推理
├── benchmark/                   # 速度测速
│   └── speed_benchmark.py      # 多后端对比
├── analysis/                    # 日志分析
│   └── log_analyzer.py         # 训练曲线/过拟合/收敛分析
├── tools/                       # CLI 工具
│   └── cli.py                  # Click CLI
├── ultralytics/                 # 本地 ultralytics 源码副本
├── models/                      # 预训练模型权重
│   ├── yolov5n.pt
│   ├── yolov8n.pt / yolov8s.pt
│   ├── yolov10n.pt
│   ├── yolo11n.pt
│   └── yolo26n.pt / yolo26s.pt
├── configs/                     # 配置文件模板
├── datasets/                    # 数据集 YAML
├── docs/                        # 文档
│   ├── README_CN.md            # 中文详细文档
│   └── bugfix-log.md           # Bug 修复日志
├── main_gui.py                  # GUI 入口
├── main.py                      # CLI 入口
├── 修改文档.md                   # 代码修改记录
└── requirements.txt             # 依赖列表
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
# GUI 额外需要
pip install PySide6 matplotlib
# 导出 ONNX 时显示 shape 信息（可选）
pip install onnx
```

### 启动 GUI

```bash
python main_gui.py
```

### 启动 Web UI

```bash
python main.py web --port 7860
```

### 命令行训练

```bash
python main.py train --variant yolov8 --size s --data configs/custom_dataset.yaml --epochs 100
```

### 命令行导出

```bash
python main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

---

## 🖥️ 桌面 GUI 功能

### 四个 Tab 页面

| Tab | 功能 | 详情 |
|-----|------|------|
| **Detect** | 图片/视频检测 | 左侧输入，右侧结果，视频支持暂停/停止 |
| **Train** | 模型训练 | 6 个实时图表 + 进度条 + Force Stop + cfg/hyp/optimizer |
| **Export** | 模型导出 | 多格式导出 + opset/dynamic/simplify + shape 详情 |
| **Benchmark** | 速度测速 | PyTorch/ONNX 多后端对比 |

### Detect 页面

![Detect](./datasets/detect_fire.png)

- 支持图片和视频文件
- 视频逐帧检测，左右同步显示原视频和检测结果
- 支持暂停/继续/停止
- 可选帧率同步或实时检测速度

### 训练页面

![Train](./datasets/train.png)

- **6 个实时图表**：box_loss、cls_loss、dfl_loss、lr、mAP（mAP50 + mAP50-95）、precision（Precision + Recall）
- **进度信息**：epoch 进度条、已用时间、预计剩余时间、GPU 显存
- **Force Stop**：点击后在当前 batch 完成后立即停止，自动保存并压缩模型（best.pt + last.pt）
- **GPU 内存管理**：训练结束自动释放显存
- **训练参数**：Optimizer 选择、Cache Images、Resume Training、cfg/hyp 文件
- **close_mosaic**：默认 10，最后 10 个 epoch 关闭 Mosaic 增强（训练中会暂停几秒重建 dataloader，属正常现象）

### 配置文件

可调参数集中在 `gui/settings.py`，修改后重启生效：

```python
# gui/settings.py

# Loss 图表更新模式:
#   "epoch" — 每个 epoch 完成后画一个点（平滑，不抖动）
#   "batch" — 每个 batch 结束都更新当前点（实时，会抖动）
LOSS_UPDATE_MODE = "batch"

# 视频检测帧率模式:
#   True  — 和原视频帧率对齐（播放速度 = 原视频速度）
#   False — 识别多快就多快（最快速度，不等待）
VIDEO_SYNC_FPS = True
```

### 导出页面

- **格式**：ONNX / TorchScript / TensorRT / TFLite / CoreML / Paddle
- **参数**：Image Size、ONNX Opset、FP16、Dynamic Batch、Simplify
- **导出详情**：input shape、output shape、文件大小、ONNX opset 版本

---

## 📊 训练输出机制

训练输出通过 **stdout/stderr 重定向** 捕获：

```
ultralytics 训练输出
    ↓
sys.stdout (EmitCapturer)
    ↓ _parse_metrics()
训练 loss: 值变化时即时发射
验证 mAP:  每 epoch 去重发射
学习率:    on_train_epoch_start 回调
    ↓
epoch_done 信号 → GUI 图表实时更新
```

- LOGGER 的 StreamHandler 在每次训练开始时更新指向新的 capturer
- 训练结束后自动释放 GPU 内存（`gc.collect()` + `cuda.empty_cache()`）

---

## ⚡ Force Stop 机制

Force Stop 不是杀进程，而是设置 `trainer.stop = True` 标志，让 ultralytics 训练循环自己走完清理流程：

```
Force Stop → trainer.stop = True
    → 当前 batch 完成后 break
    → validate() + save_model()
    → final_eval() + strip_optimizer()
    → 释放 GPU 内存
    → 正常退出
```

**优势**：
- 模型文件完整（不会中途截断）
- 模型已压缩（optimizer state 被移除，~5.5MB 而非 ~11MB）
- best.pt 和 last.pt 都正确保存
- GPU 内存正确释放

---

## 📦 支持的 YOLO 版本

| 版本 | 文件名格式 | 预训练模型 |
|------|-----------|-----------|
| YOLOv5 | yolov5n/s/m/l/x.pt | ✅ |
| YOLOv8 | yolov8n/s/m/l/x.pt | ✅ |
| YOLOv10 | yolov10n/s/m/l/x/b.pt | ✅ |
| YOLOv11 | yolo11n/s/m/l/x.pt | ✅ |
| YOLO26 | yolo26n/s/m/l/x/c/e.pt | ✅ |

---

## 🔧 配置文件

### 训练配置示例

```yaml
model:
  variant: yolov8
  size: s
  pretrained: true
  num_classes: 80

train:
  epochs: 100
  batch_size: 16
  img_size: 640
  optimizer: auto
  lr0: 0.01
```

### 数据集配置示例

```yaml
path: ./datasets/my_dataset
train: train/images
val: val/images
nc: 3
names: [cat, dog, bird]
```

---

## ⚠️ 已知问题

1. `core/models/` 和 `ModelFactory` 已实现但未被 GUI/CLI 实际使用，直接调用 ultralytics
2. `losses/` 模块的自定义损失函数是独立实现的，未集成到训练流程
3. `pruning/` 和 `quantization/` 是独立工具，未集成到 GUI
4. TensorRT 部署的输出形状硬编码为 YOLOv8 格式

---

## 📄 文档

- [中文详细文档](docs/README_CN.md)
- [代码修改记录](修改文档.md)
- [Bug 修复日志](docs/bugfix-log.md)

---

## 📄 许可证

MIT License
