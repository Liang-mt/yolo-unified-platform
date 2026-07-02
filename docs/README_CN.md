# 🚀 YOLO 统一训练平台

> **YOLOv5 / YOLOv8 / YOLOv10 / YOLOv11 / YOLO26 全系列统一框架**

一个集成化的 YOLO 模型训练、部署、分析平台，支持全系列 YOLO 模型的统一接口调用。

---

## ✨ 功能特性

| 模块 | 功能 | 状态 |
|------|------|------|
| 🧠 统一模型接口 | YOLOv5/v8/v10/v11/26 统一 API | ✅ |
| 📦 数据集工具 | 清洗、格式转换、自动划分 | ✅ |
| 🏋️ 训练引擎 | 单卡/多卡训练、自定义损失函数 | ✅ |
| ✂️ 模型剪枝 | 结构化/非结构化剪枝、BN-scale | ✅ |
| 📉 模型量化 | INT8/FP16 动态/静态量化 | ✅ |
| 📦 模型部署 | ONNX / TensorRT 导出 | ✅ |
| ⚡ 推理测速 | PyTorch/ONNX/TRT 多后端对比 | ✅ |
| 🌐 Web 界面 | Gradio 可视化操作界面 | ✅ |
| 📊 日志分析 | 训练曲线、过拟合检测、收敛分析 | ✅ |
| 🛠️ CLI 工具 | 命令行一键操作 | ✅ |

---

## 📁 项目结构

```
yolo-unified-platform/
├── gui/                    # PySide6 桌面 GUI
│   ├── main_window.py     # 主窗口（4 个 Tab）
│   ├── workers.py         # 后台线程
│   ├── styles.py          # QSS 样式
│   └── settings.py        # 可调配置
├── core/                   # 核心框架
│   ├── registry.py        # 全局注册器
│   ├── config.py          # 配置管理
│   ├── model_factory.py   # 模型工厂
│   ├── base_model.py      # 抽象基类
│   └── models/            # 各版本模型实现
├── data/                   # 数据集工具
│   ├── cleaner.py         # 数据集清洗
│   ├── converter.py       # 标注格式转换
│   ├── splitter.py        # 数据集划分
│   └── augmentor.py       # 数据增强
├── trainers/               # 训练引擎
│   ├── unified_trainer.py # 统一训练器
│   ├── multi_gpu_trainer.py # 多卡训练
│   └── callbacks.py       # 回调系统
├── losses/                 # 自定义损失函数
│   ├── focal_loss.py      # Focal / QFL / VFL / DFL
│   ├── ciou_loss.py       # CIoU / DIoU / GIoU / SIoU
│   ├── wasserstein_loss.py # Wasserstein / NWD
│   └── combined_loss.py   # 组合损失
├── pruning/                # 模型优化
│   ├── pruner.py          # 剪枝工具
│   └── quantizer.py       # 量化工具
├── deployment/             # 模型部署
│   ├── onnx_deployer.py   # ONNX 导出/推理
│   └── tensorrt_deployer.py # TensorRT 导出/推理
├── benchmark/              # 推理测速
│   └── speed_benchmark.py # 多后端速度对比
├── web/                    # Web 界面
│   └── app.py             # Gradio 应用
├── analysis/               # 日志分析
│   └── log_analyzer.py    # 训练日志分析器
├── configs/                # 配置文件
├── tools/                  # CLI 工具
│   └── cli.py
├── ultralytics/            # 本地 ultralytics 源码副本
├── models/                 # 预训练模型权重
├── docs/                   # 文档
│   ├── README_CN.md       # 中文详细文档
│   └── bugfix-log.md      # Bug 修复日志
├── main_gui.py             # GUI 入口
├── main.py                 # CLI 入口
├── 修改文档.md              # 代码修改记录
└── requirements.txt        # 依赖列表
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# GUI 额外需要
pip install PySide6 matplotlib
# 导出 ONNX 时显示 shape 信息（可选）
pip install onnx
```

### 2. 启动 GUI

```bash
python main_gui.py
```

### 3. 数据集准备

```bash
# VOC → YOLO 格式转换
python main.py data convert \
    --input path/to/voc/annotations \
    --output datasets/my_dataset \
    --source voc \
    --target yolo \
    --classes "cat,dog,bird"

# 数据集清洗
python main.py data clean \
    --images datasets/my_dataset/images \
    --labels datasets/my_dataset/labels

# 数据集划分
python main.py data split \
    --images datasets/my_dataset/images \
    --labels datasets/my_dataset/labels \
    --output datasets/my_dataset_split
```

### 4. 模型训练

```bash
# 单卡训练
python main.py train \
    --variant yolov8 \
    --size s \
    --data configs/custom_dataset.yaml \
    --epochs 100 \
    --batch 16

# 多卡训练 (通过 ultralytics DDP)
python -m torch.distributed.launch --nproc_per_node=4 main.py train \
    --variant yolov8 \
    --size m \
    --data configs/coco.yaml \
    --epochs 100
```

### 5. 模型导出

```bash
# 导出 ONNX
python main.py export --weights runs/train/exp/weights/best.pt --format onnx

# 导出 TensorRT (需要 NVIDIA GPU)
python main.py export --weights runs/train/exp/weights/best.pt --format engine --half
```

### 6. 推理测速

```bash
python main.py benchmark --weights yolov8s.pt --imgsz 640 --runs 100
```

### 7. 启动 Web 界面

```bash
python main.py web --port 7860
```

---

## 🖥️ 桌面 GUI

### 四个 Tab 页面

| Tab | 功能 | 详情 |
|-----|------|------|
| **Detect** | 图片/视频检测 | 左侧输入，右侧结果，视频支持暂停/停止 |
| **Train** | 模型训练 | 6 个实时图表 + 进度条 + Force Stop + cfg/hyp/optimizer |
| **Export** | 模型导出 | 多格式导出 + opset/dynamic/simplify + shape 详情 |
| **Benchmark** | 速度测速 | PyTorch/ONNX 多后端对比 |

### 训练页面特性

- **6 个实时图表**：box_loss、cls_loss、dfl_loss、lr、mAP（双线）、precision（双线）
- **进度信息**：epoch 进度条、已用时间、预计剩余时间、GPU 显存
- **Force Stop**：batch 级别响应，自动保存并压缩模型
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

---

## 📖 Python API 使用

```python
# ─── 训练 ────────────────────────────────────────────────────────────
from trainers.unified_trainer import UnifiedTrainer

trainer = UnifiedTrainer(
    variant="yolov8",
    size="s",
    num_classes=3,
    device="0",
)

results = trainer.train(
    data="configs/custom_dataset.yaml",
    epochs=100,
    batch_size=16,
)

# ─── 推理 ────────────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")
results = model.predict("test_image.jpg", conf=0.25)

# ─── 导出 ────────────────────────────────────────────────────────────
model.export(format="onnx", imgsz=640, simplify=True)

# ─── 测速 ────────────────────────────────────────────────────────────
from benchmark.speed_benchmark import SpeedBenchmark

bench = SpeedBenchmark()
result = bench.benchmark_pytorch(model.model, img_size=640)
bench.print_report({"pytorch": result})
```

### 自定义损失函数

```python
from losses import CIoULoss, FocalLoss, CombinedLoss

# 使用 CIoU Loss
ciou = CIoULoss()
loss = ciou(pred_boxes, target_boxes)

# 组合损失
combined = CombinedLoss(
    cls_loss={"type": "focal", "weight": 1.0, "alpha": 0.25, "gamma": 2.0},
    box_loss={"type": "ciou", "weight": 5.0},
    dfl_loss={"type": "dfl", "weight": 1.5},
)
```

### 模型剪枝与量化

```python
from pruning.pruner import ModelPruner
from pruning.quantizer import ModelQuantizer

# 剪枝
pruner = ModelPruner(model)
pruned = pruner.prune(amount=0.3, method="magnitude")

# 量化
quantizer = ModelQuantizer(model)
quantized = quantizer.quantize_dynamic()
speedup = quantizer.benchmark_speedup(model, quantized)
```

### 日志分析

```python
from analysis.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer("runs/train/exp")
analyzer.print_summary()
analyzer.plot_curves("analysis_output/")
analyzer.export_report("analysis_output/report.json")
```

---

## 🔧 配置文件

### 训练配置示例

```yaml
# configs/train_v8.yaml
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
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0
  workers: 8
  amp: true
  patience: 50

loss:
  cls:
    type: focal
    weight: 1.0
  box:
    type: ciou
    weight: 7.5
```

### 数据集配置示例

```yaml
# configs/custom_dataset.yaml
path: ./datasets/my_dataset
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: cat
  1: dog
  2: bird
```

---

## 📊 支持的模型

| 模型 | 版本 | 尺寸 | 特点 |
|------|------|------|------|
| YOLOv5 | 7.0+ | n/s/m/l/x | 经典稳定，社区资源丰富 |
| YOLOv8 | 8.0+ | n/s/m/l/x | 当前主流，精度速度均衡 |
| YOLOv10 | 10.0+ | n/s/m/l/x/b | NMS-free，端到端检测 |
| YOLOv11 | 11.0+ | n/s/m/l/x | 最新架构，性能提升 |
| YOLO26 | 26.0+ | n/s/m/l/x/c/e | 最新一代，极致性能 |

---

## 📦 支持的导出格式

| 格式 | 后缀 | 用途 |
|------|------|------|
| ONNX | .onnx | 通用部署，跨平台 |
| TensorRT | .engine | NVIDIA GPU 加速 |
| TorchScript | .torchscript | PyTorch 部署 |
| TFLite | .tflite | 移动端部署 |
| CoreML | .mlmodel | iOS 部署 |

---

## ⚠️ 常见问题

### Q: CUDA 内存不足？
```bash
# 减小 batch size
python main.py train --batch 8

# 使用更小的模型
python main.py train --size n
```

### Q: 训练完成后显存不释放？
训练结束后会自动调用 `gc.collect()` + `torch.cuda.empty_cache()` 释放显存。如果仍有残留，可在训练完成后手动执行：
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

### Q: 如何使用自定义数据集？
1. 准备图片和标注文件
2. 创建数据集 YAML 配置文件
3. 运行训练命令

### Q: 多卡训练失败？
```bash
# 检查 GPU
nvidia-smi

# 设置正确的 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 📄 文档

- [代码修改记录](../修改文档.md)
- [Bug 修复日志](bugfix-log.md)

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
