# YOLO Unified Platform — 训练图表与控制问题修复日志

## 问题概述

训练界面存在三个核心问题：

1. **训练输出重复** — 控制台每行输出都出现两次
2. **图表不更新** — 第二次训练开始，mAP/precision 图表空白
3. **强制停止无效** — Pause/Stop 按钮无法及时终止训练

---

## 问题 1：训练输出重复（每行出现两次）

### 现象

```
Ultralytics 8.4.21 🚀 Python-3.9.25 torch-2.7.0+cu128 CUDA:0 (...)
Ultralytics 8.4.21 🚀 Python-3.9.25 torch-2.7.0+cu128 CUDA:0 (...)
```

每一行都打印两遍。

### 根因

双重日志捕获。`TrainWorker.run()` 中做了两件事：

1. 替换 `sys.stdout` / `sys.stderr` 为自定义的 `EmitCapturer`
2. 向 ultralytics LOGGER 添加了一个额外的 `LogHandler`

```python
# 旧代码（有 bug）
sys.stdout = capturer  # ① LOGGER 的 StreamHandler 已经通过 stdout 流向 capturer
sys.stderr = capturer

log_handler = LogHandler(capturer)              # ② 又加了一个 handler
ultralytics_logger.addHandler(log_handler)      # 也写入同一个 capturer
```

ultralytics 的 LOGGER 自带一个 `StreamHandler(sys.stdout)`。当 `sys.stdout` 被替换为 `EmitCapturer` 后，LOGGER 的输出已经通过 `EmitCapturer.write()` 被捕获了。再添加 `LogHandler` 就导致**每条消息被同一个 capturer 处理两次**。

### 修复

删除多余的 `LogHandler`，只依赖 `EmitCapturer` 捕获 stdout/stderr：

```python
# 修复后
sys.stdout = capturer
sys.stderr = capturer

# 不再添加 LogHandler — LOGGER 的 StreamHandler 已经通过 stdout 流向 capturer
```

**文件**: `gui/workers.py` — `TrainWorker.run()` 方法

---

## 问题 2：第二次训练 mAP/precision 图表空白

### 现象

第一次训练所有图表正常。从第二次点击"Start Training"开始，box_loss / cls_loss / dfl_loss / lr 图表正常，但 mAP 和 precision 图表完全空白。

### 排查过程

#### 第一步：确认数据流

添加调试输出到 `_parse_metrics` 和 `_apply_metrics`，发现：

- **第一次训练**: `[VAL] epoch=1 map50=0.67` ✅ → `[GUI] mAP chart update` ✅
- **第二次训练**: `[VAL] epoch=10 map50=0.67` ❌ epoch 应该是 1 而不是 10

#### 第二步：定位状态残留

第二次训练时 `_cur_epoch=10`（来自第一次训练的最后一个 epoch），`_last_val_epoch=10`。

`_emit_val()` 的去重逻辑：
```python
def _emit_val(self):
    if self._last_val_epoch == self._cur_epoch:  # 10 == 10 → True
        return  # 跳过！验证数据不发射
```

#### 第三步：找到根因

ultralytics 的 LOGGER 在模块加载时创建 `StreamHandler(sys.stdout)`，**存储了当时 `sys.stdout` 对象的引用**。

```
第一次训练:
  sys.stdout → old_stdout (原始)
  LOGGER.StreamHandler.stream → old_stdout
  
  run() 开始:
    sys.stdout → capturer_A (worker_A)
    LOGGER.StreamHandler.stream → 仍然是 old_stdout !!（不是 capturer_A）
    
  run() 结束:
    sys.stdout → old_stdout (恢复)
```

等等，如果 LOGGER 的 handler 指向 old_stdout，那第一次训练的 LOGGER 输出怎么被捕获的？

实际上，第一次训练时 ultralytics 模块**还没有被导入**。当 `from ultralytics import YOLO` 执行时，模块加载，LOGGER 创建 StreamHandler，此时 `sys.stdout` 已经是 `capturer_A`。所以第一次训练是正常的。

```
第一次训练:
  run() 开始:
    sys.stdout → capturer_A (worker_A)
    from ultralytics import YOLO  ← 模块加载，LOGGER 创建 StreamHandler(capturer_A)
    LOGGER.StreamHandler.stream → capturer_A ✅
    
  run() 结束:
    sys.stdout → old_stdout (恢复)
    LOGGER.StreamHandler.stream → 仍然是 capturer_A ← 问题在这里！

第二次训练:
  run() 开始:
    sys.stdout → capturer_B (worker_B)
    LOGGER.StreamHandler.stream → 仍然是 capturer_A ← 指向旧 worker！
    
  LOGGER 输出验证结果 → 写入 capturer_A → worker_A._parse_metrics()
  worker_A._cur_epoch = 10, worker_A._last_val_epoch = 10
  → _emit_val() 去重跳过 → 验证数据不发射 → 图表空白
```

### 修复

每次训练开始时，更新 LOGGER 的 StreamHandler 指向新的 capturer：

```python
capturer = EmitCapturer(self, old_stdout)
sys.stdout = capturer
sys.stderr = capturer

# 关键修复：更新 LOGGER 的 handler 指向新的 capturer
ultralytics_logger = logging.getLogger("ultralytics")
for h in ultralytics_logger.handlers:
    if isinstance(h, logging.StreamHandler) and hasattr(h, 'stream'):
        h.stream = capturer
```

同时在训练结束后重置状态，确保下次训练干净：

```python
sys.stdout, sys.stderr = old_stdout, old_stderr
self._cur_epoch = None
self._last_val_epoch = -1
```

**文件**: `gui/workers.py` — `TrainWorker.run()` 方法

---

## 问题 3：训练输出信号洪水导致栈溢出

### 现象

训练过程中进程崩溃，退出码 `0xC00000FD`（Windows 栈溢出）。

### 根因

TQDM 进度条每秒用 `\r` 更新几十次。每次 `\r` 都触发 `_parse_metrics()`，匹配到训练行后发射 `epoch_done` 信号。信号洪水导致 Qt 事件队列溢出。

同时，在 `_parse_metrics` 中使用 `print()` 调试会导致**无限递归**：

```
print("debug") → sys.stdout (EmitCapturer) → _parse_metrics("debug")
→ 匹配 "all" → print("debug") → sys.stdout → _parse_metrics → ...
```

### 修复

1. **训练 loss 去重**：只有当 loss 值实际变化时才发射信号
2. **验证去重**：`_last_val_epoch` 追踪，每个 epoch 只发射一次
3. **禁止在 `_parse_metrics` 中使用 `print()`**：stdout 被捕获，写入会递归

```python
# 训练 loss：值变化才发射
if epoch != self._cur_epoch or box != self._cur_box or cls != self._cur_cls or dfl != self._cur_dfl:
    self.epoch_done.emit({...})

# 验证：每个 epoch 只发射一次
def _emit_val(self):
    if self._last_val_epoch == self._cur_epoch:
        return
    self._last_val_epoch = self._cur_epoch
    self.epoch_done.emit({...})
```

**文件**: `gui/workers.py` — `_parse_metrics()` 和 `_emit_val()` 方法

---

## 问题 4：Pause/Stop 按钮无法及时终止训练

### 玟象

点击 Stop 后训练继续运行，要等到当前 epoch 结束才生效。

### 根因

`on_train_epoch_start` 回调只在每个 epoch **开始**时触发。如果训练在某个 epoch 的第 3/8 个 batch，stop 标志要等到下一个 epoch 开始才被检查。

ultralytics 的训练循环中，`self.stop` 在每个 batch 结束后检查：

```python
# ultralytics/engine/trainer.py 第 495-497 行
self.run_callbacks("on_train_batch_end")
if self.stop:
    break  # 允许外部停止
```

### 修复

添加 `on_train_batch_end` 回调，在每个 batch 结束后检查 stop 标志：

```python
def on_train_batch_end(trainer):
    if self.stop_flag:
        trainer.stop = True

model.add_callback("on_train_batch_end", on_train_batch_end)
```

同时将 UI 的 Pause + Stop 两个按钮替换为单一的 "Force Stop" 按钮。

**文件**: `gui/workers.py` — 回调注册; `gui/main_window.py` — UI 按钮

---

## 修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `gui/workers.py` | 删除多余 LogHandler; 添加 LOGGER handler 更新; 训练 loss 即时发射 + 去重; 验证去重; 添加 on_train_batch_end 回调; 替换 pause/stop 为 force_stop |
| `gui/main_window.py` | 删除 Pause 按钮, Stop 改为 Force Stop; 删除 _pause_train; 图表更新改用 draw() + flush_events() |

---

## 关键教训

1. **Python logging.StreamHandler 存储 stream 对象引用**，不会动态跟踪 `sys.stdout` 的变化。替换 `sys.stdout` 后必须手动更新 handler 的 `stream` 属性。

2. **在 stdout 被重定向的代码中不能使用 `print()` 调试**。写入 stdout 会被捕获器处理，如果捕获器又触发了包含 print 的逻辑，就会无限递归。调试时应使用 `sys.__stderr__.write()`。

3. **TQDM 的 `\r` 会导致回调被频繁触发**。任何从 stdout 捕获中解析数据的逻辑都必须有去重机制，否则会导致信号洪水和栈溢出。

4. **跨训练运行的状态管理**：worker 对象在训练结束后可能被部分回收，但其引用的 capturer 仍被 LOGGER 持有。必须在训练结束时显式重置所有状态，并在新训练开始时更新所有外部引用。

---

## 附录：Force Stop 机制详解

### 两种停止方式的对比

#### ❌ 直接杀进程（常见错误做法）

```
训练进行中 → kill 进程 / thread.terminate() → 进程立即死亡
```

- 最后一次 `save_model()` 可能在 epoch 中途被调用
- 模型文件可能包含 optimizer state（Adam 优化器有 2 份参数副本，文件很大）
- 文件可能不完整或损坏
- 下次训练必须从头开始

#### ✅ 通过 trainer.stop 标志停止（本项目做法）

```
设置 trainer.stop = True
    → 当前 batch 完成
    → break 跳出 batch 循环
    → validate() 验证当前权重
    → save_model() 保存 last.pt 和 best.pt
    → break 跳出 epoch 循环
    → final_eval() 压缩模型 + 最终验证
    → 正常退出线程
```

### 完整代码流程追踪

**第 1 步：触发停止**（`gui/workers.py`）

用户点击 Force Stop → `self.stop_flag = True`

在每个 batch 结束时，回调触发：

```python
def on_train_batch_end(trainer):
    if self.stop_flag:
        trainer.stop = True  # 设置 ultralytics 内部的停止标志
```

**第 2 步：batch 循环退出**（`ultralytics/engine/trainer.py:495-497`）

```python
self.run_callbacks("on_train_batch_end")  # 回调在这里触发
if self.stop:
    break  # 当前 batch 完成后，立即跳出 batch 循环
```

**第 3 步：epoch 循环继续执行后续清理逻辑**（`trainer.py:510-555`）

虽然 batch 循环已 break，但 epoch 循环的后续代码仍会执行：

```python
self.run_callbacks("on_train_epoch_end")

# stop=True 时仍然做验证
if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
    self.metrics, self.fitness = self.validate()

# 保存模型
if self.args.save or final_epoch:
    self.save_model()  # 保存 last.pt（和 best.pt 如果更好）
```

**第 4 步：epoch 循环退出**（`trainer.py:554-555`）

```python
if self.stop:
    break  # 跳出 epoch 循环
```

**第 5 步：训练结束的清理工作**（`trainer.py:558-567`）

```python
self.final_eval()    # 用 best.pt 做最终验证 + strip optimizer
self.plot_metrics()  # 保存训练曲线图
```

**第 6 步：`final_eval()` 压缩模型**（`trainer.py:819-834`）

```python
def final_eval(self):
    ckpt = strip_optimizer(self.last)   # 从 last.pt 移除 optimizer state
    strip_optimizer(self.best)          # 从 best.pt 移除 optimizer state
    self.metrics = self.validator(model=self.best)  # 最终验证
```

### 模型文件大小分析

`save_model()` 保存的 checkpoint 包含：

```python
{
    "epoch": ...,
    "ema": deepcopy(self.ema.ema).half(),   # EMA 模型（FP16，较小）
    "optimizer": optimizer.state_dict(),     # Adam 优化器状态（很大！）
    "scaler": scaler.state_dict(),           # 混合精度 scaler
    ...
}
```

optimizer state 的大小约为模型本身的 **2-3 倍**（Adam 为每个参数维护 m 和 v 两个状态）。

`final_eval()` 中的 `strip_optimizer()` 会移除 optimizer state：

```python
def strip_optimizer(f, updates={}):
    ckpt = torch.load(f)
    for k in ('optimizer', 'updates'):
        ckpt[k] = None   # 删除 optimizer state
    torch.save(ckpt, f)
```

| 文件 | 保存时机 | 内容 | 大小 |
|------|----------|------|------|
| `last.pt` | 每个 epoch 结束 | 完整 checkpoint（含 optimizer） | ~11MB |
| `last.pt` | `final_eval` 后 | strip_optimizer 压缩 | ~5.5MB |
| `best.pt` | fitness 最佳时 | 同上 | ~5.5MB |

**这就是为什么停止后模型文件大小正常** — `strip_optimizer()` 在训练结束后把 optimizer state 移除了。

### 流程图

```
用户点击 Force Stop
    │
    ▼
self.stop_flag = True
    │
    ▼
on_train_batch_end 回调触发
    │
    ▼
trainer.stop = True
    │
    ▼
当前 batch 完成 ←── 不会中断正在运行的前向/反向传播
    │
    ▼
break 跳出 batch 循环
    │
    ▼
validate() ←── 用当前权重做验证
    │
    ▼
save_model() ←── 保存 last.pt（含 optimizer）
    │
    ▼
break 跳出 epoch 循环
    │
    ▼
final_eval()
    ├── strip_optimizer(last.pt)  ←── 压缩文件（移除 optimizer state）
    ├── strip_optimizer(best.pt)  ←── 压缩文件
    └── validate(best.pt)         ←── 最终验证
    │
    ▼
正常退出线程
```

### 关键设计思想

ultralytics 的训练循环在 `self.stop` 检查点之后保留了所有必要的清理代码。设置 `trainer.stop = True` **不是杀进程，而是让训练循环自己走完清理流程**。这保证了：

1. 模型文件完整（不会中途截断）
2. 模型文件已压缩（optimizer state 被移除）
3. 最优模型被正确保存（best.pt）
4. 训练指标被正确记录（results.csv）
5. GPU 内存被正确释放

---

## 问题 5：训练完成后 GPU 内存未释放

### 现象

训练完成后，任务管理器 / `nvidia-smi` 显示显存占用与训练时相同，没有下降。

### 根因

训练在 `TrainWorker.run()` 的子线程中执行。`model.train()` 返回后，`model`（YOLO 对象）和 `result`（训练结果）仍在 `run()` 的局部变量中被引用。Python 的垃圾回收器不会立即运行，且 YOLO 内部存在循环引用（model ↔ trainer），普通的引用计数无法回收。

```python
def run(self):
    ...
    model = YOLO(self.model_path)          # 加载模型到 GPU
    result = model.train(...)              # 训练，GPU tensors 分配
    # model 和 result 仍在作用域内
    # GPU tensors 不会被释放
    self.train_finished.emit(summary)      # 信号发射后 run() 返回
    # 但 gc 可能不会立即回收
```

### 修复

在 worker 线程中训练结束后，显式删除对象并强制垃圾回收：

```python
# 训练正常完成
del model, result
import gc
gc.collect()                    # 强制回收循环引用
if torch.cuda.is_available():
    torch.cuda.empty_cache()    # 让 CUDA 释放未使用的显存块
self.train_finished.emit(summary)
```

```python
# Force stop 中断
del model, result
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
self.train_finished.emit("\nTraining stopped.")
```

同时在主线程的 `_on_train_done()` 和 `_on_train_error()` 中也添加清理，作为双重保障：

```python
def _on_train_done(self, summary):
    ...
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 关键点：为什么需要 gc.collect()

`del model` 只删除了局部变量名 `model`，但 YOLO 对象内部存在循环引用：

```
YOLO.model → trainer
trainer.model → YOLO.model
trainer.ema → model
```

Python 的引用计数 GC 无法处理循环引用。必须调用 `gc.collect()` 触发分代垃圾回收器，遍历所有对象找到孤立的循环引用并释放。释放后，GPU tensor 的引用计数降为 0，PyTorch 的 CUDA 内存分配器才会真正释放显存。

`torch.cuda.empty_cache()` 的作用是让 PyTorch 把 CUDA 缓存的空闲内存块归还给 CUDA driver（而不是保留给下次分配）。不调用它的话，PyTorch 会保留这些块以加速下次分配，`nvidia-smi` 显示的占用不会下降。

**文件**: `gui/workers.py` — `run()` 方法; `gui/main_window.py` — `_on_train_done()` / `_on_train_error()`

---

## 附录：close_mosaic 说明

### 现象

训练途中有时输出 `Closing dataloader mosaic`，然后暂停几秒才能继续训练。不是每次都出现。

### 原因

`close_mosaic` 是 ultralytics 的数据增强参数（默认值 10），控制什么时候关闭 Mosaic 增强。

- Mosaic 增强：把 4 张图拼接成 1 张，增加多尺度、多目标学习能力
- `close_mosaic=10`：在最后 10 个 epoch 关闭 Mosaic，用正常图片训练

当到达关闭点时，ultralytics 重建 dataloader（去掉 Mosaic），这个过程需要几秒。

### 为什么不是每次都出现

| epochs | close_mosaic | 何时出现 |
|--------|-------------|----------|
| 10 | 10 | 第 1 epoch 就出现（全程关闭） |
| 100 | 10 | 第 91 epoch 出现 |
| 100 | 0 | 永远不出现 |

### 建议

保持默认 `close_mosaic=10`。关闭 Mosaic 在最后阶段有助于模型收敛到真实数据分布。

| 场景 | close_mosaic | 说明 |
|------|-------------|------|
| 默认 | 10 | 推荐 |
| 小数据集 | 5 | 数据少，早点关 |
| 大数据集 | 15~20 | 数据多，多练复杂场景 |
| 关闭 | 0 | 全程 Mosaic，一般不建议 |

---

## 附录：配置文件 gui/settings.py

所有可调参数集中在 `gui/settings.py`，修改后重启生效：

```python
# Loss 图表更新模式:
#   "epoch" — 每个 epoch 完成后画一个点（平滑，不抖动）
#   "batch" — 每个 batch 结束都更新当前点（实时，会抖动）
LOSS_UPDATE_MODE = "batch"

# 视频检测帧率模式:
#   True  — 和原视频帧率对齐
#   False — 识别多快就多快
VIDEO_SYNC_FPS = True
```

`main_window.py` 通过 `from . import settings` 导入，`workers.py` 由 `main_window.py` 传入参数，不需要直接导入 settings。

---

## 附录：训练 UI 优化（近期修改）

### 训练参数 UI 重构

**文件**: `gui/main_window.py` + `gui/workers.py`

**变更内容**：

1. **新增超参数控件**（2×2 网格）：LR Factor (lrf)、Momentum、Weight Decay、Warmup Epochs
2. **删除 cfg/hyp 按钮**：ultralytics v8 不支持 `hyp=` 参数，cfg 功能保留但按钮移除
3. **模型缩放选择器**：Train 页面新增 n/s/m/l/x 下拉框，对 `.yaml` 架构文件自动拼接缩放后缀
4. **Browse 支持 .yaml**：文件选择对话框现在支持 `.pt` 和 `.yaml` 文件
5. **Worker 生命周期优化**：训练开始前断开旧 worker 信号 + deleteLater()
6. **Benchmark 输出模型名**：JSON 输出第一行显示当前测试的模型文件名

### 模型选择联动

**文件**: `gui/main_window.py`

Train/Export/Benchmark 的模型选择逻辑：
- 下拉框选择内置 `.pt` 模型 → 直接使用
- Browse 选择自定义文件 → 名称替换到下拉框显示
- 下拉框选内置模型 → 自动清除自定义路径
- 模型路径解析：`_resolve_model(box, custom_path, scale_box)` 优先级：自定义路径 > 下拉框 > None

### 超参数传递

**文件**: `gui/workers.py`

```python
# 训练参数直接作为关键字参数传给 model.train()
train_kwargs = dict(
    data=self.data_path, epochs=self.epochs,
    batch=self.batch, imgsz=self.imgsz,
    lr0=self.lr, lrf=self.lrf,
    momentum=self.momentum,
    weight_decay=self.weight_decay,
    warmup_epochs=self.warmup_epochs,
    device=self.device,
    optimizer=self.optimizer,
    resume=self.resume, cache=self.cache,
)
result = model.train(**train_kwargs)
```

注意：ultralytics v8 的 `model.train()` 不支持 `hyp=` 参数。如果传入 `hyp=` 会报 `SystemExit`（未知配置键）。超参数必须展开为关键字参数传递。
