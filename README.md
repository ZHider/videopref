# PyAI — 个人视频喜好二分类器

轻量级、高度个性化的**视频喜好二分类器**。采用"人机协同"工作流：先自动化拆帧提取关键视觉片段 → 用户在文件系统中人工清洗低质帧 → 基于清洗后的帧序列进行特征提取与喜好预测。

**核心架构**：冻结 **DINOv3** 骨干 + **Masked Attention Pooling** + 轻量 **MLP 分类头**，面向极少标注样本下的快速收敛与本地化部署。

## 工作流

```
视频路径列表(每行一个) ─▶ 🏷️ 标注Tab：一键拆帧 ─▶ frames/{video}/0001.jpg ...
                               │ (逐视频看图，点 喜欢/不喜欢)
                               ▼
                         labels.json (导出)
                               ▼
                   训练(3.4) ─▶ Checkpoint ─▶ 🔍 推理Tab(3.5)
```

拆帧与推理完全解耦，以 **`frames/` 文件夹结构作为两阶段间的唯一数据契约**。
支持三种抽帧模式：**`uniform`**（时长自适应均匀，默认）、**`keyframe`**（只解 I 帧，快 ~10 倍，适合批量）、**`scene`**（场景检测）。

## 目录结构

```
.
├── videopref/            # 主包
│   ├── config.py         # 集中配置 / 数据契约常量
│   ├── paths.py          # 路径解析、安全化命名、帧枚举
│   ├── frames.py         # 3.1 拆帧（uniform/keyframe/scene + 分辨率上限 + 并行）
│   ├── backbone.py       # 冻结 DINOv3 骨干加载
│   ├── features.py       # 特征提取管线（[CLS] 逐帧特征）
│   ├── model.py          # 模型类（MaskedAttentionPooling/分类头）+ Checkpoint 规范
│   ├── augment.py        # 训练期数据增强
│   ├── dataset.py        # 训练数据 + 特征缓存（含帧签名失效校验）
│   ├── labeling.py       # 标注工作流（路径列表→拆帧→队列→labels.json）
│   ├── batch_infer.py    # 批量抽帧+推理（成千上万个视频）
│   ├── train.py          # 3.4 CLI 训练
│   ├── inference.py      # 推理辅助
│   └── gradio_app.py     # 3.5 Gradio 三 Tab（拆帧/标注/推理）
├── train.py              # CLI 训练入口
├── infer_batch.py        # 批量推理入口
├── app.py                # Gradio 入口
├── scripts/
│   ├── make_synthetic_data.py   # 合成测试数据
│   └── smoke_backbone.py        # 骨干冒烟测试
├── models/               # 下载的 DINOv3 权重
├── frames/               # 拆帧输出 + 人工清洗工作区
├── checkpoints/          # 训练产出
└── features_cache/       # 训练特征缓存
```

## 安装

```bash
uv sync                              # 安装 torch/torchvision/transformers/gradio 等
uv pip install modelscope            # 模型下载工具（不影响 torch 锁）
```

## 下载 DINOv3 骨干（base）

本项目默认骨干为 `facebook/dinov3-vitb16-pretrain-lvd1689m`（`feature_dim=768`）。
模型是 HF gated，但可从 **ModelScope 镜像**直接下载（无需 HF 许可）：

```bash
python - <<'PY'
import os
os.environ["MODELSCOPE_CACHE"] = r"models"
from modelscope import snapshot_download
snapshot_download("facebook/dinov3-vitb16-pretrain-lvd1689m",
                  local_dir=r"models/dinov3-vitb16-pretrain-lvd1689m",
                  allow_patterns=["*.safetensors", "*.json"])
PY
```

> 若改用其他尺寸，请同步修改 `videopref/config.py` 中的 `DEFAULT_BACKBONE_ID` / `DEFAULT_FEATURE_DIM`，并保证 Checkpoint 中 `feature_dim` 与之匹配。

## 操作指南（预处理 → 训练 → 推理）

### 阶段一：预处理（拆帧 + 标注）

**目的**：把视频列表转成帧序列写入 `frames/{视频名}/`，并逐视频标注喜欢/不喜欢。

**推荐：Gradio 标注工作流**（高效）：

1. `python app.py` → 「🏷️ 标注」Tab。
2. 在"视频路径列表"文本框粘贴**每行一个视频路径**（100~200 个都没问题）。
3. 抽帧参数保持默认（uniform 时长自适应，0.5 帧/秒，min 6 / max 64）。
4. 点 **「一键拆帧并开始标注」** → 逐视频展示帧图 → 点 **👍 喜欢 / 👎 不喜欢 / 跳过**。
5. 中途可关闭，下次点 **「继续上次标注」** 续标（进度存于 `data/label_progress.json`）。
6. 全部标完点 **「导出 labels.json」**（label: 1=喜欢, 0=不喜欢）。

**CLI 拆帧**（单视频或文件夹，不标注）：

```bash
# 单个视频
python -c "from pathlib import Path; from videopref.frames import extract_from_input; from videopref import config; print(extract_from_input(Path('path/to/video.mp4'), Path(config.FRAMES_ROOT)))"

# 整个视频文件夹（会为其中每个视频各建一个 frames 子目录）
python -c "from pathlib import Path; from videopref.frames import extract_from_input; from videopref import config; print(extract_from_input(Path('path/to/videos_dir'), Path(config.FRAMES_ROOT)))"
```

**人工清洗（可选）**：打开 `frames/{视频名}/`，在文件管理器中**直接删除**低质/无关帧（过曝、花屏、遮挡）。保留的帧即训练输入；训练时特征缓存带**帧签名校验**，清洗后自动重新提取，不会用到过期特征。

> 抽帧默认：`uniform` 时长自适应（`n = clip(时长×0.5, 6, 64)`）、纯黑 `<10`、纯白 `>245`；可在 Gradio「抽帧参数」或 `videopref/config.py` 调整。
> **`keyframe` 模式（推荐批量/大规模）**：`-skip_frame nokey` 只解码关键帧(I 帧)，比全量解码**快 ~10 倍、CPU 骤降**，代价是帧间隔不完全均匀（够"整体观感"分类用）；关键帧过少时自动回退均匀采样。
> 抽帧输出宽度上限默认 640（`EXTRACT_MAX_WIDTH`，对 224 特征无损，省 CPU/磁盘）。

**同名视频自动去重**：批量拆帧时，若不同目录下的视频同名（如 `A/movie.mp4` 与 `B/movie.mp4`），后一个会自动在目录名加路径短哈希（`movie` 与 `movie_abc11a06`），避免帧目录冲突。映射关系写入 `frames/_manifest.json`（`video_path → 帧目录名`），训练/推理解析帧目录时读取它，保证指向正确；同一视频重复拆帧幂等复用原目录。

## 抽帧策略与性能优化

三种抽帧模式（`--sampling` / Gradio「抽帧参数」）：

| 模式 | 逻辑 | 速度 |
|---|---|---|
| `uniform`（默认） | 时长自适应：`n = clip(round(时长×fps_target), min, max)`，用 `fps` 滤镜按时间均匀抽样 | 中等（全量解码） |
| **`keyframe`**（推荐批量） | `-skip_frame nokey` **只解码 I 帧**，跳过所有 P/B 帧 | **快 ~10 倍、CPU 骤降** |
| `scene` | ffmpeg 场景变化检测（`gt(scene,阈值)`），按内容突变抽帧 | 中等 |

**关键帧模式细节**：
- 先抽出全部 I 帧；若超过 `max_frames` 则**按序号均匀抽 `max_frames` 帧**（铺满全程，不再只取开头）；若不足 `min_frames` 则回退 `uniform` 均匀时间采样（保证帧数够）。
- 关键帧密度由视频编码器的 GOP 决定：典型 H.264/H.265 视频关键帧较密 → 很快；极稀疏关键帧的视频会触发回退（变慢，属保质量的兜底）。

**其它性能/一致性优化**：
- **流水线预取（prefetch）**：特征提取时，CPU 预处理（解码/缩放/归一化）在后台线程进行，与 GPU 前向重叠（有界队列背压），避免 GPU 空等 CPU。实测约 **2x 提速**。单视频帧数 > batch_size 时效果最明显，故批量推理默认 `--batch-size 16`。
- **输出分辨率上限** `EXTRACT_MAX_WIDTH=640`（DINOv3 只需 224，对分类无损，显著降低 JPEG 编码 CPU 与磁盘）。
- **并行抽帧** `--workers`（默认 4，ffmpeg 子进程并行）。
- **均匀封顶**：超过 `max_frames` 时按序号均匀抽取（`_uniform_sample_list`），避免只取视频开头。
- **递归扫描**：输入文件夹时递归扫描所有子目录（`recursive=True`，Gradio 可开关）。
- 自动剔除纯黑（灰度均值<10）/纯白（>245）帧。
- 兼容新旧 ffmpeg：`-vsync` 与 `-fps_mode` 自动适配。

> 建议批量大规模用 `--sampling keyframe --min-frames 4 --max-frames 32`，又快又省。

### 阶段二：训练

**目的**：用清洗后的帧 + 标注训练池化层与分类头（骨干冻结），产出 Checkpoint。

```bash
python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42
```

- `labels.json` 形如 `[{"video_path": "D:/videos/a.mp4", "label": 1}, ...]`；`video_path` 的 **stem（不含扩展名）必须与 `frames/` 下的子目录名一致**，否则训练会跳过该视频。
- 首次运行会提取逐帧特征并缓存到 `features_cache/`；之后复用缓存，不重复计算。
- 可选：`--augment`（训练期数据增强）、`--val-fraction`、`--batch-size`、`--log-dir`（tensorboard）、`--wandb`、`--threads`（限制 torch CPU 线程数，默认 8，降低 CPU 占用）。
- 输出：`checkpoints/model.ckpt`（最佳）与 `checkpoints/final_epoch{epochs}.ckpt`。

### 阶段三：推理

**目的**：对某个清洗后的帧目录输出喜好概率 `like_probability`。

**Gradio 方式（推荐）**：`python app.py` → 「🔍 推理」Tab → 在下拉框选一个 `frames/` 子目录和 Checkpoint → 开始推理。输出 `like_probability` 与 JSON 结构化结果（含 num_frames、config、training_stats）。

**CLI/脚本方式**：

```bash
python - <<'PY'
from pathlib import Path
from videopref.inference import infer_frames
r = infer_frames(Path("frames/like000"), "checkpoints/model.ckpt")
print(r["like_probability"])
PY
```

> 推理端无状态：每次调用实时加载骨干 + Checkpoint、提取帧特征并预测；超参数全部来自 Checkpoint，不硬编码。

### 批量推理（成千上万个视频）

对成百上千个视频批量抽帧 + 预测，**骨干与 Checkpoint 只加载一次**，并复用特征缓存：

```bash
python infer_batch.py --videos data/video_list.txt --checkpoint checkpoints/model.ckpt \
    --output predictions.csv --sampling keyframe --min-frames 4 --max-frames 32 --workers 4
```

- `--videos`：`.txt`（每行一个视频路径）或视频文件夹（**递归扫描所有子目录**）；缺帧视频自动补抽（默认 `keyframe`）。
- 输出 `predictions.csv`（utf-8-sig，Excel 直接打开），**三列：文件名、喜好概率、文件全路径**（失败/无帧的视频概率留空）。
- **进度条**：加载骨干 / 抽帧 / 推理三个阶段均有 tqdm 进度。
- **容错**：单个视频损坏/失败只跳过并记录（`[warn] 拆帧失败`），不中断整批；结束打印失败数量与前 10 个原因。
- 特征缓存在 `features_cache/`（带帧签名校验），重复运行不重复提取。
- 其他参数：`--min-frames`（默认 4）、`--max-frames`（默认 32）、`--workers`、`--max-width`、`--batch-size`（默认 16，配合预取）、`--threads`（torch CPU 线程上限，默认 8，降低 CPU 占用）、`--limit`（只测前 N 个）。

---

## 快速上手（合成数据端到端）

```bash
# 1) 生成 24 条合成视频 + 标注
python scripts/make_synthetic_data.py --n-like 12 --n-dislike 12 \
    --videos data/synthetic_videos --labels data/labels.json

# 2) 拆帧（等价于 Gradio「🎬 拆帧」Tab）
python -c "from pathlib import Path; from videopref.frames import extract_from_input; from videopref import config; print(extract_from_input(Path('data/synthetic_videos'), Path(config.FRAMES_ROOT)))"

# 3) 训练
python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42

# 4) Gradio 推理/拆帧
python app.py
```

## 命令行训练

```
python train.py --data labels.json --cache-dir ./features_cache \
    --output-dir ./checkpoints --epochs 100 --lr 1e-3 --seed 42
```

可选参数：`--batch-size`、`--val-fraction`、`--augment`（训练期数据增强）、
`--log-dir`（tensorboard）、`--wandb`、`--device`、`--backbone-dir`、`--threads`（CPU 线程上限，降占用）。

## 标注格式

```json
[
  {"video_path": "path/to/video1.mp4", "label": 1},
  {"video_path": "path/to/video2.mp4", "label": 0}
]
```

训练前请先在文件系统中核查并删除低质帧；`label` 取值 `0`(dislike) / `1`(like)。

## Checkpoint 规范

Gradio 推理端所有超参数从 Checkpoint 读取，禁止硬编码：

```json
{
  "model_state": "...",
  "config": {"backbone_id": "facebook/dinov3-vitb16-pretrain-lvd1689m", "feature_dim": 768, "max_frames": 64},
  "label_mapping": {"like": 1, "dislike": 0},
  "training_stats": {"epoch": 50, "val_auc": 0.87}
}
```

## 工程约束

- 除 DINOv3 骨干外，池化层 + 分类头参数量极小（约 1.5k），便于快速保存/加载。
- 全程 `pathlib.Path`，禁止字符串拼接路径。
- Gradio 推理端无会话状态、不缓存特征、不存储中间结果。
- 训练期数据增强不影响缓存的特征向量。

## 工程健壮性（历次修复）

- **中文路径/元数据不崩溃**：ffmpeg 子进程统一用 `utf-8 + errors="replace"` 解码输出，规避 Windows GBK `UnicodeDecodeError`；错误消息对 `res.stderr` 判空。
- **坏视频不中断批次**：`extract_from_input` 单视频失败只 `[warn]` 跳过并继续；`infer_batch` 逐视频容错，结束时汇报失败数量与原因。
- **特征缓存失效校验**：缓存附带帧签名（文件名+大小+mtime），人工清洗/重拆帧后自动重提取，避免用过期特征。
- **同名去重 + manifest**：见上文「同名视频自动去重」。
- **Checkpoint 加载安全化**：`torch.load(..., weights_only=True)`。
- **ffmpeg 版本兼容**：`-vsync`/`-fps_mode` 自动适配。
- **最佳 Checkpoint**：按 `(val_auc, -val_loss)` 联合择优，auc 平局时选 loss 更低者。

## 变更记录

- **v0.1 初版**：Spec v2.0 完整实现——DINOv3 骨干 + Masked Attention Pooling + MLP 头；拆帧（uniform/scene）/特征提取/训练/Gradio 双 Tab/Checkpoint 规范。
- **抽帧优化**：时长自适应均匀抽样；`keyframe` 模式（只解 I 帧，快 ~10 倍）；输出分辨率上限 640；并行 `workers`；均匀封顶到 `max_frames`。
- **规模扩展**：`infer_batch.py` 批量推理（一次加载骨干/模型、特征缓存、进度条、CSV 三列输出、坏视频容错、`--min/max-frames`）。
- **健壮性**：GBK 解码修复、`res.stderr` 判空、torch `weights_only=True`、`--threads` CPU 限制、重拆帧清空旧帧、空目录防御。
- **模块整理**：模型类收拢到 `model.py`；命名契约（`FramesNamer` + manifest）统一在 `paths.py`；常量去重；递归文件夹扫描。
- **标注工作流**：标注 Tab 支持「一键拆帧并标注 / 标注 frames/ 全部已拆帧 / 续标」；导出绝对路径。
