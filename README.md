# videopref — 个人视频喜好二分类器

轻量级、本地的**视频喜好二分类器**。采用"人机协同"工作流：自动拆帧提取关键视觉片段 → 用户在文件系统中人工清洗低质帧 → 基于清洗后的帧序列进行特征提取与喜好预测，从大量视频中挑出你喜欢的。

**核心架构**：冻结 **DINOv3 base** 骨干（768 维）+ **Masked Attention Pooling** + 轻量 **MLP 分类头**（仅约 1.5k 可训练参数），面向极少标注样本下的快速收敛与本地化部署。

## 特性

- **全流程 GUI 化**：Gradio 提供六 Tab——拆帧 / 标注 / 推理 / 批量推理 / 工具 / 训练，训练在后台线程运行、实时曲线，不阻塞界面。
- **CLI 保留**：`train.py`、`infer_batch.py` 等命令行能力完整保留，可脚本化/批处理。
- **三种抽帧模式**：`uniform`（时长自适应均匀，默认）、`keyframe`（只解 I 帧，快 ~10 倍）、`scene`（场景检测）。
- **特征缓存 + 帧签名失效校验**：清洗/重拆帧后自动重提取，训练与推理严格一致、不重复计算。
- **坏视频容错**：单个视频失败跳过不中断批次。

## 工作流

```
拆帧 ─▶ 人工清洗(可选) ─▶ 标注 ─▶ 训练 ─▶ 推理
```

```
视频(文件/文件夹/路径列表) ─▶ 🎬 拆帧 Tab ─▶ frames/{视频名}/0001.jpg ...
                                                    │ 人工清洗：删除低质帧(可选)
                                                    ▼
                              🏷️ 标注 Tab：对 frames/ 下已拆帧目录逐一看图点 喜欢/不喜欢
                                                    ▼
                                          导出 labels.json
                                                    ▼
                        训练(🎓 训练 Tab 或 CLI) ─▶ Checkpoint ─▶ 🔍 推理 / ⚡ 批量推理
```

拆帧与标注/训练/推理解耦，以 **`frames/` 文件夹结构 + `frames/_manifest.json`** 作为各阶段间的唯一数据契约。

## 目录结构

```
.
├── videopref/                # 主包
│   ├── config.py             # 集中常量（路径 / 抽帧 / EXTRACT_* / VIDEO_EXTENSIONS）
│   ├── paths.py              # 纯路径工具：sanitize、短哈希、帧/视频枚举
│   ├── manifest.py           # video_path→帧目录名 契约：FramesNamer + frames_dir_for_video
│   ├── ffmpeg.py             # ffmpeg 命令构建与解码执行（含新旧版本适配）
│   ├── sampling.py           # 抽帧策略：uniform / keyframe / scene + 黑白过滤
│   ├── frames.py             # 拆帧编排：extract_frames / extract_from_input
│   ├── backbone.py           # 冻结 DINOv3 骨干加载
│   ├── features.py           # 逐帧 [CLS] 特征提取
│   ├── pipeline.py           # prefetch_map：CPU 预处理与 GPU 前向重叠
│   ├── model.py              # 模型类 + Checkpoint 读写（weights_only=True）
│   ├── augment.py            # 训练期数据增强
│   ├── dataset.py            # 训练数据 + 特征缓存（帧签名失效校验）+ collate
│   ├── labeling.py           # 标注队列构建 + 进度持久化 + 会话状态机
│   ├── predictor.py          # Predictor：骨干+模型一次性加载（推理/批量推理复用）
│   ├── batch_infer.py        # 批量推理 CLI
│   ├── train.py              # 训练（CLI + 可回调，供 Gradio 复用）
│   ├── inference.py          # 单视频推理
│   └── gradio_app.py         # Gradio 六 Tab
├── train.py                  # CLI 训练入口
├── infer_batch.py            # CLI 批量推理入口
├── app.py                    # Gradio 入口
├── random_pick_videos.py     # CLI：随机选取视频（工具 Tab 复用）
├── move_low_score_files.py   # CLI：按 CSV 移动低分文件（工具 Tab 复用）
├── scripts/
│   ├── make_synthetic_data.py    # 合成测试数据（暖色=喜欢 / 冷色=不喜欢）
│   └── smoke_backbone.py         # 骨干冒烟测试
├── models/                   # 下载的 DINOv3 权重
├── frames/                   # 拆帧输出 + 人工清洗工作区
├── checkpoints/              # 训练产出
└── features_cache/           # 训练特征缓存
```

## 安装

### 前置条件

- **Python 3.11**（`uv` 会按仓库内 `.python-version` 自动选择）。
- **ffmpeg / ffprobe**：拆帧/抽帧强依赖，二者都必须加入系统 PATH，否则报「未找到 ffmpeg」。
  - Windows：从 [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) 下载 `ffmpeg-release-full` 解压，把里面的 `bin` 目录加入 PATH（或复制其中的 `ffmpeg.exe`/`ffprobe.exe` 到 PATH 已有目录），重开终端。
  - 验证：
    ```bash
    ffmpeg -version
    ffprobe -version
    ```
    两个命令都能输出版本号即为就绪。
- **GPU（可选）**：NVIDIA 显卡 + CUDA 可用时自动走 GPU（本项目 torch 走 cu128 源）；无 GPU 会回退 CPU，速度较慢。
- **网络**：Windows 直连 HF 可能遇 SSL/GBK 问题，推荐用 ModelScope 下载骨干（见下节）。

### 安装依赖

```bash
uv sync                              # 安装 torch(cu128)/transformers/gradio/modelscope 等
```

## DINOv3 骨干（自动下载）

默认骨干为 `facebook/dinov3-vitb16-pretrain-lvd1689m`（`feature_dim=768`）。模型在 HF 为 gated，本项目改用 **ModelScope** 源（无需 HF 许可），且**首次运行时自动下载**：

- 当 `models/dinov3-vitb16-pretrain-lvd1689m/` 下缺少 `config.json` 或 `*.safetensors` 时，`load_backbone`（拆帧后的训练/推理/批量推理首步）会自动从 ModelScope 下载权重到该目录。
- 下载与 modelscope 缓存都落在项目文件夹内（`models/` 与 `.modelscope/`），**不会写入用户 HOME 目录**，也不依赖 HuggingFace。
- 运行时加载始终 `local_files_only=True`（从本地加载，不联网 HF）。

> 若改用其他尺寸，请同步修改 `videopref/config.py` 的 `DEFAULT_BACKBONE_ID` / `DEFAULT_FEATURE_DIM`，并保证 Checkpoint 中 `feature_dim` 与之匹配。

## 快速上手（Gradio）

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，即可使用六个 Tab：

| Tab | 作用 |
|---|---|
| 🎬 **拆帧** | 上传单个视频 / 填文件夹路径 / 粘贴路径列表 → 拆帧到 `frames/` |
| 🏷️ **标注** | 标注 `frames/` 下已拆帧目录，逐一看图点 👍/👎，可续标、导出 `labels.json` |
| 🔍 **推理** | 选一个帧目录 + Checkpoint → 输出 `like_probability` 与结构化 JSON |
| ⚡ **批量推理** | 路径列表 + Checkpoint → 批量抽帧/预测 → 结果表 + CSV |
| 🧰 **工具** | 随机选取视频 / 按 CSV 移动低分文件 |
| 🎓 **训练** | 后台线程训练，实时曲线 + 日志，训练期间可切其他 Tab |

## 操作指南

### 阶段一：拆帧 + 清洗 + 标注

1. **拆帧**（「🎬 拆帧」Tab）：提供视频文件、文件夹或每行一个的路径列表，选择抽帧模式与参数，点「开始拆帧」。输出到 `frames/{视频名}/0001.jpg...`。
2. **人工清洗（可选）**：打开 `frames/{视频名}/`，直接删除低质/无关帧（过曝、花屏、遮挡）。保留帧即训练输入；特征缓存带**帧签名校验**，清洗后自动重提取。
3. **标注**（「🏷️ 标注」Tab）：
   - 点「**标注 frames/ 全部已拆帧视频**」载入 `frames/` 下所有已拆帧目录；或点「**继续上次标注**」续标（进度存于 `data/label_progress.json`）。
   - 逐视频看图，点 **👍 喜欢 / 👎 不喜欢 / 跳过 / 上一步**。
   - 预览框的「每行预览数」与「预览高度」可用滑条实时调整。
   - 全部标完点「**导出 labels.json**」（label: 1=喜欢, 0=不喜欢）。

> 标注只针对已拆帧目录，不再负责拆帧；拆帧统一走「🎬 拆帧」Tab 或 `extract_from_input`（见下）。

**CLI 拆帧**（单视频或文件夹）：

```bash
python -c "from pathlib import Path; from videopref.frames import extract_from_input; from videopref import config; print(extract_from_input(Path('path/to/videos_dir'), Path(config.FRAMES_ROOT)))"
```

**同名视频自动去重**：不同目录下同名视频会在目录名加路径短哈希（`movie` / `movie_abc11a06`），映射写入 `frames/_manifest.json`（`video_path → 帧目录名`），训练/推理据此定位帧目录；同一视频重复拆帧幂等复用原目录。

### 阶段二：训练

**目的**：用清洗后的帧 + 标注训练池化层与分类头（骨干冻结），产出 Checkpoint。

**Gradio（推荐）**：「🎓 训练」Tab → 填 `labels.json` 路径、缓存/输出目录、epochs/lr/seed 等 → 点「开始训练」。后台线程执行，训练期间可切换其他 Tab，实时查看 loss/auc 曲线与日志。

**CLI**：

```bash
python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42
```

- `labels.json` 形如 `[{"video_path": "D:/videos/a.mp4", "label": 1}, ...]`；`video_path` 的 stem 与 `frames/` 子目录名对应（经 manifest 解析）。
- 首次运行会提取逐帧特征并缓存到 `features_cache/`，之后复用缓存不重复计算。
- 可选：`--augment`（训练期数据增强）、`--val-fraction`、`--batch-size`、`--threads`（CPU 线程上限，降占用）、`--log-dir`（tensorboard）、`--wandb`、`--device`、`--backbone-dir`。
- 输出：`checkpoints/model.ckpt`（按 `(val_auc, -val_loss)` 择优）与 `checkpoints/final_epoch{epochs}.ckpt`。

### 阶段三：推理

**单视频**：「🔍 推理」Tab 选帧目录 + Checkpoint → `like_probability`；或脚本：

```bash
python - <<'PY'
from pathlib import Path
from videopref.inference import infer_frames
r = infer_frames(Path("frames/like000"), "checkpoints/model.ckpt")
print(r["like_probability"])
PY
```

**批量（大量视频）**：「⚡ 批量推理」Tab 粘贴路径列表 + 选 Checkpoint；或 CLI：

```bash
python infer_batch.py --videos data/video_list.txt --checkpoint checkpoints/model.ckpt \
    --output predictions.csv --sampling keyframe --min-frames 4 --max-frames 32 --workers 4
```

- `--videos`：`.txt`（每行一个路径）或文件夹（递归扫描）；缺帧视频自动补抽。
- 输出 `predictions.csv`（utf-8-sig）：**文件名、喜好概率、文件全路径**（失败/无帧概率留空）。
- 骨干与 Checkpoint 只加载一次；特征缓存带帧签名校验，重跑不重复提取。
- 容错：单视频失败跳过不中断整批，结束汇报失败数量与原因。

### 工具（🧰 工具 Tab / 独立 CLI）

**随机选取视频**：

```bash
python random_pick_videos.py <src_dir> [-n COUNT] [--seed S] [--no-recursive] [-o FILE]
```

**按 CSV 移动低分文件**（三列：`文件名,喜好分数,文件全路径`，低于阈值移动）：

```bash
python move_low_score_files.py <csv> [-d 目标文件夹] [-t 阈值] [--dry-run]
```

## 抽帧策略

| 模式 | 逻辑 | 速度 |
|---|---|---|
| `uniform`（默认） | 时长自适应：`n = clip(round(时长×fps_target), min, max)`，`fps` 滤镜按时间均匀抽样 | 中等（全量解码） |
| `keyframe`（推荐批量） | `-skip_frame nokey` 只解 I 帧 | **快 ~10 倍、CPU 骤降** |
| `scene` | ffmpeg 场景检测（`gt(scene,阈值)`） | 中等 |

其他：输出宽度上限默认 640、自动剔除纯黑(<10)/纯白(>245)帧、并行拆帧 `--workers`、均匀封顶到 `max_frames`、递归扫描子目录、ffmpeg `-vsync`/`-fps_mode` 自动适配。

## Checkpoint 规范

推理端所有超参数从 Checkpoint 读取，禁止硬编码：

```json
{
  "model_state": "...",
  "config": {"backbone_id": "facebook/dinov3-vitb16-pretrain-lvd1689m", "feature_dim": 768, "max_frames": 64},
  "label_mapping": {"like": 1, "dislike": 0},
  "training_stats": {"epoch": 50, "val_auc": 0.87}
}
```

## 快速上手（合成数据端到端）

```bash
# 1) 生成 24 条合成视频 + 标注
python scripts/make_synthetic_data.py --n-like 12 --n-dislike 12 \
    --videos data/synthetic_videos --labels data/labels.json

# 2) 拆帧
python -c "from pathlib import Path; from videopref.frames import extract_from_input; from videopref import config; print(extract_from_input(Path('data/synthetic_videos'), Path(config.FRAMES_ROOT)))"

# 3) 训练
python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42

# 4) Gradio
python app.py
```

## 工程健壮性

- **中文路径不崩溃**：ffmpeg 子进程统一 `utf-8 + errors="replace"` 解码，规避 Windows GBK `UnicodeDecodeError`；`res.stderr` 判空。
- **坏视频不中断批次**：单视频失败 `[warn]` 跳过并继续，结束汇报失败数量与原因。
- **特征缓存失效校验**：缓存带帧签名（文件名+大小+mtime），清洗/重拆帧后自动重提取。
- **同名去重 + manifest**：见上文「同名视频自动去重」。
- **Checkpoint 加载安全化**：`torch.load(..., weights_only=True)`。
- **最佳 Checkpoint**：按 `(val_auc, -val_loss)` 联合择优。

## 变更记录

- **v0.1 初版**：DINOv3 骨干 + Masked Attention Pooling + MLP 头；拆帧（uniform/scene）/特征提取/训练/Gradio/Checkpoint 规范。
- **规模扩展**：批量推理（一次加载骨干/模型、特征缓存、进度条、CSV 三列、坏视频容错）。
- **重构**：抽出 `Predictor` 公共层；`frames.py` 拆为 `ffmpeg.py`/`sampling.py`/编排；`paths.py` 拆出 `manifest.py`；`features.py` 抽 `pipeline.prefetch_map`；标注会话状态机下沉 `labeling.py`；清理死代码。
- **GUI 化**：Gradio 扩展为六 Tab（拆帧/标注/推理/批量推理/工具/训练）；训练后台线程 + 实时曲线；标注预览高度/列数可调；工具 Tab 适配 `random_pick_videos` 与 `move_low_score_files`。
