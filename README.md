# visual-pref — 个人媒体喜好二分类器

轻量、本地的**个人喜好二分类器**，同时支持**视频 🎬 与图片 🖼**。采用"人机协同"工作流：自动拆帧/摄入 → 用户在文件系统中人工清洗 → 基于清洗后的帧序列进行特征提取与喜好预测，从海量视频和图片中挑出你喜欢的。

**核心架构**：冻结 **DINOv3 base** 骨干（768 维）+ **Masked Attention Pooling** + 轻量 **MLP 分类头**（仅约 1.5k 可训练参数），面向极少标注样本下的快速收敛与本地化部署。

> **视频与图片共用同一模型与工作流**：模型输入的永远是"逐帧特征"。视频是多帧序列，图片则是 T=1 的单帧序列（`Masked Attention Pooling` 对 T=1 自然退化为单帧聚合）。因此一个 Checkpoint 即可同时给挑视频和挑图片用，无需分开训练。

## 特性

- **全流程 GUI 化**：Gradio 六 Tab——拆帧/摄入、标注、推理、批量推理、工具、训练；训练与批量推理在后台线程运行 + 实时曲线/HTML 进度条，不阻塞界面。
- **图片为一等公民**：图片以 `frames/image/` 下**单文件**独立存放（保留原扩展名、零重编码），与视频分区，元数据带 `kind` 类型。
- **CLI 保留**：`train.py`、`infer_batch.py`、`random_pick_videos.py`、`move_low_score_files.py` 命令行能力完整，可脚本化/批处理。
- **三种抽帧模式**：`uniform`（时长自适应均匀，默认）、`keyframe`（只解 I 帧，快 ~10 倍）、`scene`（场景检测）；图片摄入不走 ffmpeg。
- **特征缓存 + 帧签名失效校验**：清洗/重摄入后自动重提取，训练与推理严格一致、不重复计算。
- **坏文件容错**：单个视频/图片失败跳过不中断批次。

## 工作流

```
拆帧/摄入 ─▶ 人工清洗(可选) ─▶ 标注 ─▶ 训练 ─▶ 推理
```

```
媒体(视频文件/图片文件/文件夹/路径列表) ─▶ 🎬 拆帧 Tab ─┬─ 视频 ─▶ frames/video/{名}/0001.jpg ...
                                                          └─ 图片 ─▶ frames/image/{名}.{扩展名}  (单文件, 原样保留)
                                        │ 人工清洗：删除低质帧 / 替换图片 (可选)
                                        ▼
                         🏷️ 标注 Tab：逐项看图点 喜欢/不喜欢 (视频多帧 🎬 / 图片单帧 🖼)
                                        ▼
                                       导出 labels.json (带 kind)
                                        ▼
                    训练(🎓 训练 Tab 或 CLI) ─▶ Checkpoint ─▶ 🔍 推理 / ⚡ 批量推理
```

各阶段以 **`frames/` 目录结构 + `frames/_manifest.json`** 作为唯一数据契约解耦。

## 数据契约

```
frames/
  _manifest.json            # 媒体路径 → 条目元数据 {dir, kind, ext}
  video/                    # 视频工作区：一个视频一个目录
    <sanitized名>/0001.jpg...      (同名加路径短哈希去重)
  image/                    # 图片工作区：一个图片一个文件
    <sanitized名>.<原扩展名>        (字节级原样保留)
```

**`frames/_manifest.json`**（结构化元数据，提升信息密度）：

```json
{
  "F:/.../a.mp4": {"dir": "video/a",     "kind": "video", "ext": ".mp4"},
  "C:/.../b.png": {"dir": "image/b.png", "kind": "image", "ext": ".png"}
}
```

- `dir`：条目相对 `frames/` 根的子路径（视频=目录，图片=文件）。
- `kind`：`"video"` | `"image"`。`ext`：源文件扩展名。
- `video_path` 解析（`frames_dir_for_video`）优先查 manifest，命中返回条目路径（视频→目录、图片→文件）；未命中按扩展名回退到对应子区。

**`data/labels.json`**：

```json
[{"video_path": "D:/videos/a.mp4", "label": 1, "kind": "video"},
 {"video_path": "C:/pics/b.png",   "label": 0, "kind": "image"}]
```

`video_path` 存绝对路径，`kind` 记录类型（训练/推理按 manifest 解析，不硬编码）。

**其他产物**：`data/label_progress.json`（标注续标进度）、`features_cache/`（按条目子路径 key 分区缓存：`video/foo.pt`、`image/foo.pt`）、`checkpoints/model.ckpt`（含 config/label_mapping/training_stats）。

## 目录结构

```
.
├── visualpref/                   # 主包
│   ├── config.py                 # 集中常量（路径/抽帧/EXTRACT_*/VIDEO+IMAGE_EXTENSIONS/子区名）
│   ├── paths.py                  # 纯路径工具：sanitize、短哈希、帧/媒体枚举、is_image/is_media
│   ├── manifest.py               # 媒体→条目元数据契约：FramesNamer + frames_dir_for_video
│   ├── ffmpeg.py                 # ffmpeg 命令构建与执行（含新旧版本适配）
│   ├── sampling.py               # 抽帧策略：uniform/keyframe/scene + 黑白过滤
│   ├── frames.py                 # 摄入编排：extract_frames / extract_from_input / ingest_image
│   ├── backbone.py               # 冻结 DINOv3 骨干加载（ModelScope 自动下载）
│   ├── features.py               # 逐帧 [CLS] 特征提取；frames_dir_to_paths（目录/文件统一）
│   ├── pipeline.py               # prefetch_map：CPU 预处理与 GPU 前向重叠
│   ├── model.py                  # 模型类 + Checkpoint 读写（weights_only=True）
│   ├── augment.py                # 训练期数据增强
│   ├── dataset.py                # 训练数据 + 特征缓存（帧签名失效校验）+ collate
│   ├── labeling.py               # 标注队列构建 + 进度持久化 + 会话状态机
│   ├── predictor.py              # Predictor：骨干+模型一次性加载（推理/批量推理复用）
│   ├── batch_infer.py            # 批量推理（视频/图片）
│   ├── train.py                  # 训练（CLI + 可回调，供 Gradio 复用）
│   ├── inference.py              # 单条目推理
│   ├── gradio_app.py             # Gradio 六 Tab 组合入口（薄层）
│   └── ui/                       # Gradio 各 Tab 实现（按职责一模块一 Tab）
│       ├── common.py             # 共享：checkpoint/条目扫描、File 解析、进度条、抽帧参数
│       ├── extract_tab.py        # 🎬 拆帧/摄入 + 清空
│       ├── label_tab.py          # 🏷️ 标注
│       ├── infer_tab.py          # 🔍 单条推理
│       ├── batch_tab.py          # ⚡ 批量推理
│       ├── tools_tab.py          # 🧰 工具
│       └── train_tab.py          # 🎓 训练
├── train.py                      # CLI 训练入口
├── infer_batch.py                # CLI 批量推理入口
├── app.py                        # Gradio 入口
├── random_pick_videos.py         # CLI：随机选取媒体（工具 Tab 复用）
├── move_low_score_files.py       # CLI：按 CSV 移动低分文件（工具 Tab 复用）
├── scripts/
│   ├── make_synthetic_data.py    # 合成测试视频数据（暖色=喜欢 / 冷色=不喜欢）
│   └── smoke_backbone.py         # 骨干冒烟测试
├── models/                       # 下载的 DINOv3 权重
├── frames/                       # 摄入输出 + 人工清洗工作区（video/ 与 image/ 分区）
├── checkpoints/                  # 训练产出
└── features_cache/               # 训练特征缓存
```

## 安装

### 前置条件

- **Python 3.11**（`uv` 按仓库内 `.python-version` 自动选择）。
- **ffmpeg / ffprobe**：拆帧/抽帧强依赖，二者须加入系统 PATH。Windows 可从 [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) 下载 `ffmpeg-release-full`，把 `bin` 加入 PATH 后重开终端，验证：
  ```bash
  ffmpeg -version && ffprobe -version
  ```
- **GPU（可选）**：NVIDIA + CUDA 时自动走 GPU（torch 走 cu128 源）；无 GPU 回退 CPU（较慢）。
- **网络**：Windows 直连 HF 可能遇 SSL/GBK 问题，本项目用 ModelScope 下载骨干（见下）。

### 安装依赖

```bash
uv sync          # 安装 torch(cu128)/transformers/gradio/modelscope/sklearn 等
```

## DINOv3 骨干（自动下载）

默认骨干 `facebook/dinov3-vitb16-pretrain-lvd1689m`（`feature_dim=768`）。HF 上为 gated，本项目改用 **ModelScope** 源（无需 HF 许可），且**首次运行时自动下载**：

- `models/dinov3-vitb16-pretrain-lvd1689m/` 下缺 `config.json` 或 `*.safetensors` 时，`load_backbone`（训练/推理/批量推理首步）自动从 ModelScope 下载到该目录。
- 下载与 modelscope 缓存均落在项目内（`models/` 与 `.modelscope/`），**不写用户 HOME、不依赖 HF**。
- 运行时加载始终 `local_files_only=True`。

> 改尺寸时同步改 `visualpref/config.py` 的 `DEFAULT_BACKBONE_ID` / `DEFAULT_FEATURE_DIM`，并保证 Checkpoint 中 `feature_dim` 匹配。

## 快速上手（Gradio）

```bash
uv run python app.py          # 默认 127.0.0.1:7860
```

`app.py` 用 argparse 启动：`--server-name` / `--server-port` 显式解析，其余任意 `--key value`
都会**原样透传**为 `**kwargs` 交给 `demo.launch`，因此 Gradio `Blocks.launch` 支持的所有
参数都可用，无需改代码，例如：

```bash
uv run python app.py --server-port 7861 --share          # 自定义端口 + 公开分享
uv run python app.py --auth "me:secret" --max-file-size "50mb"
uv run python app.py --inbrowser                         # 启动后自动打开浏览器
```

浏览器打开 `http://127.0.0.1:7860`：

| Tab | 作用 |
|---|---|
| 🎬 **拆帧** | 上传/文件夹/路径列表（视频+图片）→ 视频拆帧到 `frames/video/`、图片摄入到 `frames/image/` |
| 🏷️ **标注** | 标注已摄入条目（视频多帧 🎬 / 图片单帧 🖼），逐项 👍/👎/跳过/上一步，可续标、导出 `labels.json` |
| 🔍 **推理** | 选一个条目 + Checkpoint → `like_probability` + 结构化 JSON |
| ⚡ **批量推理** | 媒体路径列表 + Checkpoint → 批量摄入/抽帧/预测 → 结果表 + CSV |
| 🧰 **工具** | 随机选取媒体 / 按 CSV 移动低分文件 |
| 🎓 **训练** | 后台线程训练，实时曲线 + 日志，训练期间可切其他 Tab |

## 操作指南

### 阶段一：摄入 + 清洗 + 标注

1. **拆帧/摄入**（「🎬 拆帧」Tab）：提供媒体文件、文件夹或每行一个的路径列表。视频按所选抽帧模式拆帧到 `frames/video/{名}/`；**图片原样摄入**为 `frames/image/{名}.{原扩展名}`（默认不缩放、零重编码）。
2. **人工清洗（可选）**：打开 `frames/video/{名}/` 删除低质帧；图片可替换 `frames/image/{名}.{扩展名}`。特征缓存带**帧签名校验**，清洗后自动重提取。
3. **标注**（「🏷️ 标注」Tab）：点「标注 frames/ 全部已摄入媒体」或「继续上次标注」（进度存 `data/label_progress.json`）。逐项看图点 👍/👎/跳过/上一步；预览「每行预览数/高度」可调。完毕点「导出 labels.json」（label: 1=喜欢, 0=不喜欢，含 `kind`）。

**CLI 摄入/拆帧**（单文件或文件夹）：

```bash
uv run python -c "from pathlib import Path; from visualpref.frames import extract_from_input; from visualpref import config; print(extract_from_input(Path('path/to/inputs'), Path(config.FRAMES_ROOT)))"
```

**同名自动去重**：不同目录下同名视频 → `video/foo` 与 `video/foo_<hash>`；同名图片 → `image/foo.png` 与 `image/foo_<hash>.png`。映射写入 `_manifest.json`，训练/推理据此定位；同一媒体重复摄入幂等复用原条目。

### 阶段二：训练

**Gradio（推荐）**：「🎓 训练」Tab → 填 `labels.json`、缓存/输出目录、epochs/lr/seed → 开始训练。后台线程执行，实时看 loss/auc 曲线与日志。

**CLI**：

```bash
uv run python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42
```

- `labels.json` 可同时含视频与图片条目（`kind` 记录类型）；条目经 manifest 解析到 `frames/video|image/`。
- 首次运行提取逐帧特征并缓存（视频按帧、图片单帧），之后复用缓存。
- 可选：`--augment`（训练期增强）、`--val-fraction`、`--batch-size`、`--threads`、`--log-dir`、`--wandb`、`--device`、`--backbone-dir`、`--backbone-id`。
- 输出：`checkpoints/model.ckpt`（按 `(val_auc, -val_loss)` 择优）与 `checkpoints/final_epoch{epochs}.ckpt`。

### 阶段三：推理

**单条目**：「🔍 推理」Tab 选条目 + Checkpoint → `like_probability`；或脚本：

```bash
uv run python -c "from pathlib import Path; from visualpref.inference import infer_frames; r = infer_frames(Path('frames/video/like000'), 'checkpoints/model.ckpt'); print(r['like_probability'])"
```

**批量**：「⚡ 批量推理」Tab 粘贴路径列表 + Checkpoint；或 CLI：

```bash
uv run python infer_batch.py --videos data/video_list.txt --checkpoint checkpoints/model.ckpt \
    --output predictions.csv --sampling keyframe --min-frames 4 --max-frames 32 --workers 4
```

- `--videos`：`.txt`（每行一个媒体路径）或文件夹（递归扫描，视频+图片）；缺帧/未摄入的条目自动补抽/摄入。
- 输出 `predictions.csv`（utf-8-sig）：**文件名、喜好概率、文件全路径**（失败/无帧概率留空）。
- 骨干与 Checkpoint 只加载一次；特征缓存带签名校验，重跑不重复提取；坏文件跳过不中断整批。

### 工具（🧰 工具 Tab / 独立 CLI）

**随机选取媒体**（视频+图片）：

```bash
uv run python random_pick_videos.py <src_dir> [-n COUNT] [--seed S] [--no-recursive] [-o FILE]
```

**按 CSV 移动低分文件**（三列：`文件名,喜好分数,文件全路径`，低于阈值移动）：

```bash
uv run python move_low_score_files.py <csv> [-d 目标文件夹] [-t 阈值] [--dry-run]
```

## 抽帧策略

| 模式 | 逻辑 | 速度 |
|---|---|---|
| `uniform`（默认） | 时长自适应：`n = clip(round(时长×fps_target), min, max)`，`fps` 滤镜按时间均匀抽样 | 中等（全量解码） |
| `keyframe`（推荐批量） | `-skip_frame nokey` 只解 I 帧 | **快 ~10 倍、CPU 骤降** |
| `scene` | ffmpeg 场景检测（`gt(scene,阈值)`） | 中等 |

其他：视频输出宽度上限默认 640、自动剔除纯黑(<10)/纯白(>245)帧、并行处理 `--workers`、均匀封顶到 `max_frames`、递归扫描子目录、ffmpeg `-vsync`/`-fps_mode` 自动适配。**图片摄入不走 ffmpeg**：默认原样复制、保留原始分辨率与扩展名（`IMAGE_MAX_WIDTH=0`），宽度上限仅对视频生效，保证图片零质量损失。

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
# 1) 生成 24 条合成视频 + 标注（含 kind）
uv run python scripts/make_synthetic_data.py --n-like 12 --n-dislike 12 \
    --videos data/synthetic_videos --labels data/labels.json

# 2) 拆帧（视频 -> frames/video/）
uv run python -c "from pathlib import Path; from visualpref.frames import extract_from_input; from visualpref import config; print(extract_from_input(Path('data/synthetic_videos'), Path(config.FRAMES_ROOT)))"

# 3) 训练
uv run python train.py --data data/labels.json --cache-dir features_cache \
    --output-dir checkpoints --epochs 100 --lr 1e-3 --seed 42

# 4) Gradio
uv run python app.py
```

## 工程健壮性

- **中文路径不崩溃**：ffmpeg 子进程统一 `utf-8 + errors="replace"` 解码，规避 Windows GBK `UnicodeDecodeError`；`res.stderr` 判空。
- **坏文件不中断批次**：单文件失败 `[warn]` 跳过并继续，结束汇报失败数量与原因。
- **特征缓存失效校验**：缓存带帧签名（文件名+大小+mtime），清洗/重摄入后自动重提取。
- **同名去重 + 结构化 manifest**：见上文「同名自动去重」。
- **缓存 key 分区**：`features_cache/` 按 `video/...` / `image/...` 分区，视频与图片同名条目互不覆盖。
- **Checkpoint 加载安全化**：`torch.load(..., weights_only=True)`。
- **最佳 Checkpoint**：按 `(val_auc, -val_loss)` 联合择优。

## 变更记录

- **v0.2.0**：更名为 `visual-pref` / `visualpref`；`app.py` 支持 argparse 透传任意 `--key value` 到 `demo.launch`。
- **v0.1 初版**：DINOv3 骨干 + Masked Attention Pooling + MLP 头；拆帧（uniform/scene）/特征提取/训练/Gradio/Checkpoint 规范。
- **规模扩展**：批量推理（一次加载骨干/模型、特征缓存、进度条、CSV 三列、坏文件容错）。
- **重构**：抽出 `Predictor` 公共层；`frames.py` 拆出 `ffmpeg.py`/`sampling.py`；`paths.py` 拆出 `manifest.py`；`features.py` 抽 `pipeline.prefetch_map`；标注状态机下沉 `labeling.py`。
- **GUI 化**：Gradio 六 Tab + 后台线程训练/实时曲线。
- **媒体统一 + 图片一等公民**：新增图片摄入（`ingest_image`），`frames/` 按 `video/` 与 `image/` 分区；图片以单文件存放、保留原始质量；manifest 结构化（`dir`/`kind`/`ext`），labels.json 带 `kind`；批量推理/随机选取等支持视频+图片。
- **UI 模块化**：`gradio_app.py` 拆分为 `visualpref/ui/` 包（common + 六个 Tab 模块），入口瘦身为组合层；标注标题与推理下拉显示 🎬/🖼 类型图标。
