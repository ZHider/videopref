"""集中式配置与数据契约常量。

所有默认路径基于 ``PROJECT_ROOT``（仓库根目录），全程使用 ``pathlib.Path``。
Gradio 推理端一律从 Checkpoint 读取超参数，禁止硬编码。
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 工程根目录与数据契约目录
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# frames/{sanitized_video_name}/0001.jpg ...  —— 拆帧输出 + 人工清洗工作区。
# frames/ 下按媒体类型分区：
#   frames/video/{key}/0001.jpg...  —— 视频拆帧工作区（一个视频一个目录）
#   frames/image/{key}.{ext}        —— 图片工作区（一个图片一个文件，原样保留）
FRAMES_ROOT: Path = PROJECT_ROOT / "frames"
FRAMES_VIDEO_SUBDIR: str = "video"
FRAMES_IMAGE_SUBDIR: str = "image"
FRAMES_VIDEO_ROOT: Path = FRAMES_ROOT / FRAMES_VIDEO_SUBDIR
FRAMES_IMAGE_ROOT: Path = FRAMES_ROOT / FRAMES_IMAGE_SUBDIR

# 下载后的 DINOv3 骨干权重目录
MODELS_DIR: Path = PROJECT_ROOT / "models"

# 训练产出 Checkpoint 目录
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"

# 训练特征缓存目录（.pt / .npy）
FEATURES_CACHE_DIR: Path = PROJECT_ROOT / "features_cache"

# 标注数据默认目录
DATA_DIR: Path = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# DINOv3 骨干（base, feature_dim=768）
# ---------------------------------------------------------------------------
DEFAULT_BACKBONE_ID: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_BACKBONE_DIR: Path = MODELS_DIR / "dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_FEATURE_DIM: int = 768
DEFAULT_IMAGE_SIZE: int = 224

# ---------------------------------------------------------------------------
# 视频聚合 / 训练默认超参数
# ---------------------------------------------------------------------------
DEFAULT_MAX_FRAMES: int = 64

# ---------------------------------------------------------------------------
# 拆帧策略：时长自适应均匀抽样（默认） / 场景检测（可选）
# ---------------------------------------------------------------------------
DEFAULT_SAMPLING: str = "uniform"   # "uniform" | "scene" | "keyframe"(只解I帧,快但粗糙)
FPS_TARGET: float = 0.5             # 目标抽帧密度（帧/秒），短视频少、长视频多
MIN_FRAMES: int = 6                 # 单视频最少帧数（保证池化与人工判断）
DEFAULT_SCENE_THRESHOLD: float = 0.3   # ffmpeg scene 检测阈值
BLACK_FRAME_MEAN: int = 10             # 灰度均值低于此值视为纯黑帧
WHITE_FRAME_MEAN: int = 245            # 灰度均值高于此值视为纯白帧
FRAME_EXT: str = ".jpg"
FRAME_FILENAME_WIDTH: int = 4          # 零填充宽度，如 0001.jpg

# 支持的视频扩展名（文件夹扫描时用于过滤）
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}
)

# 支持的图片扩展名。图片按"单帧视频"处理，全流程复用帧契约
# （frames/{name}/0001.jpg -> 特征 -> 池化 -> 分类）。
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)

# 媒体（视频 + 图片）扩展名合集，供扫描/分流使用
MEDIA_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

# 模型输入规格：DINOv3 输入为正方形 224×224（preprocessor_config: size={224,224}，
# resample=bicubic）。抽帧/图片摄入统一 center-crop 成该正方形，喂模型时
# processor 的 resize 退化为恒等（跳过昂贵缩放，CPU 大头）。
MODEL_INPUT_SIZE: int = DEFAULT_IMAGE_SIZE

# 抽帧/摄入输出规格（正方形边长）。默认与模型输入一致（224×224 center-crop）；
# 设为 0 则不做缩放（保留原始分辨率，仅在需要时）。
EXTRACT_MAX_WIDTH: int = MODEL_INPUT_SIZE
EXTRACT_WORKERS: int = 4           # 批量抽帧并行视频数（利用多核，缩短整批耗时）
EXTRACT_HWACCEL: str | None = None # 硬件解码："cuda" 等；长/高分辨率视频可降 CPU，小视频不划算

# 图片摄入输出规格：默认与模型输入一致（224×224 center-crop，存 JPEG），
# 与视频帧同一宽度标准，节省磁盘/CPU。设为 0 可回退"原样复制"。
IMAGE_MAX_WIDTH: int = MODEL_INPUT_SIZE

# ---------------------------------------------------------------------------
# 标签契约
# ---------------------------------------------------------------------------
LABEL_MAPPING: dict[str, int] = {"like": 1, "dislike": 0}

# ---------------------------------------------------------------------------
# 目录/文件命名
# ---------------------------------------------------------------------------
CHECKPOINT_FILENAME: str = "model.ckpt"
BACKBONE_CONFIG_NAME: str = "config.json"
