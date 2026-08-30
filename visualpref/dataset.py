"""训练数据构建 + 特征缓存。

- 从 ``labels.json``（[{video_path, label}]）解析标注。
- 按 ``frames/{sanitized_stem}`` 定位清洗后的帧目录。
- 用冻结骨干提取每帧 ``[CLS]`` 特征并持久化缓存（.pt），避免训练时重复计算。
- 缓存存储冻结骨干的逐帧特征（不可训练固定值）；池化层/分类头在训练中实时
  以这些特征为输入计算梯度，因此训练无需重跑骨干。
- 数据增强（``--augment``）仅在训练时对帧图像实时应用，不影响缓存特征。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config
from .augment import build_augment_transform, processor_norm
from .features import extract_frame_features, frames_dir_to_paths
from .manifest import frames_dir_for_video
from .paths import feature_cache_path, video_key_of


# ---------------------------------------------------------------------------
# 标注解析
# ---------------------------------------------------------------------------
def load_labels(labels_path: Path | str) -> list[dict]:
    """读取 labels.json：``[{"video_path": "...", "label": 0/1}]``。"""
    labels_path = Path(labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("labels.json 顶层应为数组。")
    for item in data:
        if "video_path" not in item or "label" not in item:
            raise ValueError("每条标注必须包含 video_path 与 label。")
        item["label"] = int(item["label"])
    return data


def resolve_frames_dir(video_path: str | Path) -> Path:
    """将标注中的 video_path 解析为清洗后的帧目录。"""
    return frames_dir_for_video(Path(video_path))


# ---------------------------------------------------------------------------
# 帧特征缓存
# ---------------------------------------------------------------------------
def ensure_video_features(
    frames_dir: Path,
    cache_dir: Path,
    backbone,
    processor,
    device,
    batch_size: int = 8,
) -> torch.Tensor:
    """返回该视频的逐帧特征 [N, D]，必要时用冻结骨干提取并写缓存。

    缓存附带"帧签名"（文件名+大小+修改时间）元数据；帧被人工清洗/重拆帧后
    签名变化，自动判定缓存失效并重新提取，保证训练与用户清洗后的帧严格一致。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = video_key_of(frames_dir)
    cache_path = feature_cache_path(cache_dir, key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = frames_dir_to_paths(frames_dir)
    signature = _frame_signature(frame_paths)

    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                if json.load(f).get("signature") == signature:
                    return torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            pass  # 元数据损坏则重新提取

    feats = extract_frame_features(backbone, processor, frame_paths, device, batch_size=batch_size)
    torch.save(feats, cache_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"signature": signature}, f)
    return feats


def _frame_signature(frame_paths: list[Path]) -> list[tuple[str, int, int]]:
    """帧签名：[(文件名, 大小, mtime_ns)]，用于缓存失效判断。"""
    sig = []
    for p in frame_paths:
        try:
            st = p.stat()
            sig.append((p.name, st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append((p.name, 0, 0))
    return sig


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class VideoFeatureDataset(Dataset):
    """基于缓存逐帧特征的数据集（不应用增强）。每项 = (features [N,D], label)。"""

    def __init__(self, items: list[tuple[torch.Tensor, int]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feats, label = self.items[idx]
        return feats, label


class AugmentedFrameDataset(Dataset):
    """训练期实时从帧图像提取特征（应用增强）。每项 = (frames_dir, label)。"""

    def __init__(self, entries, backbone, processor, device, transform, batch_size=8):
        self.entries = entries  # list[(frames_dir, label)]
        self.backbone = backbone
        self.processor = processor
        self.device = device
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        frames_dir, label = self.entries[idx]
        paths = frames_dir_to_paths(frames_dir)
        if not paths:
            return torch.empty((0, 0), dtype=torch.float32), label
        feats = extract_frame_features(
            self.backbone,
            self.processor,
            paths,
            self.device,
            batch_size=self.batch_size,
            transform=self.transform,
        )
        return feats, label


# ---------------------------------------------------------------------------
# collate：变长帧序列按 batch 内最大长度 padding + mask
# ---------------------------------------------------------------------------
def collate_videos(batch):
    """把 ``(features[N,D], label)`` 列表打包为 (padded[B,T,D], mask[B,T], labels[B])。

    训练时一个 batch 含多个视频、帧数不同，需 padding；掩码注意力据此忽略 pad。
    （推理单视频、无 padding 时 mask 全为 True。）
    """
    feats_list, labels = zip(*batch)
    labels = torch.tensor(list(labels), dtype=torch.long)
    max_t = max(f.shape[0] for f in feats_list)
    d = feats_list[0].shape[1] if max_t > 0 else 0
    B = len(feats_list)
    padded = torch.zeros((B, max_t, d), dtype=torch.float32)
    mask = torch.zeros((B, max_t), dtype=torch.bool)
    for i, f in enumerate(feats_list):
        t = f.shape[0]
        if t == 0:
            continue
        padded[i, :t] = f
        mask[i, :t] = True
    return padded, mask, labels


# ---------------------------------------------------------------------------
# 构建 train/val
# ---------------------------------------------------------------------------
def build_train_val(
    labels: list[dict],
    cache_dir: Path,
    backbone,
    processor,
    device,
    val_fraction: float = 0.2,
    batch_size: int = 8,
    seed: int = 42,
):
    """构建 (train_ds, val_ds)。基于缓存特征，标签分层划分。

    帧目录为空/无帧的视频被跳过（空文件夹异常由调用方决定是否防御）。
    """
    grouped: dict[int, list] = defaultdict(list)
    for item in labels:
        frames_dir = resolve_frames_dir(item["video_path"])
        feats = ensure_video_features(frames_dir, cache_dir, backbone, processor, device, batch_size)
        if feats.shape[0] == 0:
            continue
        grouped[int(item["label"])].append((feats, int(item["label"])))

    rng = torch.Generator().manual_seed(seed)
    train_items, val_items = [], []
    for label, items in grouped.items():
        idx = torch.randperm(len(items), generator=rng).tolist()
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        val_idx = set(idx[:n_val])
        for i in idx:
            (val_items if i in val_idx else train_items).append(items[i])

    return VideoFeatureDataset(train_items), VideoFeatureDataset(val_items)


def build_augmented_train(
    labels: list[dict],
    cache_dir: Path,
    backbone,
    processor,
    device,
    val_fraction: float = 0.2,
    seed: int = 42,
    image_size: int = config.DEFAULT_IMAGE_SIZE,
    batch_size: int = 8,
):
    """构建增强训练集（实时从帧提取）+ 缓存验证集。

    Returns
    -------
    (train_aug_ds, val_ds)
    """
    mean, std = processor_norm(processor)
    transform = build_augment_transform(image_size, mean=mean, std=std)

    grouped: dict[int, list] = defaultdict(list)
    for item in labels:
        frames_dir = resolve_frames_dir(item["video_path"])
        grouped[int(item["label"])].append((frames_dir, int(item["label"])))

    rng = torch.Generator().manual_seed(seed)
    train_entries, val_entries = [], []
    for label, items in grouped.items():
        idx = torch.randperm(len(items), generator=rng).tolist()
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        val_idx = set(idx[:n_val])
        for i in idx:
            (val_entries if i in val_idx else train_entries).append(items[i])

    # 过滤无帧的目录（空帧会导致 feature_dim=0 崩溃）
    def _has_frames(entry):
        return len(frames_dir_to_paths(entry[0])) > 0

    train_entries = [e for e in train_entries if _has_frames(e)]

    train_aug = AugmentedFrameDataset(train_entries, backbone, processor, device, transform, batch_size)
    # 验证集基于缓存特征（不增强）
    val_items = []
    for frames_dir, label in val_entries:
        feats = ensure_video_features(frames_dir, cache_dir, backbone, processor, device, batch_size)
        if feats.shape[0] > 0:
            val_items.append((feats, label))
    return train_aug, VideoFeatureDataset(val_items)
