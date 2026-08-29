"""训练期数据增强（仅训练阶段运行时应用，不影响缓存的特征向量）。

从文件路径直接构造经增强的张量。图像尺寸/均值/方差取自骨干 image_processor，
确保与无增强路径的归一化一致。
"""

from __future__ import annotations

from pathlib import Path

from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet 默认归一化（DINOv3 与之一致；会按需被 processor 覆盖）
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_augment_transform(
    image_size: int,
    mean=_MEAN,
    std=_STD,
    color_jitter: float = 0.3,
    max_rotation: float = 5.0,
) -> transforms.Compose:
    """构造训练期数据增强。

    随机水平翻转 + 颜色抖动 + 轻微旋转，最后 Resize/ToTensor/Normalize。
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.1),
            transforms.RandomRotation(degrees=max_rotation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def augment_path(transform, path: Path):
    """对单张帧文件应用增强变换并返回张量。"""
    return transform(str(path))


def processor_norm(processor) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """从 image_processor 提取归一化 mean/std，用于训练期增强保持一致。"""
    mean = tuple(getattr(processor, "image_mean", _MEAN) or _MEAN)
    std = tuple(getattr(processor, "image_std", _STD) or _STD)
    return mean, std


def processor_image_size(processor) -> int:
    """从 image_processor 提取输入尺寸。"""
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        h = size.get("height") or size.get("shortest_edge")
        w = size.get("width") or size.get("shortest_edge")
        return int(h if h else (w if w else 224))
    if isinstance(size, (tuple, list)):
        return int(size[0])
    if isinstance(size, int):
        return int(size)
    return 224
