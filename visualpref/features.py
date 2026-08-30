"""特征提取管线。

- ``extract_frame_features``：冻结 DINOv3 批量提取每帧 ``[CLS]`` 特征。
- ``frames_dir_to_paths``：枚举帧文件（按文件名排序恢复时序）。

CPU 预处理（解码/缩放/归一化）与 GPU 前向的重叠由 ``pipeline.prefetch_map``
负责，本模块只关心"预处理单个 batch + 模型前向"。

推理端不做特征缓存，实时提取，保证与用户清洗后的帧状态严格一致。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .paths import list_frame_files
from .pipeline import prefetch_map

__all__ = ["extract_frame_features", "frames_dir_to_paths"]


@torch.no_grad()
def extract_frame_features(
    model: nn.Module,
    processor,
    frame_paths: list[Path],
    device: torch.device | str,
    batch_size: int = 8,
    transform=None,
    prefetch: int = 2,
) -> torch.Tensor:
    """用冻结骨干批量提取每帧 ``[CLS]`` 特征。

    Parameters
    ----------
    transform : 可选的数据增强变换（仅训练时传入）。若提供，则跳过
        processor 的标准化流程而改用自定义 transform（应包含 Resize/ToTensor/Normalize）。
        否则使用 ``processor`` 做预处理。
    prefetch : 流水线预取深度。CPU 预处理（解码/缩放/归一化）在后台线程进行，
        与 GPU 前向重叠，避免 GPU 空等 CPU（≥2 才有预取效果）。

    Returns
    -------
    [N, D] float32 tensor（CPU）。
    """
    if len(frame_paths) == 0:
        return torch.empty((0, int(model.config.hidden_size)), dtype=torch.float32)

    chunks = [frame_paths[i : i + batch_size] for i in range(0, len(frame_paths), batch_size)]

    def _preprocess_cpu(chunk):
        """纯 CPU 预处理（解码/缩放/归一化），返回 CPU 张量。线程安全，不碰 CUDA。"""
        if transform is not None:
            from PIL import Image

            imgs = [Image.open(p).convert("RGB") for p in chunk]
            return torch.stack([transform(img) for img in imgs])
        inputs = processor(images=[str(p) for p in chunk], return_tensors="pt")
        return inputs["pixel_values"]

    feats = []
    for chunk in prefetch_map(_preprocess_cpu, chunks, depth=prefetch):
        out = model(chunk.to(device))
        # pooler_output = sequence_output[:, 0, :] 即 [CLS] 特征
        feats.append(out.pooler_output.float().cpu())
    return torch.cat(feats, dim=0)


def frames_dir_to_paths(item: Path) -> list[Path]:
    """返回一个媒体条目对应的所有图片路径（按文件名排序恢复时序）。

    - 目录（视频工作区 ``frames/video/<key>/``）：枚举 ``*.jpg`` 帧。
    - 文件（图片 ``frames/image/<key>.<ext>``）：返回 ``[该文件]`` 单元素列表。
    """
    item = Path(item)
    if item.is_dir():
        return list_frame_files(item)
    if item.is_file():
        return [item]
    return []
