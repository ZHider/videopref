"""特征提取管线。

- ``extract_frame_features``：冻结 DINOv3 批量提取每帧 ``[CLS]`` 特征。
- ``frames_dir_to_paths``：枚举帧文件（按文件名排序恢复时序）。

模型类（``VideoPreferenceModel`` 等）位于 ``model.py``，此处 re-export 以便兼容既有导入。

推理端不做特征缓存，实时提取，保证与用户清洗后的帧状态严格一致。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import config
from .model import MaskedAttentionPooling, PreferenceHead, VideoPreferenceModel  # noqa: F401  (re-export)
from .paths import list_frame_files

__all__ = [
    "extract_frame_features",
    "frames_dir_to_paths",
    "MaskedAttentionPooling",
    "PreferenceHead",
    "VideoPreferenceModel",
]


@torch.no_grad()
def extract_frame_features(
    model: nn.Module,
    processor,
    frame_paths: list[Path],
    device: torch.device | str,
    batch_size: int = 8,
    transform=None,
) -> torch.Tensor:
    """用冻结骨干批量提取每帧 ``[CLS]`` 特征。

    Parameters
    ----------
    transform : 可选的数据增强变换（仅训练时传入）。若提供，则跳过
        processor 的标准化流程而改用自定义 transform（应包含 Resize/ToTensor/Normalize）。
        否则使用 ``processor`` 做预处理。

    Returns
    -------
    [N, D] float32 tensor（CPU）。
    """
    if len(frame_paths) == 0:
        return torch.empty((0, int(model.config.hidden_size)), dtype=torch.float32)

    from PIL import Image

    feats = []
    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start : start + batch_size]
        if transform is not None:
            # 增强路径：先用 PIL 打开图像，再应用 transform（含 Resize/ToTensor/Normalize）
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            batch = torch.stack([transform(img) for img in imgs]).to(device)
        else:
            inputs = processor(images=[str(p) for p in chunk], return_tensors="pt")
            batch = inputs["pixel_values"].to(device)
        out = model(batch)
        # pooler_output = sequence_output[:, 0, :] 即 [CLS] 特征
        feats.append(out.pooler_output.float().cpu())
    return torch.cat(feats, dim=0)


def frames_dir_to_paths(frames_dir: Path) -> list[Path]:
    """枚举帧文件（按文件名排序恢复时序）。"""
    return list_frame_files(frames_dir)
