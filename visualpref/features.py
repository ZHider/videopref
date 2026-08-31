"""特征提取管线。

- ``extract_frame_features``：冻结 DINOv3 批量提取每帧 ``[CLS]`` 特征。

帧枚举（目录/文件统一）属于 ``MediaItem.frame_paths``（见 ``items.py``），
本模块不再关心"条目 -> 帧列表"的解析。

CPU 预处理（解码/缩放/归一化）与 GPU 前向的重叠由 ``pipeline.prefetch_map``
负责，本模块只关心"预处理单个 batch + 模型前向"。

推理端不做特征缓存，实时提取，保证与用户清洗后的帧状态严格一致。
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .pipeline import prefetch_map

__all__ = ["extract_frame_features"]


@torch.inference_mode()
def extract_frame_features(
    model: nn.Module,
    processor,
    frame_paths: list[Path],
    device: torch.device | str,
    batch_size: int = 8,
    transform=None,
    prefetch: int = 2,
    amp: bool = True,
) -> torch.Tensor:
    """用冻结骨干批量提取每帧 ``[CLS]`` 特征。

    Parameters
    ----------
    transform : 可选的数据增强变换（仅训练时传入）。若提供，则跳过
        processor 的标准化流程而改用自定义 transform（应包含 Resize/ToTensor/Normalize）。
        否则使用 ``processor`` 做预处理。
    prefetch : 流水线预取深度。CPU 预处理（解码/缩放/归一化）在后台线程进行，
        与 GPU 前向重叠，避免 GPU 空等 CPU（≥2 才有预取效果）。
    amp : GPU 上前向是否用 BF16 半精度（``torch.autocast``）。RTX 5070 Ti 等
        Blackwell/Ampere 系列可显著提速（冻结骨干约翻倍）且精度几乎无损；
        CPU 或关闭时回退 FP32。

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

    use_amp = amp and torch.device(device).type == "cuda"
    cast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )
    feats = []
    with cast_ctx:
        for chunk in prefetch_map(_preprocess_cpu, chunks, depth=prefetch):
            out = model(chunk.to(device))
            # pooler_output = sequence_output[:, 0, :] 即 [CLS] 特征
            feats.append(out.pooler_output.float().cpu())
    return torch.cat(feats, dim=0)
