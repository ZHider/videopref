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
    import queue
    import threading

    if len(frame_paths) == 0:
        return torch.empty((0, int(model.config.hidden_size)), dtype=torch.float32)

    chunks = [frame_paths[i : i + batch_size] for i in range(0, len(frame_paths), batch_size)]
    n = len(chunks)

    def _preprocess_cpu(chunk):
        """纯 CPU 预处理（解码/缩放/归一化），返回 CPU 张量。线程安全，不碰 CUDA。"""
        if transform is not None:
            from PIL import Image

            imgs = [Image.open(p).convert("RGB") for p in chunk]
            return torch.stack([transform(img) for img in imgs])
        inputs = processor(images=[str(p) for p in chunk], return_tensors="pt")
        return inputs["pixel_values"]

    # 同步路径：只有一批或未启用预取时，不用线程/队列，直接顺序处理
    if n <= 1 or prefetch < 1:
        feats = []
        for chunk in chunks:
            out = model(_preprocess_cpu(chunk).to(device))
            feats.append(out.pooler_output.float().cpu())
        return torch.cat(feats, dim=0)

    # 预取路径：生产者线程纯 CPU 预处理 -> 有界队列；主线程 .to(device)+GPU 前向。
    # 生产者在队列满时阻塞（背压），使 CPU 预处理与 GPU 前向重叠，GPU 不空等 CPU。
    q = queue.Queue(maxsize=max(1, prefetch))

    def _producer():
        try:
            for chunk in chunks:
                q.put(_preprocess_cpu(chunk))
        finally:
            q.put(None)  # 哨兵

    threading.Thread(target=_producer, daemon=True).start()

    feats = []
    got = 0
    while got < n:
        cpu_batch = q.get()
        if cpu_batch is None:
            break
        got += 1
        out = model(cpu_batch.to(device))
        # pooler_output = sequence_output[:, 0, :] 即 [CLS] 特征
        feats.append(out.pooler_output.float().cpu())
    return torch.cat(feats, dim=0)


def frames_dir_to_paths(frames_dir: Path) -> list[Path]:
    """枚举帧文件（按文件名排序恢复时序）。"""
    return list_frame_files(frames_dir)
