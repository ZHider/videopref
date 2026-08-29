"""冻结 DINOv3 骨干加载器。

使用 transformers 的 ``DINOv3ViTModel`` 从本地目录加载官方 base 权重，
返回 ``(model, image_processor, feature_dim)``。模型被冻结并置为 eval 模式。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import config

try:
    from transformers import DINOv3ViTImageProcessor, DINOv3ViTModel
except Exception:  # pragma: no cover - 兼容较旧 transformers
    from transformers import AutoModel as DINOv3ViTModel
    from transformers import AutoImageProcessor as DINOv3ViTImageProcessor


def load_backbone(
    model_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    local_files_only: bool = True,
) -> tuple[nn.Module, DINOv3ViTImageProcessor, int]:
    """从本地目录加载冻结的 DINOv3 骨干。

    Parameters
    ----------
    model_dir : 骨干权重目录（含 config.json / model.safetensors）。
        缺省使用 ``config.DEFAULT_BACKBONE_DIR``。
    device : 目标设备，缺省按 CUDA 是否可用自动选择。
    local_files_only : 仅从本地加载，不联网。

    Returns
    -------
    (model, image_processor, feature_dim)
        model 已冻结并 ``eval()``；feature_dim 取自配置的 hidden_size。
    """
    if model_dir is None:
        model_dir = config.DEFAULT_BACKBONE_DIR
    model_dir = Path(model_dir)
    if not (model_dir / config.BACKBONE_CONFIG_NAME).is_file():
        raise FileNotFoundError(
            f"找不到 DINOv3 骨干配置: {model_dir}. "
            f"请先将模型下载到 {model_dir}（config.json + model.safetensors）。"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DINOv3ViTModel.from_pretrained(str(model_dir), local_files_only=local_files_only)
    processor = DINOv3ViTImageProcessor.from_pretrained(str(model_dir), local_files_only=local_files_only)

    feature_dim = int(model.config.hidden_size)

    model.to(device)
    model.eval()
    freeze_model(model)

    return model, processor, feature_dim


def freeze_model(model: nn.Module) -> None:
    """冻结全部参数：骨干在训练/推理全程不更新。"""
    for param in model.parameters():
        param.requires_grad_(False)


def trainable_parameter_count(model: nn.Module) -> int:
    """统计可训练参数数量（应只含池化层与分类头）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
