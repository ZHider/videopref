"""冻结 DINOv3 骨干加载器。

使用 transformers 的 ``DINOv3ViTModel`` 从本地目录加载官方 base 权重，
返回 ``(model, image_processor, feature_dim)``。模型被冻结并置为 eval 模式。

本地缺权重时，若给定 ``model_id`` 会自动从 ModelScope 下载（HF 为 gated，
ModelScope 无需许可；运行时加载始终 ``local_files_only=True``，不联网 HF）。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from . import config

try:
    from transformers import DINOv3ViTImageProcessor, DINOv3ViTModel
except Exception:  # pragma: no cover - 兼容较旧 transformers
    from transformers import AutoModel as DINOv3ViTModel
    from transformers import AutoImageProcessor as DINOv3ViTImageProcessor


def _backbone_ready(model_dir: Path) -> bool:
    """本地是否已具备可加载的骨干：config.json + 至少一个 *.safetensors。"""
    return (model_dir / config.BACKBONE_CONFIG_NAME).is_file() and any(model_dir.glob("*.safetensors"))


def ensure_backbone(
    model_dir: Path | str,
    model_id: str | None = None,
) -> Path:
    """确保骨干权重在本地；缺失且给定 ``model_id`` 时自动从 ModelScope 下载。

    Parameters
    ----------
    model_dir : 骨干权重目录。
    model_id : ModelScope/HF 模型标识（如 ``config.DEFAULT_BACKBONE_ID``）。
        为 None 时不做自动下载，缺失直接报错。

    Returns
    -------
    确保就绪的 ``model_dir``（Path）。
    """
    model_dir = Path(model_dir)
    if _backbone_ready(model_dir):
        return model_dir

    if model_id is None:
        raise FileNotFoundError(
            f"找不到 DINOv3 骨干权重: {model_dir}（需 config.json + *.safetensors），"
            "且未提供 model_id 供自动下载。"
        )

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError("自动下载骨干需先安装 modelscope：`uv add modelscope`") from e

    # 权重与 modelscope 缓存一律落在项目文件夹内，不写用户 HOME 目录
    os.environ.setdefault("MODELSCOPE_CACHE", str(config.PROJECT_ROOT / ".modelscope"))
    snapshot_download(model_id, local_dir=str(model_dir), allow_patterns=["*.safetensors", "*.json"])
    if not _backbone_ready(model_dir):
        raise RuntimeError(f"从 ModelScope 下载后仍未找到权重: {model_dir}")
    return model_dir


def load_backbone(
    model_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    local_files_only: bool = True,
) -> tuple[nn.Module, DINOv3ViTImageProcessor, int]:
    """加载冻结的 DINOv3 骨干（本地缺失时自动从 ModelScope 下载默认骨干）。

    Parameters
    ----------
    model_dir : 骨干权重目录。缺省使用 ``config.DEFAULT_BACKBONE_DIR``；
        该默认目录缺失时用 ``config.DEFAULT_BACKBONE_ID`` 自动下载。
    device : 目标设备，缺省按 CUDA 是否可用自动选择。
    local_files_only : 从本地加载不联网（True 默认，自动下载完成后再从本地加载）。

    Returns
    -------
    (model, image_processor, feature_dim)
        model 已冻结并 ``eval()``；feature_dim 取自配置的 hidden_size。
    """
    if model_dir is None:
        model_dir = config.DEFAULT_BACKBONE_DIR
        model_id = config.DEFAULT_BACKBONE_ID
    else:
        model_id = None  # 自定义目录：不自动下载（无法确定 model_id）
    model_dir = ensure_backbone(model_dir, model_id)

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
