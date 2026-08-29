"""模型定义 + Checkpoint 规范。

- 模型类（``MaskedAttentionPooling`` / ``PreferenceHead`` / ``VideoPreferenceModel``）
  与 Checkpoint 读写同处本模块，语义一致。
- 超参数一律引用 ``config`` 常量，禁止重复硬编码。
- Checkpoint 加载使用 ``weights_only=True``（payload 全为 dict+张量，安全且向后兼容）。
- Gradio 推理端所有超参数从 Checkpoint 读取，禁止硬编码。

Checkpoint 结构::

    {
      "model_state": {...},
      "config": {
        "backbone_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "feature_dim": 768,
        "max_frames": 64
      },
      "label_mapping": {"like": 1, "dislike": 0},
      "training_stats": {"epoch": 50, "val_auc": 0.87}
    }
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from . import config


# ---------------------------------------------------------------------------
# 掩码注意力池化
# ---------------------------------------------------------------------------
class MaskedAttentionPooling(nn.Module):
    """变长帧序列的可学习加权池化。

    以可学习 query 向量与每帧 ``[CLS]`` 特征做点积得到注意力分数，
    对有效位置做 softmax 得到权重，再加权求和为单一视频级特征。

    mask: ``True`` 表示有效帧。推理时单视频输入、无 padding，mask 全为 True。
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Parameter(torch.empty(feature_dim))
        nn.init.normal_(self.query, std=1.0 / math.sqrt(feature_dim))

    def forward(self, frame_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """聚合帧特征。

        Parameters
        ----------
        frame_features : [B, T, D]
        mask : [B, T] bool，True=有效；缺省全部有效。

        Returns
        -------
        [B, D] 视频级特征。
        """
        scores = torch.einsum("btd,d->bt", frame_features, self.query)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
            # 整行全为无效（理论上不出现，防御性处理）
            all_invalid = ~mask.bool().any(dim=-1, keepdim=True)
            scores = torch.where(all_invalid, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (frame_features * weights).sum(dim=1)  # [B, D]


# ---------------------------------------------------------------------------
# 二分类头
# ---------------------------------------------------------------------------
class PreferenceHead(nn.Module):
    """浅层 MLP：Linear -> Logit。概率在外部经 Sigmoid 计算。"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, video_feature: torch.Tensor) -> torch.Tensor:
        """返回未归一化 logit [B, 1]。"""
        return self.fc(video_feature)


# ---------------------------------------------------------------------------
# 组合模型（池化 + 分类头）
# ---------------------------------------------------------------------------
class VideoPreferenceModel(nn.Module):
    """仅含池化层与分类头的可训练模型。骨干冻结在外。"""

    def __init__(self, feature_dim: int = config.DEFAULT_FEATURE_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.pooling = MaskedAttentionPooling(feature_dim)
        self.head = PreferenceHead(feature_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """输出喜好概率（Sigmoid 后）。``return_logits=True`` 时输出 logit。"""
        video_feat = self.pooling(frame_features, mask)
        logit = self.head(video_feat).squeeze(-1)  # [B]
        if return_logits:
            return logit
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def default_config(**overrides) -> dict:
    """默认训练配置；全部引用 config 常量，避免重复硬编码。"""
    cfg = {
        "backbone_id": config.DEFAULT_BACKBONE_ID,
        "feature_dim": config.DEFAULT_FEATURE_DIM,
        "max_frames": config.DEFAULT_MAX_FRAMES,
    }
    cfg.update(overrides)
    return cfg


def save_checkpoint(
    path: Path,
    model: VideoPreferenceModel,
    config_: dict,
    label_mapping: dict,
    training_stats: dict,
) -> Path:
    """按规范保存 Checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
        "config": dict(config_),
        "label_mapping": dict(label_mapping),
        "training_stats": dict(training_stats),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | str, device=None) -> dict:
    """加载 Checkpoint，返回含 ``model_state``/``config``/``label_mapping``/``training_stats`` 的字典。

    使用 ``weights_only=True``：payload 仅含内置类型与张量，安全且兼容历史 checkpoint。
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if device is not None:
        for k, v in payload["model_state"].items():
            payload["model_state"][k] = v.to(device)
    return payload


def build_model_from_config(config_: dict, device=None) -> VideoPreferenceModel:
    """依据 Checkpoint 的 config 重建模型（权重由调用方 load_state_dict 载入）。"""
    feature_dim = int(config_.get("feature_dim", config.DEFAULT_FEATURE_DIM))
    model = VideoPreferenceModel(feature_dim=feature_dim)
    return model
