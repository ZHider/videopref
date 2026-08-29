"""推理辅助：从 Checkpoint 重建模型与骨干，对清洗后的帧目录做预测。

推理端无会话状态、不缓存特征、不存储中间结果；所有超参数从 Checkpoint 读取。
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import config
from .backbone import load_backbone
from .features import extract_frame_features, frames_dir_to_paths
from .model import VideoPreferenceModel, load_checkpoint


def infer_frames(
    frames_dir: Path | str,
    checkpoint_path: Path | str,
    model_dir: Path | str | None = None,
    device=None,
    batch_size: int = 8,
) -> dict:
    """对一帧目录做推理，返回结构化结果。

    Returns
    -------
    dict::
        {
          "frames_dir": str,
          "num_frames": int,
          "like_probability": float,
          "label_mapping": dict,
          "config": dict,
          "training_stats": dict,
        }
    """
    frames_dir = Path(frames_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 从 Checkpoint 读取所有超参数
    payload = load_checkpoint(checkpoint_path, device=device)
    ckpt_config = payload["config"]
    label_mapping = payload.get("label_mapping", config.LABEL_MAPPING)
    feature_dim = int(ckpt_config.get("feature_dim", config.DEFAULT_FEATURE_DIM))

    # 2) 加载冻结骨干（用 Checkpoint 记录的 backbone_id/目录）
    backbone_id = ckpt_config.get("backbone_id", config.DEFAULT_BACKBONE_ID)
    if model_dir is None:
        model_dir = config.DEFAULT_BACKBONE_DIR
    backbone, processor, _ = load_backbone(model_dir, device=device)

    # 3) 提取帧特征 + 聚合 + 分类
    frame_paths = frames_dir_to_paths(frames_dir)
    feats = extract_frame_features(backbone, processor, frame_paths, device, batch_size=batch_size)
    if feats.shape[0] == 0:
        raise ValueError(f"帧目录为空或无帧: {frames_dir}")

    model = VideoPreferenceModel(feature_dim=feature_dim).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    with torch.no_grad():
        prob = model(feats.to(device).unsqueeze(0), mask=None)  # [1]
        like_prob = float(prob.squeeze().cpu())

    return {
        "frames_dir": str(frames_dir),
        "num_frames": int(feats.shape[0]),
        "like_probability": round(like_prob, 6),
        "label_mapping": label_mapping,
        "config": ckpt_config,
        "training_stats": payload.get("training_stats", {}),
    }
