"""推理辅助：从 Checkpoint 重建模型与骨干，对媒体条目做预测。

推理端无会话状态、不缓存特征、不存储中间结果；所有超参数从 Checkpoint 读取。
"""

from __future__ import annotations

from pathlib import Path

from .items import MediaItem
from .predictor import Predictor


def infer_frames(
    item: MediaItem,
    checkpoint_path: Path | str,
    model_dir: Path | str | None = None,
    device=None,
    batch_size: int = 8,
) -> dict:
    """对一媒体条目（视频=帧目录 / 图片=单文件）做推理，返回结构化结果。

    Returns
    -------
    dict::
        {
          "item_path": str,
          "num_frames": int,
          "like_probability": float,
          "label_mapping": dict,
          "config": dict,
          "training_stats": dict,
        }
    """
    # 全部超参数从 Checkpoint 读取；骨干/模型只加载一次
    predictor = Predictor(checkpoint_path, backbone_dir=model_dir, device=device)
    like_prob, num_frames = predictor.predict_item(item, batch_size=batch_size)

    return {
        "item_path": str(item.path),
        "num_frames": num_frames,
        "like_probability": round(like_prob, 6),
        "label_mapping": predictor.label_mapping,
        "config": predictor.config,
        "training_stats": predictor.payload.get("training_stats", {}),
    }
