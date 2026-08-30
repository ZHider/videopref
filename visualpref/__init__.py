"""visual-pref — 个人媒体（视频/图片）喜好二分类器。

核心架构: 冻结 DINOv3 骨干 + Masked Attention Pooling + 轻量 MLP 分类头。
人机协同: 拆帧/摄入 -> 用户人工清洗 -> 特征提取/训练/推理。
"""

from . import config, paths

__all__ = ["config", "paths"]
__version__ = "0.2.0"
