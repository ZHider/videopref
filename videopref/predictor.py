"""推理预测器：一次性加载冻结骨干 + Checkpoint + 池化分类模型。

单视频推理（``inference.py``）与批量推理（``batch_infer.py``）过去各自重复
"加载骨干 -> 读 Checkpoint -> 重建模型 -> load_state_dict -> eval" 的逻辑。
本模块把这一整套收敛为 ``Predictor``：构造时只加载一次，之后可反复对
逐帧特征（或帧目录）做预测。所有超参数一律从 Checkpoint 读取，禁止硬编码。
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import config
from .backbone import load_backbone
from .features import extract_frame_features, frames_dir_to_paths
from .model import VideoPreferenceModel, load_checkpoint


class Predictor:
    """封装冻结骨干 + 池化分类模型，供单视频/批量推理复用。

    Attributes
    ----------
    device : torch.device
    backbone : nn.Module（已冻结、eval）
    processor : image processor
    model : VideoPreferenceModel（池化 + 分类头，已载入权重、eval）
    payload : 完整 checkpoint 字典（含 config / label_mapping / training_stats）
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        backbone_dir: Path | str | None = None,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if backbone_dir is None:
            backbone_dir = config.DEFAULT_BACKBONE_DIR
        self.device = device

        # 1) Checkpoint：所有超参数从这里读取
        self.payload = load_checkpoint(checkpoint_path, device=device)
        self.config = self.payload["config"]
        self.label_mapping = self.payload.get("label_mapping", config.LABEL_MAPPING)
        self.feature_dim = int(self.config.get("feature_dim", config.DEFAULT_FEATURE_DIM))

        # 2) 冻结骨干（只加载一次）
        self.backbone, self.processor, _ = load_backbone(backbone_dir, device=device)

        # 3) 池化 + 分类模型（只加载一次）
        self.model = VideoPreferenceModel(feature_dim=self.feature_dim).to(device)
        self.model.load_state_dict(self.payload["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_feats(self, feats: torch.Tensor) -> float:
        """对逐帧特征 ``[N, D]`` 做池化 + 分类，返回喜好概率标量（调用方保证 N>0）。"""
        prob = self.model(feats.to(self.device).unsqueeze(0), mask=None)
        return float(prob.squeeze().cpu())

    def predict_frame_paths(
        self,
        frame_paths: list[Path],
        batch_size: int = 8,
    ) -> tuple[float, int]:
        """实时提取帧特征并预测。

        Returns
        -------
        (like_probability, num_frames)
        """
        feats = extract_frame_features(
            self.backbone,
            self.processor,
            frame_paths,
            self.device,
            batch_size=batch_size,
        )
        if feats.shape[0] == 0:
            raise ValueError("帧目录为空或无帧")
        return self.predict_feats(feats), int(feats.shape[0])

    def predict_frames_dir(
        self,
        frames_dir: Path | str,
        batch_size: int = 8,
    ) -> tuple[float, int]:
        """对一帧目录实时提取特征并预测。"""
        frame_paths = frames_dir_to_paths(Path(frames_dir))
        return self.predict_frame_paths(frame_paths, batch_size=batch_size)
