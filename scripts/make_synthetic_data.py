#!/usr/bin/env python
"""生成合成测试数据：两种“风格”的假视频 + 标注。

- like(1)：暖色调（橙红渐变）。
- dislike(0)：冷色调（蓝青渐变）。

每条视频由若干不同帧编码为真实 mp4，供拆帧->训练->推理端到端验证。
用法::

    python scripts/make_synthetic_data.py --n-like 12 --n-dislike 12 \\
        --videos ./data/synthetic_videos --labels ./data/labels.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def make_frames(n_frames: int, warm: bool, seed: int) -> list[np.ndarray]:
    """生成 n_frames 张 HxW 渐变图（RGB uint8）。warm=True 暖色，False 冷色。"""
    rng = np.random.default_rng(seed)
    H, W = 240, 320
    frames = []
    base = (255, 120, 60) if warm else (60, 120, 255)
    for i in range(n_frames):
        img = np.zeros((H, W, 3), dtype=np.float32)
        # 沿宽度渐变 + 每帧微变，制造场景差异
        t = i / max(n_frames - 1, 1)
        c0 = np.array(base, dtype=np.float32) * (0.7 + 0.6 * t)
        c1 = np.array(base[::-1] if rng.random() < 0.5 else base, dtype=np.float32) * (0.5 + 0.5 * (1 - t))
        for w in range(W):
            k = w / (W - 1)
            color = c0 * (1 - k) + c1 * k
            img[:, w] = color
        # 加入一个移动色块以增强场景变化
        x = int((i * 17) % W)
        y = int((i * 13) % H)
        img[y : y + 40, x : x + 40] += 40
        img = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(img)
    return frames


def encode_video(frames: list[np.ndarray], out_mp4: Path) -> None:
    """把帧序列编码为 mp4（libx264）。"""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(tmpd / f"{i:06d}.png")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg")
        res = subprocess.run(
            [
                ffmpeg, "-y",
                "-framerate", "5",
                "-i", str(tmpd / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_mp4),
            ],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"编码失败: {res.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-like", type=int, default=12)
    ap.add_argument("--n-dislike", type=int, default=12)
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--videos", default="data/synthetic_videos")
    ap.add_argument("--labels", default="data/labels.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    videos_dir = Path(args.videos)
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels = []

    def gen(n, warm, prefix):
        for i in range(n):
            name = f"{prefix}{i:03d}.mp4"
            frames = make_frames(args.frames_per_video, warm, seed=args.seed + i)
            encode_video(frames, videos_dir / name)
            labels.append(
                {
                    "media_path": str(videos_dir / name),
                    "label": 1 if warm else 0,
                    "kind": "video",
                }
            )

    gen(args.n_like, warm=True, prefix="like")
    gen(args.n_dislike, warm=False, prefix="dislike")

    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"[done] 生成 {len(labels)} 条视频 -> {videos_dir}")
    print(f"[done] 标注 -> {labels_path}")


if __name__ == "__main__":
    main()
