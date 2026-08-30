"""Gradio UI 共享工具：条目/checkpoint 扫描、File 值解析、HTML 进度条、抽帧参数组件。

另含 ``JobState``：供批量推理/训练 Tab 使用的线程安全单例任务状态
（单进程 Gradio 足够；集中锁与字段访问，避免各处散乱操作模块级 dict）。
"""

from __future__ import annotations

import threading
from pathlib import Path

import gradio as gr

from .. import config
from ..items import MediaItem


class JobState:
    """线程安全的任务状态容器（后台线程写、Timer 轮询读）。

    用法：``state = JobState({"running": False, ...})``；后台线程
    ``state.update(...)``，UI 线程 ``state.snapshot()`` 取一致快照。
    """

    def __init__(self, fields: dict):
        self._lock = threading.Lock()
        self._data = dict(fields)

    def update(self, **kwargs) -> None:
        with self._lock:
            self._data.update(kwargs)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]


def list_checkpoints() -> list[str]:
    """扫描 checkpoints/ 下所有 .ckpt，返回绝对路径列表。"""
    d = Path(config.CHECKPOINTS_DIR)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.ckpt") if p.is_file())


def list_frame_items() -> list[str]:
    """扫描 frames/ 两个子区，返回已摄入条目 key（``video/foo`` / ``image/foo.png``）。"""
    return [it.key for it in MediaItem.scan(config.FRAMES_ROOT)]


def frame_item_choices() -> list[tuple[str, str]]:
    """条目下拉选项：(带类型图标的显示名, key)。图片=🖼，视频=🎬。"""
    return [
        ("🖼 " + k if k.startswith(config.FRAMES_IMAGE_SUBDIR + "/") else "🎬 " + k, k)
        for k in list_frame_items()
    ]


def file_value_to_path(v) -> Path | None:
    """gradio 6 的 File 值可能是 str 或 FileData。"""
    if v is None:
        return None
    if isinstance(v, str):
        return Path(v)
    if hasattr(v, "path"):
        return Path(v.path)
    if hasattr(v, "name"):
        return Path(v.name)
    return Path(str(v))


def progress_html(cur: int, total: int, desc: str = "") -> str:
    """生成一个 HTML 进度条（Tab 内可见，由 Timer 轮询更新）。"""
    pct = int(cur / total * 100) if total else 0
    return (
        '<div style="width:100%;background:#e5e7eb;border-radius:6px;height:18px;overflow:hidden">'
        f'<div style="width:{pct}%;background:#22c55e;height:18px;border-radius:6px"></div></div>'
        f'<div style="margin-top:4px;font-size:13px;color:#374151">{desc or "处理中"} {cur}/{total} ({pct}%)</div>'
    )


def sampling_accordion():
    """抽帧参数（uniform/keyframe/scene）。拆帧与批量推理各建一份。"""
    with gr.Accordion("抽帧参数（可选）", open=False):
        sampling = gr.Radio(
            ["uniform", "keyframe", "scene"],
            value=config.DEFAULT_SAMPLING,
            label="抽帧方式（uniform=时长均匀，keyframe=只解关键帧更快但粗糙，scene=场景检测）",
        )
        fps_target = gr.Slider(0.1, 2.0, value=config.FPS_TARGET, step=0.05, label="目标帧密度（帧/秒）")
        min_frames = gr.Slider(2, 16, value=float(config.MIN_FRAMES), step=1, label="最少帧数")
        max_frames = gr.Slider(8, 128, value=float(config.DEFAULT_MAX_FRAMES), step=1, label="最多帧数")
        scene_threshold = gr.Slider(0.05, 1.0, value=config.DEFAULT_SCENE_THRESHOLD, step=0.01, label="scene 阈值（scene 模式）")
        black_thresh = gr.Slider(1, 60, value=float(config.BLACK_FRAME_MEAN), step=1, label="纯黑帧灰度均值阈值")
        white_thresh = gr.Slider(200, 254, value=float(config.WHITE_FRAME_MEAN), step=1, label="纯白帧灰度均值阈值")
    return sampling, fps_target, min_frames, max_frames, scene_threshold, black_thresh, white_thresh
