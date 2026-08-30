"""🔍 推理 Tab：选条目（视频/图片）+ Checkpoint -> 喜好概率 + JSON。"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from .. import config
from ..inference import infer_frames
from ..items import MediaItem
from .common import frame_item_choices, list_checkpoints


def do_infer(frames_key, checkpoint_path, backbone_dir):
    if not frames_key:
        return None, json.dumps({"error": "请选择媒体条目"}, ensure_ascii=False, indent=2)
    if not checkpoint_path:
        return None, json.dumps({"error": "请选择 Checkpoint"}, ensure_ascii=False, indent=2)
    try:
        item = MediaItem.from_entry_path(Path(config.FRAMES_ROOT) / frames_key)
        result = infer_frames(item, checkpoint_path, model_dir=backbone_dir)
        return (
            result["like_probability"],
            json.dumps(result, ensure_ascii=False, indent=2),
        )
    except Exception as e:
        return None, json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)


def build_infer_tab():
    with gr.Tab("🔍 推理"):
        with gr.Row():
            frames_dd = gr.Dropdown(
                label="媒体条目（扫描 frames/video 与 frames/image）",
                choices=frame_item_choices(),
                interactive=True,
            )
            ckpt_dd = gr.Dropdown(
                label="Checkpoint（扫描 checkpoints/）",
                choices=list_checkpoints(),
                interactive=True,
            )
        with gr.Row():
            refresh_btn = gr.Button("刷新列表")
            backbone_dir = gr.Textbox(
                label="骨干权重目录（可选）",
                value=str(config.DEFAULT_BACKBONE_DIR),
            )
        infer_btn = gr.Button("开始推理", variant="primary")
        like_prob = gr.Number(label="like_probability (0-1)", value=None, interactive=False)
        json_out = gr.Textbox(label="JSON 结构化结果", lines=12, interactive=False)

        def refresh():
            return gr.update(choices=frame_item_choices()), gr.update(choices=list_checkpoints())

        refresh_btn.click(refresh, outputs=[frames_dd, ckpt_dd])
        infer_btn.click(
            do_infer,
            inputs=[frames_dd, ckpt_dd, backbone_dir],
            outputs=[like_prob, json_out],
        )
        frames_dd.change(
            do_infer,
            inputs=[frames_dd, ckpt_dd, backbone_dir],
            outputs=[like_prob, json_out],
        )
