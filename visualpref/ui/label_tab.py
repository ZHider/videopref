"""🏷️ 标注 Tab：对 frames/ 已摄入媒体（视频/图片）逐项看图标注，导出 labels.json。"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .. import config
from ..labeling import (
    advance,
    build_queue_from_frames,
    current_entry,
    load_progress,
    new_state,
    render_state,
    sanitize_state,
    save_labels,
    save_progress,
)


def do_resume(state):
    saved = load_progress()
    if saved is None:
        return "没有可续标的历史进度。", "", "", [], state
    state = sanitize_state(saved, config.FRAMES_ROOT)
    save_progress(state)
    name, prog, imgs = render_state(state)
    return f"已恢复上次标注（{len(state['queue'])} 个媒体）。", name, prog, imgs, state


def do_scan_all(state):
    """扫描 frames/ 下全部已摄入媒体并开始标注（不重新拆帧）。"""
    queue = build_queue_from_frames(Path(config.FRAMES_ROOT))
    if not queue:
        return "frames/ 下没有已摄入条目，请先拆帧/摄入。", "", "", [], state
    state = {"queue": queue, "idx": 0, "labels": {}, "skipped": []}
    save_progress(state)
    name, prog, imgs = render_state(state)
    return f"已载入 frames/ 下 {len(queue)} 个媒体，开始标注。", name, prog, imgs, state


def _commit(state) -> tuple[str, str, list, dict]:
    """前进/回退后统一持久化并渲染。"""
    save_progress(state)
    name, prog, imgs = render_state(state)
    return name, prog, imgs, state


def do_like(state):
    state = advance(state, label=1)
    return _commit(state)


def do_dislike(state):
    state = advance(state, label=0)
    return _commit(state)


def do_skip(state):
    entry = current_entry(state)
    if entry is not None:
        state.setdefault("skipped", []).append(entry["key"])
    state = advance(state, label=None)
    return _commit(state)


def do_prev(state):
    state["idx"] = max(0, int(state.get("idx", 0)) - 1)
    return _commit(state)


def do_export(state):
    labels_path = config.DATA_DIR / "labels.json"
    n = save_labels(labels_path, state.get("queue", []), state.get("labels", {}))
    return f"已导出 {n} 条标注 -> {labels_path}（label: 1=喜欢, 0=不喜欢）"


def build_label_tab():
    with gr.Tab("🏷️ 标注"):
        gr.Markdown("先到「🎬 拆帧」Tab 拆帧/摄入，再在这里对 `frames/` 下已摄入的媒体（视频多帧 🎬 / 图片单帧 🖼）逐一看图标注。")
        state = gr.State(new_state())
        with gr.Row():
            btn_scan_all = gr.Button("标注 frames/ 全部已摄入媒体", variant="primary")
            btn_resume = gr.Button("继续上次标注")
        status_out = gr.Textbox(label="状态", lines=2, interactive=False)
        video_name = gr.Markdown("等待开始…")
        progress_lbl = gr.Markdown("")
        with gr.Row():
            columns_ctl = gr.Slider(1, 10, value=5, step=1, label="每行预览数")
            height_ctl = gr.Slider(200, 900, value=600, step=50, label="预览高度（px）")
        gallery = gr.Gallery(
            label="帧预览（按时间顺序；图片仅 1 张）", columns=5, height=600, object_fit="contain",
        )
        with gr.Row():
            btn_like = gr.Button("👍 喜欢", variant="primary")
            btn_dislike = gr.Button("👎 不喜欢", variant="stop")
            btn_skip = gr.Button("跳过")
            btn_prev = gr.Button("上一步")
        btn_export = gr.Button("导出 labels.json")
        export_out = gr.Textbox(label="导出结果", lines=2, interactive=False)

        label_inputs = [state]
        label_outputs = [video_name, progress_lbl, gallery, state]

        btn_scan_all.click(do_scan_all, inputs=label_inputs, outputs=[status_out, video_name, progress_lbl, gallery, state])
        btn_resume.click(do_resume, inputs=label_inputs, outputs=[status_out, video_name, progress_lbl, gallery, state])
        btn_like.click(do_like, inputs=label_inputs, outputs=label_outputs)
        btn_dislike.click(do_dislike, inputs=label_inputs, outputs=label_outputs)
        btn_skip.click(do_skip, inputs=label_inputs, outputs=label_outputs)
        btn_prev.click(do_prev, inputs=label_inputs, outputs=label_outputs)
        btn_export.click(do_export, inputs=label_inputs, outputs=[export_out])

        def set_gallery_columns(columns):
            return gr.update(columns=int(columns))

        def set_gallery_height(height):
            return gr.update(height=int(height))

        columns_ctl.change(set_gallery_columns, inputs=[columns_ctl], outputs=[gallery])
        height_ctl.change(set_gallery_height, inputs=[height_ctl], outputs=[gallery])
