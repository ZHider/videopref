"""🎬 拆帧 Tab：媒体（视频/图片）摄入 + 清空全部记录。"""

from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr

from .. import config
from ..frames import extract_from_input
from ..items import MediaItem
from ..labeling import parse_media_list
from .common import file_value_to_path, sampling_accordion


def do_extract(video_file, folder_path, video_list_text, sampling, fps_target, min_frames, max_frames, scene_threshold, black_thresh, white_thresh, recursive, progress=gr.Progress()):
    """拆帧/摄入。优先级：上传文件 > 文件夹路径 > 路径列表文本。"""
    try:
        input_src = None
        f = file_value_to_path(video_file)
        if f is not None:
            input_src = f
        elif folder_path.strip():
            input_src = Path(folder_path.strip())
        elif video_list_text.strip():
            paths = parse_media_list(video_list_text)
            if not paths:
                return "路径列表中没有有效媒体（视频或图片）。", ""
            input_src = paths  # list 形式交给 extract_from_input
        else:
            return "请提供媒体文件、文件夹路径或媒体路径列表。", ""

        out_items = extract_from_input(
            input_src,
            Path(config.FRAMES_ROOT),
            sampling=sampling,
            scene_threshold=float(scene_threshold),
            fps_target=float(fps_target),
            min_frames=int(min_frames),
            max_frames=int(max_frames),
            black_threshold=int(black_thresh),
            white_threshold=int(white_thresh),
            recursive=bool(recursive),
            progress=progress,
        )
        lines = [f"处理完成，共 {len(out_items)} 个条目（视频工作区/图片文件）："]
        lines += [f"  {d.path}" for d in out_items]
        return "\n".join(lines), "\n".join(str(d.path) for d in out_items)
    except Exception as e:  # 容错：将异常反馈到 UI
        return f"拆帧失败：{e}", ""


def do_clear_frames(confirmed: bool) -> str:
    """清空 frames/ 下的全部媒体条目（video/ 与 image/ 两个子区）与 _manifest.json。"""
    if not confirmed:
        return "未勾选确认，已取消清空。"
    root = Path(config.FRAMES_ROOT)
    if not root.is_dir():
        return "frames/ 目录不存在，无需清空。"
    dirs = files = removed = 0
    for p in MediaItem.iter_entry_paths(root):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            dirs += 1
        else:
            p.unlink(missing_ok=True)
            files += 1
    mf = root / "_manifest.json"
    if mf.is_file():
        mf.unlink(missing_ok=True)
        removed += 1
    return f"已清空 {dirs} 个视频工作区、{files} 个图片文件、{removed} 个 manifest。"


def build_extract_tab():
    with gr.Tab("🎬 拆帧"):
        gr.Markdown(
            "将媒体摄入为**模型输入规格**（224×224 正方形，center-crop 保持纵横比、不变形）：\n"
            "- 视频按所选抽帧模式拆帧到 `frames/video/{名}/`（每帧 center-crop 到 224×224）\n"
            "- 图片摄入为 `frames/image/{名}.jpg`（center-crop 到 224×224；`IMAGE_SIZE=0` 可回退原样复制）\n"
            "- 输出已是模型输入尺寸，喂模型时跳过 resize（省 CPU）；标注图相应变小、居中裁切"
        )
        with gr.Row():
            video_file = gr.File(
                label="媒体文件（上传单个视频或图片）",
                file_count="single",
                file_types=[
                    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv",
                    ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff",
                ],
            )
            folder_path = gr.Textbox(label="媒体文件夹路径（可选）", placeholder="例如 D:/videos")
        video_list_text_extract = gr.Textbox(
            label="媒体路径列表（可选，每行一个视频或图片文件路径）",
            lines=8,
            placeholder="D:/videos/a.mp4\nD:/pics/b.jpg\n...",
        )
        sampling_g, fps_g, min_g, max_g, scene_g, black_g, white_g = sampling_accordion()
        recursive_cb = gr.Checkbox(value=True, label="文件夹输入时递归扫描所有子目录")
        btn_extract = gr.Button("开始拆帧/摄入", variant="primary")
        out_msg = gr.Textbox(label="处理结果", lines=4, interactive=False)
        out_paths = gr.Textbox(label="输出条目路径", lines=3, interactive=False)

        btn_extract.click(
            do_extract,
            inputs=[video_file, folder_path, video_list_text_extract, sampling_g, fps_g, min_g, max_g, scene_g, black_g, white_g, recursive_cb],
            outputs=[out_msg, out_paths],
        )

        with gr.Row():
            clear_confirm = gr.Checkbox(value=False, label="我确认要清空 frames/ 下的全部媒体条目（video/ 与 image/，含 manifest）")
            btn_clear = gr.Button("清空全部记录", variant="stop")
        clear_out = gr.Textbox(label="清空结果", lines=2, interactive=False)
        btn_clear.click(do_clear_frames, inputs=[clear_confirm], outputs=[clear_out])
