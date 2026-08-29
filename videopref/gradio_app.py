"""3.5 Gradio UI（五 Tab：拆帧 / 标注 / 推理 / 批量推理 / 工具）。

- 🎬 拆帧：视频文件 / 文件夹 / **视频路径列表文本**（每行一个路径）
  -> 时长自适应均匀抽样 + 纯黑白过滤 -> 写入 frames/。
- 🏷️ 标注：对 ``frames/`` 下已拆帧的视频逐一看图点「喜欢/不喜欢」
  完成分类；进度持久化到 ``data/label_progress.json``，可续标；导出 labels.json。
- 🔍 推理：Dropdown 动态扫描 frames/ 子文件夹 + Checkpoint -> 输出
  like_probability + JSON。
- ⚡ 批量推理：视频路径列表 + Checkpoint -> 批量抽帧/特征/预测 -> 结果表 + CSV。
- 🧰 工具：随机选取视频（random_pick_videos）+ 按 CSV 移动低分文件
  （move_low_score_files）。

拆帧与标注解耦：标注 Tab 不再负责拆帧，只标注已存在的帧目录。
推理/批量推理 Tab 不维护会话状态；所有超参数从 Checkpoint 读取，禁止硬编码。
标注 Tab 的队列/进度属于交互会话状态（gr.State + 磁盘持久化），
这是标注流程的固有需求，与推理端无状态不冲突。
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from . import config
from .batch_infer import run_batch_inference, write_results
from .frames import extract_from_input
from .inference import infer_frames
from .labeling import (
    advance,
    build_queue_from_frames,
    current_entry,
    load_progress,
    new_state,
    parse_video_list,
    render_state,
    sanitize_state,
    save_labels,
    save_progress,
)


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------
def _list_checkpoints() -> list[str]:
    d = Path(config.CHECKPOINTS_DIR)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.ckpt") if p.is_file())


def _list_frame_folders() -> list[str]:
    root = Path(config.FRAMES_ROOT)
    if not root.is_dir():
        return []
    folders = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(f.suffix.lower() == ".jpg" for f in p.iterdir()):
            folders.append(p.name)
    return folders


def _file_value_to_path(v) -> Path | None:
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


# ---------------------------------------------------------------------------
# 标注：事件（会话状态机逻辑在 labeling.py，这里只做 UI 绑定与持久化）
# ---------------------------------------------------------------------------
def do_resume(state):
    saved = load_progress()
    if saved is None:
        return "没有可续标的历史进度。", "", "", [], state
    state = sanitize_state(saved, config.FRAMES_ROOT)
    save_progress(state)
    name, prog, imgs = render_state(state)
    return f"已恢复上次标注（{len(state['queue'])} 个视频）。", name, prog, imgs, state


def do_scan_all(state):
    """扫描 frames/ 下全部已拆帧视频并开始标注（不重新拆帧）。"""
    queue = build_queue_from_frames(Path(config.FRAMES_ROOT))
    if not queue:
        return "frames/ 下没有含帧的子目录，请先拆帧。", "", "", [], state
    state = {"queue": queue, "idx": 0, "labels": {}, "skipped": []}
    save_progress(state)
    name, prog, imgs = render_state(state)
    return f"已载入 frames/ 下 {len(queue)} 个视频，开始标注。", name, prog, imgs, state


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


# ---------------------------------------------------------------------------
# 拆帧
# ---------------------------------------------------------------------------
def do_extract(video_file, folder_path, video_list_text, sampling, fps_target, min_frames, max_frames, scene_threshold, black_thresh, white_thresh, recursive, progress=gr.Progress()):
    """拆帧。优先级：上传文件 > 文件夹路径 > 路径列表文本。"""
    try:
        input_src = None
        f = _file_value_to_path(video_file)
        if f is not None:
            input_src = f
        elif folder_path.strip():
            input_src = Path(folder_path.strip())
        elif video_list_text.strip():
            paths = parse_video_list(video_list_text)
            if not paths:
                return "路径列表中没有有效视频。", ""
            input_src = paths  # list 形式交给 extract_from_input
        else:
            return "请提供视频文件、文件夹路径或视频路径列表。", ""

        out_dirs = extract_from_input(
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
        lines = [f"处理完成，共 {len(out_dirs)} 个输出文件夹："]
        lines += [f"  {d}" for d in out_dirs]
        return "\n".join(lines), "\n".join(str(d) for d in out_dirs)
    except Exception as e:  # 容错：将异常反馈到 UI
        return f"拆帧失败：{e}", ""


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def do_infer(frames_folder, checkpoint_path, backbone_dir):
    if not frames_folder:
        return None, json.dumps({"error": "请选择帧目录"}, ensure_ascii=False, indent=2)
    if not checkpoint_path:
        return None, json.dumps({"error": "请选择 Checkpoint"}, ensure_ascii=False, indent=2)
    try:
        frames_dir = Path(config.FRAMES_ROOT) / frames_folder
        result = infer_frames(frames_dir, checkpoint_path, model_dir=backbone_dir)
        return (
            result["like_probability"],
            json.dumps(result, ensure_ascii=False, indent=2),
        )
    except Exception as e:
        return None, json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------
def do_batch_infer(
    video_list_text,
    sampling,
    fps_target,
    min_frames,
    max_frames,
    scene_threshold,
    black_thresh,
    white_thresh,
    checkpoint_path,
    backbone_dir,
    batch_size,
    workers,
    threads,
    max_width,
    export_csv,
    progress=gr.Progress(),
):
    """批量推理：解析路径列表 -> run_batch_inference -> 结果表 + CSV。"""
    paths = parse_video_list(video_list_text)
    if not paths:
        return [], "未找到有效视频路径（请每行一个视频文件或文件夹路径，且存在）。"
    if not checkpoint_path:
        return [], "请选择 Checkpoint。"

    try:
        results = run_batch_inference(
            paths,
            checkpoint_path,
            sampling=sampling,
            batch_size=int(batch_size),
            workers=int(workers),
            threads=int(threads),
            max_width=int(max_width),
            min_frames=int(min_frames),
            max_frames=int(max_frames),
            progress=progress,
            show_progress=False,
        )
    except Exception as e:
        return [], f"批量推理失败：{e}"

    rows = [
        [
            Path(r["video_path"]).name,
            "" if r["like_probability"] is None else f"{r['like_probability']:.6f}",
            r["video_path"],
        ]
        for r in results
    ]

    ok = sum(1 for r in results if r.get("error") is None)
    like = sum(1 for r in results if r.get("predicted_label") == 1)
    fail = len(results) - ok

    summary = f"完成 {ok}/{len(results)} 个，预测为喜欢 {like} 个，失败 {fail} 个。"
    if export_csv:
        out = write_results(results, config.DATA_DIR / "predictions.csv")
        summary += f" 已导出 CSV -> {out}"
    return rows, summary


# ---------------------------------------------------------------------------
# 工具：随机选取视频 / 按 CSV 移动低分文件
# ---------------------------------------------------------------------------
def do_tool_random_pick(src_dir, count, seed, recursive):
    """从目录随机选取若干视频，返回 (路径列表文本, 统计)。"""
    from random_pick_videos import find_videos, pick

    if not src_dir or not src_dir.strip():
        return "", "请填写源目录路径。"
    root = Path(src_dir.strip())
    if not root.is_dir():
        return "", f"目录不存在: {root}"
    videos = find_videos(root, recursive=bool(recursive))
    if not videos:
        return "", f"目录下未找到视频文件（共扫描 {len(videos)}）。"
    picked = pick(videos, int(count), int(seed) if seed not in (None, "") else None)
    text = "\n".join(str(p) for p in picked)
    summary = f"共找到 {len(videos)} 个视频，随机选取 {len(picked)} 个。"
    return text, summary


def do_tool_move_low_score(csv_file, dest_dir, threshold, dry_run):
    """读 CSV，把分数低于阈值的文件移到 dest_dir，返回 (结果表, 统计)。"""
    import csv as _csv

    from move_low_score_files import move_low_score_files

    src = _file_value_to_path(csv_file)
    if src is None or not src.is_file():
        return [], "请上传 CSV 文件。"
    if not dest_dir or not dest_dir.strip():
        return [], "请填写目标文件夹。"
    dest = Path(dest_dir.strip())
    try:
        with open(src, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.reader(f))
    except Exception as e:
        return [], f"读取 CSV 失败：{e}"

    results, stats = move_low_score_files(rows, dest, threshold=float(threshold), dry_run=bool(dry_run))

    status_label = {
        "moved": "已移动",
        "ignored_header": "表头",
        "skipped_empty": "空行",
        "skipped_cols": "列数不足",
        "skipped_invalid": "分数无效",
        "skipped_high": "分数达标",
        "skipped_missing": "缺路径",
        "failed_notfound": "文件不存在",
        "failed_move": "移动失败",
    }
    table = [
        [
            r.get("filename", ""),
            "" if r.get("score") is None else f"{r['score']:.4f}",
            status_label.get(r["status"], r["status"]),
            r.get("dest", "") or r.get("detail", ""),
        ]
        for r in results
        if r["status"] != "ignored_header"
    ]
    action = "预览" if dry_run else "移动"
    summary = (
        f"{action} {stats['moved']} 个，跳过 {stats['skipped']} 个，失败 {stats['failed']} 个。"
        f"{'（dry-run 未实际移动）' if dry_run else ''}"
    )
    return table, summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _sampling_accordion():
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


def build_ui():
    with gr.Blocks(title="个人视频喜好二分类器") as demo:
        gr.Markdown(
            "# 🎬 个人视频喜好二分类器\n"
            "**冻结 DINOv3 骨干 + Masked Attention Pooling + 轻量 MLP 分类头**。\n"
            "流程：拆帧 → 逐视频看图点「喜欢/不喜欢」→ 训练 → 推理。"
        )

        # ---------------- Tab 1: 拆帧 ----------------
        with gr.Tab("🎬 拆帧"):
            with gr.Row():
                video_file = gr.File(
                    label="视频文件（上传单个视频）",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"],
                )
                folder_path = gr.Textbox(label="视频文件夹路径（可选）", placeholder="例如 D:/videos")
            video_list_text_extract = gr.Textbox(
                label="视频路径列表（可选，每行一个视频文件路径）",
                lines=8,
                placeholder="D:/videos/a.mp4\nD:/videos/b.mp4\n...",
            )
            sampling_g, fps_g, min_g, max_g, scene_g, black_g, white_g = _sampling_accordion()
            recursive_cb = gr.Checkbox(value=True, label="文件夹输入时递归扫描所有子目录")
            btn_extract = gr.Button("开始拆帧", variant="primary")
            out_msg = gr.Textbox(label="处理结果", lines=4, interactive=False)
            out_paths = gr.Textbox(label="输出文件夹路径", lines=3, interactive=False)

            btn_extract.click(
                do_extract,
                inputs=[video_file, folder_path, video_list_text_extract, sampling_g, fps_g, min_g, max_g, scene_g, black_g, white_g, recursive_cb],
                outputs=[out_msg, out_paths],
            )

        # ---------------- Tab 2: 标注 ----------------
        with gr.Tab("🏷️ 标注"):
            gr.Markdown("先到「🎬 拆帧」Tab 拆帧，再在这里对 `frames/` 下已拆帧的视频逐一看图标注。")
            state = gr.State(new_state())
            with gr.Row():
                btn_scan_all = gr.Button("标注 frames/ 全部已拆帧视频", variant="primary")
                btn_resume = gr.Button("继续上次标注")
            status_out = gr.Textbox(label="状态", lines=2, interactive=False)
            video_name = gr.Markdown("等待开始…")
            progress_lbl = gr.Markdown("")
            gallery = gr.Gallery(label="帧预览（按时间顺序）", columns=5, height=320, object_fit="contain")
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

        # ---------------- Tab 3: 推理 ----------------
        with gr.Tab("🔍 推理"):
            with gr.Row():
                frames_dd = gr.Dropdown(
                    label="帧目录（扫描 frames/）",
                    choices=_list_frame_folders(),
                    interactive=True,
                )
                ckpt_dd = gr.Dropdown(
                    label="Checkpoint（扫描 checkpoints/）",
                    choices=_list_checkpoints(),
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
                return gr.update(choices=_list_frame_folders()), gr.update(choices=_list_checkpoints())

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

        # ---------------- Tab 4: 批量推理 ----------------
        with gr.Tab("⚡ 批量推理"):
            gr.Markdown(
                "对成百上千个视频批量推理：缺帧视频自动拆帧（默认 `keyframe` 快速模式），"
                "骨干与 Checkpoint 只加载一次，坏视频自动跳过。结果以表格展示并可导出 CSV。"
            )
            batch_list_text = gr.Textbox(
                label="视频路径列表（每行一个视频文件或文件夹路径）",
                lines=8,
                placeholder="D:/videos/a.mp4\nF:/GV/b.mp4\nD:/some_folder\n...",
            )
            batch_sampling, batch_fps, batch_min, batch_max, batch_scene, batch_black, batch_white = _sampling_accordion()
            with gr.Row():
                batch_ckpt = gr.Dropdown(label="Checkpoint", choices=_list_checkpoints(), interactive=True)
                batch_backbone = gr.Textbox(label="骨干权重目录（可选）", value=str(config.DEFAULT_BACKBONE_DIR))
            with gr.Row():
                batch_batchsize = gr.Slider(1, 64, value=16, step=1, label="特征提取 batch（调大提升 GPU 利用率）")
                batch_workers = gr.Slider(1, 16, value=float(config.EXTRACT_WORKERS), step=1, label="并行拆帧视频数")
                batch_threads = gr.Slider(1, 16, value=8, step=1, label="torch CPU 线程数上限")
                batch_maxwidth = gr.Slider(0, 1280, value=float(config.EXTRACT_MAX_WIDTH), step=16, label="拆帧宽度上限（0=不缩放）")
            batch_export = gr.Checkbox(value=True, label="同时导出 CSV 到 data/predictions.csv")
            btn_batch = gr.Button("开始批量推理", variant="primary")
            batch_summary = gr.Textbox(label="汇总", lines=2, interactive=False)
            batch_table = gr.Dataframe(
                headers=["文件名", "喜好概率", "文件全路径"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            btn_batch.click(
                do_batch_infer,
                inputs=[
                    batch_list_text, batch_sampling, batch_fps, batch_min, batch_max,
                    batch_scene, batch_black, batch_white, batch_ckpt, batch_backbone,
                    batch_batchsize, batch_workers, batch_threads, batch_maxwidth, batch_export,
                ],
                outputs=[batch_table, batch_summary],
            )

        # ---------------- Tab 5: 工具 ----------------
        with gr.Tab("🧰 工具"):
            with gr.Accordion("🎲 随机选取视频", open=False):
                rp_src = gr.Textbox(label="源目录路径", placeholder="例如 F:/GV")
                with gr.Row():
                    rp_count = gr.Slider(1, 500, value=50, step=1, label="选取数量")
                    rp_seed = gr.Number(value=None, label="随机种子（可复现，留空=不固定）")
                    rp_recursive = gr.Checkbox(value=True, label="递归扫描子目录")
                rp_btn = gr.Button("随机选取", variant="primary")
                rp_summary = gr.Textbox(label="统计", lines=1, interactive=False)
                rp_out = gr.Textbox(label="选中的视频路径", lines=8, interactive=False)
                rp_btn.click(do_tool_random_pick, inputs=[rp_src, rp_count, rp_seed, rp_recursive], outputs=[rp_out, rp_summary])

            with gr.Accordion("📦 按 CSV 移动低分文件", open=False):
                gr.Markdown("CSV 三列：`文件名,喜好分数(0-1.0),文件全路径`。低于阈值的文件会被移动到目标文件夹。")
                with gr.Row():
                    ml_csv = gr.File(label="上传 CSV 文件", file_count="single", file_types=[".csv"])
                    ml_dest = gr.Textbox(label="目标文件夹", placeholder="例如 F:/GV/dislike")
                with gr.Row():
                    ml_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="分数阈值（低于则移动）")
                    ml_dry = gr.Checkbox(value=True, label="dry-run（仅预览，不实际移动）")
                ml_btn = gr.Button("处理", variant="primary")
                ml_summary = gr.Textbox(label="统计", lines=1, interactive=False)
                ml_table = gr.Dataframe(
                    headers=["文件名", "分数", "状态", "目标/说明"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                ml_btn.click(do_tool_move_low_score, inputs=[ml_csv, ml_dest, ml_thresh, ml_dry], outputs=[ml_table, ml_summary])

    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, **kwargs):
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, **kwargs)


if __name__ == "__main__":
    launch()
