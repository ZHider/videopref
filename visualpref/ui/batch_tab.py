"""⚡ 批量推理 Tab：后台线程 + 全局状态 + Timer 轮询（HTML 进度条）。"""

from __future__ import annotations

import threading
from pathlib import Path

import gradio as gr

from .. import config
from ..batch_infer import run_batch_inference, write_results
from ..labeling import parse_media_list
from .common import JobState, list_checkpoints, progress_html, sampling_accordion

_batch = JobState(
    {
        "running": False,
        "cur": 0,
        "total": 0,
        "desc": "",
        "rows": [],
        "summary": "",
        "done": False,
        "error": None,
    }
)


def _build_batch_rows(results: list[dict]) -> list[list]:
    return [
        [
            Path(r["media_path"]).name,
            "" if r["like_probability"] is None else f"{r['like_probability']:.6f}",
            r["media_path"],
        ]
        for r in results
    ]


def _on_batch_prog(pair, desc=None, **kwargs) -> None:
    """run_batch_inference 的进度回调：写入全局状态（供 Timer 轮询显示）。"""
    i, total = pair
    _batch.update(cur=int(i), total=int(total), desc=desc or "")


def _batch_worker(paths, checkpoint_path, sampling, batch_size, workers, threads,
                  size, min_frames, max_frames, export_csv) -> None:
    """后台线程执行批量推理，把进度/结果写入全局 _batch。"""
    try:
        results = run_batch_inference(
            paths,
            checkpoint_path,
            sampling=sampling,
            batch_size=int(batch_size),
            workers=int(workers),
            threads=int(threads),
            size=int(size),
            min_frames=int(min_frames),
            max_frames=int(max_frames),
            progress=_on_batch_prog,
            show_progress=False,
        )
        ok = sum(1 for r in results if r.get("error") is None)
        like = sum(1 for r in results if r.get("predicted_label") == 1)
        fail = len(results) - ok
        summary = f"完成 {ok}/{len(results)} 个，预测为喜欢 {like} 个，失败 {fail} 个。"
        if export_csv:
            out = write_results(results, config.DATA_DIR / "predictions.csv")
            summary += f" 已导出 CSV -> {out}"
        _batch.update(rows=_build_batch_rows(results), summary=summary, done=True)
    except Exception as e:  # noqa: BLE001 - 反馈到 UI
        _batch.update(done=True, error=f"{e}")
    finally:
        _batch.update(running=False)


def do_batch_infer_start(video_list_text, sampling, fps_target, min_frames, max_frames,
                         scene_threshold, black_thresh, white_thresh, checkpoint_path,
                         backbone_dir, batch_size, workers, threads, size, export_csv):
    """启动后台批量推理（不阻塞 UI，进度经 Timer 轮询更新）。"""
    paths = parse_media_list(video_list_text)
    if not paths:
        return "未找到有效媒体路径（请每行一个视频或图片文件/文件夹路径，且存在）。"
    if not checkpoint_path:
        return "请选择 Checkpoint。"
    if _batch["running"]:
        return "已有批量推理在进行中，请等待完成。"
    _batch.update(running=True, cur=0, total=0, desc="", rows=[], summary="", done=False, error=None)
    threading.Thread(
        target=_batch_worker,
        args=(paths, checkpoint_path, sampling, batch_size, workers, threads,
              size, min_frames, max_frames, export_csv),
        daemon=True,
    ).start()
    return f"已启动批量推理：{len(paths)} 个媒体。请稍候…"


def batch_tick():
    """被 gr.Timer 轮询：返回 (进度/汇总文本, 结果表, 进度条 HTML)。

    gr.Timer 串行触发，进度条由本函数每次重绘为 Tab 内 HTML 组件，
    稳定可见且不会像 gr.Progress 事件进度条那样重复或缺失。
    """
    st = _batch.snapshot()
    if st["running"]:
        html = progress_html(st["cur"], st["total"], st["desc"]) if st["total"] else progress_html(0, 1, "准备中")
        return "处理中…", [], html  # 实时进度由进度条显示，汇总框保持简短
    if st["done"]:
        if st["error"]:
            return f"❌ 批量推理出错：{st['error']}", [], ""
        return st["summary"], st["rows"], ""
    return "", [], ""


def build_batch_tab():
    with gr.Tab("⚡ 批量推理"):
        gr.Markdown(
            "对成百上千个媒体（视频/图片）批量推理：缺帧视频自动拆帧（默认 `keyframe` 快速模式），"
            "图片直接摄入为单文件；缺帧/未摄入的媒体统一 center-crop 到 224×224（模型输入规格）。"
            "骨干与 Checkpoint 只加载一次，坏文件自动跳过。结果以表格展示并可导出 CSV。"
        )
        batch_list_text = gr.Textbox(
            label="媒体路径列表（每行一个视频或图片文件/文件夹路径）",
            lines=8,
            placeholder="D:/videos/a.mp4\nF:/GV/b.mp4\nD:/pics/c.jpg\nD:/some_folder\n...",
        )
        batch_sampling, batch_fps, batch_min, batch_max, batch_scene, batch_black, batch_white = sampling_accordion()
        with gr.Row():
            batch_ckpt = gr.Dropdown(label="Checkpoint", choices=list_checkpoints(), interactive=True)
            batch_backbone = gr.Textbox(label="骨干权重目录（可选）", value=str(config.DEFAULT_BACKBONE_DIR))
        with gr.Row():
            batch_batchsize = gr.Slider(1, 64, value=16, step=1, label="特征提取 batch（调大提升 GPU 利用率）")
            batch_workers = gr.Slider(1, 16, value=float(config.EXTRACT_WORKERS), step=1, label="并行拆帧视频数")
            batch_threads = gr.Slider(1, 16, value=8, step=1, label="torch CPU 线程数上限")
            batch_size_ctl = gr.Slider(64, 512, value=float(config.EXTRACT_SIZE), step=32, label="抽帧/摄入输出正方形边长（center-crop；默认与模型输入一致）")
        batch_export = gr.Checkbox(value=True, label="同时导出 CSV 到 data/predictions.csv")
        btn_batch = gr.Button("开始批量推理", variant="primary")
        batch_progress = gr.HTML()  # Tab 内可见的进度条（Timer 轮询更新）
        batch_summary = gr.Textbox(label="汇总", lines=2, interactive=False)
        batch_table = gr.Dataframe(
            headers=["文件名", "喜好概率", "文件全路径"],
            datatype=["str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        btn_batch.click(
            do_batch_infer_start,
            inputs=[
                batch_list_text, batch_sampling, batch_fps, batch_min, batch_max,
                batch_scene, batch_black, batch_white, batch_ckpt, batch_backbone,
                batch_batchsize, batch_workers, batch_threads, batch_size_ctl, batch_export,
            ],
            outputs=[batch_summary],
        )
        batch_timer = gr.Timer(1.0)
        batch_timer.tick(batch_tick, outputs=[batch_summary, batch_table, batch_progress])
