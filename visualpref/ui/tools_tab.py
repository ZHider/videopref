"""🧰 工具 Tab：随机选取媒体 + 按 CSV 移动低分文件。"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import gradio as gr

from .common import file_value_to_path


def do_tool_random_pick(src_dir, count, seed, recursive):
    """从目录随机选取若干媒体，返回 (路径列表文本, 统计)。"""
    from random_pick_videos import find_videos, pick

    if not src_dir or not src_dir.strip():
        return "", "请填写源目录路径。"
    root = Path(src_dir.strip())
    if not root.is_dir():
        return "", f"目录不存在: {root}"
    videos = find_videos(root, recursive=bool(recursive))
    if not videos:
        return "", f"目录下未找到媒体文件（共扫描 {len(videos)}）。"
    picked = pick(videos, int(count), int(seed) if seed not in (None, "") else None)
    text = "\n".join(str(p) for p in picked)
    summary = f"共找到 {len(videos)} 个媒体，随机选取 {len(picked)} 个。"
    return text, summary


def do_tool_move_low_score(csv_file, dest_dir, threshold, dry_run):
    """读 CSV，把分数低于阈值的文件移到 dest_dir，返回 (结果表, 统计)。"""
    from move_low_score_files import move_low_score_files

    src = file_value_to_path(csv_file)
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


def build_tools_tab():
    with gr.Tab("🧰 工具"):
        with gr.Accordion("🎲 随机选取媒体", open=False):
            rp_src = gr.Textbox(label="源目录路径", placeholder="例如 F:/GV")
            with gr.Row():
                rp_count = gr.Slider(1, 500, value=50, step=1, label="选取数量")
                rp_seed = gr.Number(value=None, label="随机种子（可复现，留空=不固定）")
                rp_recursive = gr.Checkbox(value=True, label="递归扫描子目录")
            rp_btn = gr.Button("随机选取", variant="primary")
            rp_summary = gr.Textbox(label="统计", lines=1, interactive=False)
            rp_out = gr.Textbox(label="选中的媒体路径", lines=8, interactive=False)
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
