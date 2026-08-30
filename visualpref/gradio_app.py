"""Gradio UI 入口：组合六个 Tab（拆帧 / 标注 / 推理 / 批量推理 / 工具 / 训练）。

各 Tab 的实现拆分在 ``visualpref/ui/`` 包内（按职责一模块一 Tab），本模块只做组合，
避免单文件过大。支持视频与图片统一工作流（图片按单文件一等公民处理）。

流程：拆帧/摄入 → 逐项看图点「喜欢/不喜欢」→ 训练 → 推理。
推理/批量推理/训练 Tab 不维护会话状态；所有超参数从 Checkpoint/参数读取，禁止硬编码。
标注 Tab 的队列/进度属于交互会话状态（gr.State + 磁盘持久化）。
训练/批量推理在后台线程执行，经 ``gr.Timer`` 轮询全局状态刷新，不阻塞 UI。
"""

from __future__ import annotations

import gradio as gr

from .ui.batch_tab import build_batch_tab
from .ui.extract_tab import build_extract_tab
from .ui.infer_tab import build_infer_tab
from .ui.label_tab import build_label_tab
from .ui.tools_tab import build_tools_tab
from .ui.train_tab import build_train_tab


def build_ui():
    with gr.Blocks(title="个人媒体喜好二分类器") as demo:
        gr.Markdown(
            "# 🎬 个人媒体喜好二分类器\n"
            "**冻结 DINOv3 骨干 + Masked Attention Pooling + 轻量 MLP 分类头**。\n"
            "支持**视频 🎬 与图片 🖼**（图片按单文件一等公民处理，与视频共用同一模型/工作流）。\n"
            "流程：拆帧/摄入 → 逐项看图点「喜欢/不喜欢」→ 训练 → 推理。"
        )

        build_extract_tab()
        build_label_tab()
        build_infer_tab()
        build_batch_tab()
        build_tools_tab()
        build_train_tab()

    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, **kwargs):
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, **kwargs)


if __name__ == "__main__":
    launch()
