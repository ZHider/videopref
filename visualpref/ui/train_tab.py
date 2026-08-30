"""🎓 训练 Tab：后台线程训练 + 全局状态 + Timer 轮询（实时曲线 + 日志）。"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from .. import config
from ..train import run_training
from .common import JobState

_train = JobState(
    {
        "running": False,
        "epoch": 0,
        "total": 0,
        "history": [],
        "logs": [],
        "done": False,
        "error": None,
        "best_path": None,
        "final_path": None,
    }
)


def _train_worker(args) -> None:
    """后台线程执行 run_training，把进度/日志写入全局 _train。"""

    def _prog(epoch, total, metrics):
        _train.update(epoch=epoch, total=total, history=_train["history"] + [dict(metrics)])

    def _log(msg):
        _train.update(logs=_train["logs"] + [str(msg)])

    try:
        result = run_training(args, progress=_prog, log=_log, use_tqdm=False)
        _train.update(
            done=True,
            best_path=str(result["best_path"]),
            final_path=str(result["final_path"]),
        )
    except Exception as e:  # noqa: BLE001 - 反馈到 UI
        _train.update(done=True, error=f"{e}")
    finally:
        _train.update(running=False)


def do_train_start(labels_path, cache_dir, output_dir, epochs, lr, seed, batch_size, threads, val_fraction, augment, backbone_dir):
    """启动后台训练线程。"""
    labels_path = (labels_path or "").strip()
    if not labels_path or not Path(labels_path).is_file():
        return "labels.json 不存在，请填写正确路径。"
    if _train["running"]:
        return "已有训练在进行中，请等待完成。"
    _train.update(
        running=True, epoch=0, total=0, history=[], logs=[], done=False,
        error=None, best_path=None, final_path=None,
    )

    args = SimpleNamespace(
        data=labels_path,
        cache_dir=cache_dir or str(config.FEATURES_CACHE_DIR),
        output_dir=output_dir or str(config.CHECKPOINTS_DIR),
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
        batch_size=int(batch_size),
        threads=int(threads),
        val_fraction=float(val_fraction),
        augment=bool(augment),
        log_dir=None,
        wandb=False,
        device=None,
        backbone_dir=backbone_dir or str(config.DEFAULT_BACKBONE_DIR),
        backbone_id=config.DEFAULT_BACKBONE_ID,
    )
    threading.Thread(target=_train_worker, args=(args,), daemon=True).start()
    return f"已启动训练：{epochs} 个 epoch，输出到 {args.output_dir}。请稍候…"


def _build_train_plot(history):
    """把各 epoch 指标转为 DataFrame，供 ``gr.LinePlot`` 画训练曲线。"""
    import pandas as pd

    if not history:
        return None
    return pd.DataFrame(history)


def train_tick():
    """被 gr.Timer 周期调用，返回 (状态, 日志, 曲线, 完成提示)。"""
    state = _train.snapshot()

    if state["running"]:
        status = f"训练中：epoch {state['epoch']}/{state['total']}"
    elif state["done"]:
        if state["error"]:
            status = f"训练出错：{state['error']}"
        else:
            status = f"训练完成！共 {state['total']} 个 epoch。"
    else:
        status = "未开始（填写参数后点击「开始训练」）。"

    logs_text = "\n".join(state["logs"][-200:])
    fig = _build_train_plot(state["history"])

    done_info = ""
    if state["done"]:
        if state["error"]:
            done_info = f"❌ 训练出错：{state['error']}"
        else:
            done_info = f"✅ 完成。最佳 checkpoint -> {state['best_path']}；最终 checkpoint -> {state['final_path']}"
    return status, logs_text, fig, done_info


def build_train_tab():
    with gr.Tab("🎓 训练"):
        gr.Markdown(
            "在后台线程训练，训练期间可切换其他 Tab。曲线与日志每秒刷新；"
            "参数含义同 CLI：`train.py --help`。"
        )
        tr_labels = gr.Textbox(label="标注文件 labels.json", value=str(config.DATA_DIR / "labels.json"))
        with gr.Row():
            tr_cache = gr.Textbox(label="特征缓存目录", value=str(config.FEATURES_CACHE_DIR))
            tr_output = gr.Textbox(label="Checkpoint 输出目录", value=str(config.CHECKPOINTS_DIR))
        with gr.Row():
            tr_epochs = gr.Number(value=100, precision=0, label="epochs")
            tr_lr = gr.Number(value=1e-3, label="lr")
            tr_seed = gr.Number(value=42, precision=0, label="seed")
            tr_batch = gr.Number(value=16, precision=0, label="batch_size")
            tr_threads = gr.Number(value=8, precision=0, label="torch 线程数")
            tr_valfrac = gr.Number(value=0.2, label="val_fraction")
        tr_augment = gr.Checkbox(value=False, label="训练期数据增强（--augment）")
        tr_backbone = gr.Textbox(label="骨干权重目录", value=str(config.DEFAULT_BACKBONE_DIR))
        tr_start = gr.Button("开始训练", variant="primary")
        tr_status = gr.Textbox(label="状态", lines=2, interactive=False)
        tr_plot = gr.LinePlot(
            x="epoch",
            y=["train_loss", "val_loss", "val_auc"],
            title="训练曲线",
            height=400,
        )
        tr_logs = gr.Textbox(label="日志", lines=12, interactive=False)
        tr_done = gr.Textbox(label="结果", lines=2, interactive=False)

        tr_start.click(
            do_train_start,
            inputs=[tr_labels, tr_cache, tr_output, tr_epochs, tr_lr, tr_seed,
                    tr_batch, tr_threads, tr_valfrac, tr_augment, tr_backbone],
            outputs=[tr_status],
        )
        tr_timer = gr.Timer(1.0)
        tr_timer.tick(train_tick, outputs=[tr_status, tr_logs, tr_plot, tr_done])
