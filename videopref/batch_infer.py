"""批量离线推理：对成百上千个视频抽帧 -> 提取特征 -> 喜好概率。

设计要点（针对规模）：
- 骨干与 Checkpoint 只加载**一次**，循环处理所有视频（避免逐视频重载 343MB 骨干）。
- 复用 ``ensure_video_features`` 的**特征缓存**（带帧签名失效校验），重复运行不重复提取。
- 抽帧默认 ``keyframe``（只解 I 帧，快但粗糙）；缺帧视频自动补抽。
- 输出 CSV（utf-8-sig）：文件名、喜好概率、文件全路径三列。

用法示例::

    python infer_batch.py --videos data/video_list.txt --checkpoint checkpoints/model.ckpt \\
        --output data/predictions.csv --sampling keyframe
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from . import config
from .backbone import load_backbone
from .dataset import ensure_video_features
from .frames import extract_from_input
from .labeling import parse_video_list
from .model import VideoPreferenceModel, load_checkpoint
from .paths import frames_dir_for_video, iter_video_files, list_frame_files


def resolve_videos(input_path: Path | str, recursive: bool = True) -> list[Path]:
    """把输入解析为视频路径列表：.txt(每行一个) / 文件夹(递归扫描) / 单个视频。"""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".txt":
        return parse_video_list(p.read_text(encoding="utf-8"))
    if p.is_file():
        return [p] if p.suffix.lower() in config.VIDEO_EXTENSIONS else []
    if p.is_dir():
        return iter_video_files(p, recursive=recursive)
    raise ValueError(f"输入既不是文件也不是文件夹: {input_path}")


def run_batch_inference(
    videos: list[Path],
    checkpoint_path: Path | str,
    frames_root: Path | str = config.FRAMES_ROOT,
    cache_dir: Path | str = config.FEATURES_CACHE_DIR,
    backbone_dir: Path | str | None = None,
    device=None,
    sampling: str = "keyframe",
    batch_size: int = 64,
    workers: int = config.EXTRACT_WORKERS,
    max_width: int = config.EXTRACT_MAX_WIDTH,
    min_frames: int = 4,
    max_frames: int = 32,
    threads: int = 8,
    progress=None,
    show_progress: bool = True,
) -> list[dict]:
    """对视频列表做批量推理，返回结果列表。

    ``threads``：torch CPU 线程数上限。推理时 GPU 前向与 CPU 预处理（解码/缩放）
    串行，torch 默认用满所有核会在 CPU 侧自旋，限制线程数可显著降低 CPU 占用。
    ``progress``：可选 ``(i, total)`` 回调（供程序化调用）；不传时用 tqdm 显示进度条
    （``show_progress=False`` 可关闭）。
    """
    from tqdm import tqdm

    if threads and threads > 0:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            pass

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_root = Path(frames_root)
    cache_dir = Path(cache_dir)
    if backbone_dir is None:
        backbone_dir = config.DEFAULT_BACKBONE_DIR

    # 1) 骨干（只加载一次）
    backbone, processor, _ = load_backbone(backbone_dir, device=device)

    # 2) Checkpoint + 模型（只加载一次）
    payload = load_checkpoint(checkpoint_path, device=device)
    feature_dim = int(payload["config"].get("feature_dim", config.DEFAULT_FEATURE_DIM))
    model = VideoPreferenceModel(feature_dim=feature_dim).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # 3) 缺帧视频自动补抽（keyframe 快速模式）
    missing = [v for v in videos if not list_frame_files(frames_dir_for_video(v, frames_root))]
    if missing:
        if progress is not None:
            extract_from_input(
                missing, frames_root, sampling=sampling, max_width=max_width,
                workers=workers, min_frames=min_frames, max_frames=max_frames,
                progress=progress,
            )
        else:
            bar = tqdm(total=len(missing), desc="抽帧", unit="个", disable=not show_progress)

            def _ext_prog(pair, **kwargs):
                done_val, _ = pair
                bar.update(done_val - bar.n)

            extract_from_input(
                missing, frames_root, sampling=sampling, max_width=max_width,
                workers=workers, min_frames=min_frames, max_frames=max_frames,
                progress=_ext_prog,
            )
            bar.close()

    # 4) 逐视频特征 + 预测
    results: list[dict] = []
    total = len(videos)
    bar = None if progress is not None else tqdm(total=total, desc="推理", unit="个", disable=not show_progress)
    for i, v in enumerate(videos):
        if progress is not None:
            progress((i + 1, total), desc=f"推理 {v.name}")
        try:
            frames_dir = frames_dir_for_video(v, frames_root)
            feats = ensure_video_features(frames_dir, cache_dir, backbone, processor, device, batch_size)
            if feats.shape[0] == 0:
                results.append(
                    {
                        "video_path": str(v),
                        "frames_dir": frames_dir.name,
                        "num_frames": 0,
                        "like_probability": None,
                        "predicted_label": None,
                        "error": "无帧",
                    }
                )
                if bar is not None:
                    bar.update(1)
                continue
            with torch.no_grad():
                prob = float(model(feats.to(device).unsqueeze(0), mask=None).squeeze().cpu())
            results.append(
                {
                    "video_path": str(v),
                    "frames_dir": frames_dir.name,
                    "num_frames": int(feats.shape[0]),
                    "like_probability": round(prob, 6),
                    "predicted_label": 1 if prob >= 0.5 else 0,
                    "error": None,
                }
            )
        except Exception as e:
            # 单个视频失败不中断整批
            results.append(
                {
                    "video_path": str(v),
                    "frames_dir": frames_dir.name,
                    "num_frames": 0,
                    "like_probability": None,
                    "predicted_label": None,
                    "error": f"{e}",
                }
            )
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return results


def write_results(results: list[dict], output_path: Path | str) -> Path:
    """写 CSV（utf-8-sig）：三列——文件名、喜好概率、文件全路径。

    概率为空表示该视频处理失败（无帧）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "喜好概率", "文件全路径"])
        for r in results:
            prob = r.get("like_probability")
            prob_str = "" if prob is None else ("%.6f" % prob)
            w.writerow([Path(r["video_path"]).name, prob_str, r["video_path"]])
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="批量抽帧 + 喜好推理")
    p.add_argument("--videos", required=True, help="视频路径列表(.txt 每行一个) 或 文件夹")
    p.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    p.add_argument("--output", default=str(config.DATA_DIR / "predictions.csv"), help="输出 CSV 路径")
    p.add_argument("--cache-dir", default=str(config.FEATURES_CACHE_DIR), help="特征缓存目录")
    p.add_argument("--backbone-dir", default=str(config.DEFAULT_BACKBONE_DIR), help="骨干权重目录")
    p.add_argument("--sampling", default="keyframe", choices=["uniform", "keyframe", "scene"], help="抽帧方式")
    p.add_argument("--batch-size", type=int, default=64, help="特征提取 batch（调大可提升 GPU 利用率、减少 CPU 开销）")
    p.add_argument("--workers", type=int, default=config.EXTRACT_WORKERS, help="并行抽帧视频数")
    p.add_argument("--threads", type=int, default=8, help="torch CPU 线程数上限（降低 CPU 占用；0=不限制）")
    p.add_argument("--max-width", type=int, default=config.EXTRACT_MAX_WIDTH, help="抽帧宽度上限")
    p.add_argument("--min-frames", type=int, default=4, help="单视频最少帧数")
    p.add_argument("--max-frames", type=int, default=32, help="单视频最多帧数（超出则均匀抽取）")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 个（测试用，0=全部）")
    p.add_argument("--device", default=None, help="cuda / cpu")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    videos = resolve_videos(args.videos)
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]
    print(f"[info] 待处理视频: {len(videos)} 个")
    if not videos:
        raise SystemExit("没有可处理的视频。")
    results = run_batch_inference(
        videos,
        args.checkpoint,
        frames_root=config.FRAMES_ROOT,
        cache_dir=args.cache_dir,
        backbone_dir=args.backbone_dir,
        device=args.device,
        sampling=args.sampling,
        batch_size=args.batch_size,
        workers=args.workers,
        max_width=args.max_width,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        threads=args.threads,
    )
    out = write_results(results, args.output)
    ok = sum(1 for r in results if r.get("error") is None)
    like = sum(1 for r in results if r.get("predicted_label") == 1)
    print(f"[done] 完成 {ok}/{len(results)}，预测为喜欢 {like} 个 -> {out}")
    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"[warn] 失败 {len(failed)} 个：")
        for r in failed[:10]:
            print(f"  - {r['video_path']}: {r['error']}")
        if len(failed) > 10:
            print(f"  ... 其余 {len(failed) - 10} 个略（见 CSV 中概率留空的行）")


if __name__ == "__main__":
    main()
