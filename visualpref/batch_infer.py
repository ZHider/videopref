"""批量离线推理：对成百上千个媒体文件（视频/图片）抽帧或摄入 -> 提取特征 -> 喜好概率。

设计要点（针对规模）：
- 骨干与 Checkpoint 只加载**一次**，循环处理所有媒体（避免逐文件重载 343MB 骨干）。
- 复用 ``ensure_features`` 的**特征缓存**（带帧签名失效校验），重复运行不重复提取。
- 视频抽帧默认 ``keyframe``（只解 I 帧，快但粗糙）；缺帧自动补抽。
  图片直接摄入为单帧（``ingest_image``），无抽帧开销。
- 输出 CSV（utf-8-sig）：文件名、喜好概率、文件全路径三列。

用法示例::

    python infer_batch.py --media data/media_list.txt --checkpoint checkpoints/model.ckpt \\
        --output data/predictions.csv --sampling keyframe
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from . import config
from .dataset import ensure_features
from .frames import extract_from_input
from .items import MediaItem
from .labeling import parse_media_list
from .manifest import item_key_for, load_manifest, resolve_item
from .paths import is_media, iter_media_files
from .predictor import Predictor


def resolve_media(input_path: Path | str, recursive: bool = True) -> list[Path]:
    """把输入解析为媒体路径列表（视频 + 图片）：.txt(每行一个) / 文件夹 / 单个文件。"""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".txt":
        return parse_media_list(p.read_text(encoding="utf-8"))
    if p.is_file():
        return [p] if is_media(p) else []
    if p.is_dir():
        return iter_media_files(p, recursive=recursive)
    raise ValueError(f"输入既不是文件也不是文件夹: {input_path}")


def run_batch_inference(
    media_paths: list[Path],
    checkpoint_path: Path | str,
    frames_root: Path | str = config.FRAMES_ROOT,
    cache_dir: Path | str = config.FEATURES_CACHE_DIR,
    backbone_dir: Path | str | None = None,
    device=None,
    sampling: str = "keyframe",
    batch_size: int = 64,
    workers: int = config.EXTRACT_WORKERS,
    size: int = config.EXTRACT_SIZE,
    min_frames: int = 4,
    max_frames: int = 32,
    threads: int = 8,
    progress=None,
    show_progress: bool = True,
) -> list[dict]:
    """对媒体列表（视频 + 图片）做批量推理，返回结果列表。

    ``threads``：torch CPU 线程数上限。推理时 GPU 前向与 CPU 预处理（解码/缩放）
    串行，torch 默认用满所有核会在 CPU 侧自旋，限制线程数可显著降低 CPU 占用。
    ``progress``：可选 ``(i, total)`` 回调（供程序化调用）；不传时用 tqdm 显示进度条
    （``show_progress=False`` 可关闭）。

    视频缺帧自动抽帧；图片无既有帧时自动摄入为单帧（保留原始分辨率/质量）。
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

    # 骨干 + Checkpoint + 模型（各只加载一次）
    predictor = Predictor(checkpoint_path, backbone_dir=backbone_dir, device=device)

    # 3) 缺帧媒体自动补抽/摄入（keyframe 快速模式）
    # 预加载一次 manifest 并复用于每个媒体的条目解析，避免逐媒体重复读磁盘；
    # 缺帧判定改为一次全量扫描 frames_root + 集合查表，避免每个媒体都扫目录。
    if progress is not None:
        progress((0, 1), desc="检查已有帧目录…")
    manifest = load_manifest(frames_root)
    has_frames = {it.key for it in MediaItem.scan(frames_root)}
    missing = [
        v for v in media_paths
        if item_key_for(v, frames_root, manifest=manifest) not in has_frames
    ]
    if missing:
        if progress is not None:
            extract_from_input(
                missing, frames_root, sampling=sampling, size=size,
                workers=workers, min_frames=min_frames, max_frames=max_frames,
                progress=progress,
            )
        else:
            bar = tqdm(total=len(missing), desc="抽帧", unit="个", disable=not show_progress)

            def _ext_prog(pair, **kwargs):
                done_val, _ = pair
                bar.update(done_val - bar.n)

            extract_from_input(
                missing, frames_root, sampling=sampling, size=size,
                workers=workers, min_frames=min_frames, max_frames=max_frames,
                progress=_ext_prog,
            )
            bar.close()

    # 4) 逐媒体特征 + 预测
    results: list[dict] = []
    total = len(media_paths)
    bar = None if progress is not None else tqdm(total=total, desc="推理", unit="个", disable=not show_progress)
    for i, media in enumerate(media_paths):
        if progress is not None:
            progress((i + 1, total), desc=f"推理 {media.name}")
        try:
            item = resolve_item(media, frames_root)
            feats = ensure_features(
                item, cache_dir, predictor.backbone, predictor.processor, predictor.device, batch_size
            )
            if feats.shape[0] == 0:
                results.append(
                    {
                        "media_path": str(media),
                        "item_path": item.path.name,
                        "num_frames": 0,
                        "like_probability": None,
                        "predicted_label": None,
                        "error": "无帧",
                    }
                )
                if bar is not None:
                    bar.update(1)
                continue
            prob = predictor.predict_feats(feats)
            results.append(
                {
                    "media_path": str(media),
                    "item_path": item.path.name,
                    "num_frames": int(feats.shape[0]),
                    "like_probability": round(prob, 6),
                    "predicted_label": 1 if prob >= 0.5 else 0,
                    "error": None,
                }
            )
        except Exception as e:
            # 单个媒体失败不中断整批
            results.append(
                {
                    "media_path": str(media),
                    "item_path": "",
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

    概率为空表示该媒体处理失败（无帧）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "喜好概率", "文件全路径"])
        for r in results:
            prob = r.get("like_probability")
            prob_str = "" if prob is None else ("%.6f" % prob)
            w.writerow([Path(r["media_path"]).name, prob_str, r["media_path"]])
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="批量抽帧 + 喜好推理")
    p.add_argument("--media", "--videos", dest="media", required=True, help="媒体路径列表(.txt 每行一个) 或 文件夹（视频/图片）")
    p.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    p.add_argument("--output", default=str(config.DATA_DIR / "predictions.csv"), help="输出 CSV 路径")
    p.add_argument("--cache-dir", default=str(config.FEATURES_CACHE_DIR), help="特征缓存目录")
    p.add_argument("--backbone-dir", default=str(config.DEFAULT_BACKBONE_DIR), help="骨干权重目录")
    p.add_argument("--sampling", default="keyframe", choices=["uniform", "keyframe", "scene"], help="抽帧方式")
    p.add_argument("--batch-size", type=int, default=16, help="特征提取 batch（配合流水线预取：batch 略小于单视频帧数时，CPU 预处理与 GPU 前向重叠最充分）")
    p.add_argument("--workers", type=int, default=config.EXTRACT_WORKERS, help="并行抽帧视频数")
    p.add_argument("--threads", type=int, default=8, help="torch CPU 线程数上限（降低 CPU 占用；0=不限制）")
    p.add_argument("--size", type=int, default=config.EXTRACT_SIZE, help="抽帧/摄入输出正方形边长（center-crop，默认与模型输入一致）")
    p.add_argument("--min-frames", type=int, default=4, help="单视频最少帧数")
    p.add_argument("--max-frames", type=int, default=32, help="单视频最多帧数（超出则均匀抽取）")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 个（测试用，0=全部）")
    p.add_argument("--device", default=None, help="cuda / cpu")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    media = resolve_media(args.media)
    if args.limit and args.limit > 0:
        media = media[: args.limit]
    print(f"[info] 待处理媒体: {len(media)} 个")
    if not media:
        raise SystemExit("没有可处理的媒体。")
    results = run_batch_inference(
        media,
        args.checkpoint,
        frames_root=config.FRAMES_ROOT,
        cache_dir=args.cache_dir,
        backbone_dir=args.backbone_dir,
        device=args.device,
        sampling=args.sampling,
        batch_size=args.batch_size,
        workers=args.workers,
        size=args.size,
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
            print(f"  - {r['media_path']}: {r['error']}")
        if len(failed) > 10:
            print(f"  ... 其余 {len(failed) - 10} 个略（见 CSV 中概率留空的行）")


if __name__ == "__main__":
    main()
