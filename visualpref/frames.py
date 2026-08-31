"""3.1 视频预处理与关键帧采样（编排层）。

抽帧策略（``sampling`` 参数）：
- ``uniform``（默认）：**时长自适应均匀时间抽样**。帧数预算
  ``n = clip(round(duration × fps_target), min_frames, max_frames)``，
  短视频少抽、长视频多抽、时间尽量平均，符合"整体观感"类偏好任务。
- ``scene``：ffmpeg 场景变化检测（``select=gt(scene,THRESH)``），按内容突变抽帧；
  若未选出帧则回退到均匀抽样。
- ``keyframe``：``-skip_frame nokey`` 只解 I 帧（快但粗糙），帧不足时回退均匀抽样。

具体抽样策略在 ``sampling.py``，ffmpeg 命令构建/解码在 ``ffmpeg.py``，本模块只负责
编排：选策略 -> 纯黑白过滤 -> 零填充重新编号落盘。公共 API 为 ``extract_frames``
与 ``extract_from_input``，供批量/标注/Gradio 复用。

输出契约::

    frames/{sanitized_video_name}/
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import config
from .ffmpeg import probe_duration
from .items import MediaItem
from .manifest import FramesNamer
from .paths import frame_filename, iter_media_files
from .sampling import (
    adaptive_frame_budget,
    is_black_or_white,
    keyframe_sample,
    scene_extract,
    uniform_sample,
    uniform_sample_list,
)


def extract_frames(
    video: Path,
    out_dir: Path,
    sampling: str = config.DEFAULT_SAMPLING,
    scene_threshold: float = config.DEFAULT_SCENE_THRESHOLD,
    fps_target: float = config.FPS_TARGET,
    min_frames: int = config.MIN_FRAMES,
    max_frames: int = config.DEFAULT_MAX_FRAMES,
    black_threshold: int = config.BLACK_FRAME_MEAN,
    white_threshold: int = config.WHITE_FRAME_MEAN,
    size: int = config.EXTRACT_MAX_WIDTH,
    hwaccel: str | None = config.EXTRACT_HWACCEL,
) -> Path:
    """对单个视频拆帧并写入 ``out_dir``，返回 out_dir。

    流程：按抽样策略取帧 -> 剔除纯黑/纯白 -> 零填充重新编号。
    过滤后不足 ``min_frames`` 时回退保留原始抽取结果，避免空目录。

    - ``size``：输出正方形边长（center-crop 成 size×size），默认与模型输入
      一致（224），喂模型时 processor 的 resize 退化为恒等（0=不缩放）。
    - ``hwaccel``：可选硬件解码（如 ``"cuda"``），长/高分辨率视频可降 CPU。
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"视频不存在: {video}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清空目录内旧帧，避免重拆帧时残留文件导致计数/时序错误
    for old in out_dir.glob(f"*{config.FRAME_EXT}"):
        old.unlink()

    # 时长只探测一次，供预算计算与均匀采样复用（避免同一视频跑两次 ffprobe）
    duration = probe_duration(video)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if sampling == "scene":
            count = scene_extract(video, tmp_dir, scene_threshold, size=size, hwaccel=hwaccel)
            if count < 1:
                budget = adaptive_frame_budget(
                    video, fps_target, min_frames, max_frames, duration=duration
                )
                uniform_sample(video, tmp_dir, budget, size=size, hwaccel=hwaccel, duration=duration)
        elif sampling == "keyframe":
            count = keyframe_sample(video, tmp_dir, size=size, hwaccel=hwaccel)
            if count < min_frames:
                # 关键帧过少则回退均匀采样，保证帧数足够
                budget = adaptive_frame_budget(
                    video, fps_target, min_frames, max_frames, duration=duration
                )
                uniform_sample(video, tmp_dir, budget, size=size, hwaccel=hwaccel, duration=duration)
        else:  # uniform（默认）
            budget = adaptive_frame_budget(video, fps_target, min_frames, max_frames, duration=duration)
            uniform_sample(video, tmp_dir, budget, size=size, hwaccel=hwaccel, duration=duration)

        raw_frames = sorted(tmp_dir.glob("cap_*.jpg"))
        kept = [p for p in raw_frames if not is_black_or_white(p, black_threshold, white_threshold)]

        # 过滤后过少则回退保留原始帧，避免空/过少结果（防御）
        if len(kept) < min_frames and raw_frames:
            kept = raw_frames

        # 均匀抽到最多 max_frames 帧（不按时间，按序号均分），避免只取开头
        kept = uniform_sample_list(kept, max_frames)
        for idx, src in enumerate(kept, start=1):
            dst = out_dir / frame_filename(idx)
            shutil.copy2(src, dst)

    return out_dir


def ingest_image(image: Path, target: Path, size: int = config.IMAGE_MAX_WIDTH) -> Path:
    """把图片摄入为 ``frames/image/`` 下的单文件条目，返回其路径。

    图片在模型层等价于 T=1 的视频：单文件条目即可让整条下游链路
    （清洗/标注/特征缓存/池化/训练/推理）原样工作（``MediaItem.frame_paths``
    对文件条目返回单元素列表）。

    - 默认 ``size=config.IMAGE_MAX_WIDTH``（与模型输入一致 224）：center-crop 成
      ``size×size`` 正方形（保持纵横比、中心裁切）并重编码为 JPEG，与视频帧/模型
      预处理语义一致（喂模型时 processor 的 resize 退化为恒等）。条目路径由
      ``FramesNamer`` 按同一规格分配为 ``image/{名}.jpg``，摄入输出与 manifest 严格一致。
    - ``size=0`` 时**原样复制**：保留原始分辨率、像素与扩展名（零质量损失，
      仅显式要求时使用）。
    """
    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(f"图片不存在: {image}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if size and size > 0:
        dst = target.with_suffix(".jpg")
        with Image.open(image) as im:
            im = im.convert("RGB")
            # center-crop 到正方形（短边），再缩放到 size×size —— 与官方
            # Resize+CenterCrop 语义一致（保持纵横比、中心裁切，不变形）
            w, h = im.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            if side != size:
                im = im.resize((size, size), Image.LANCZOS)
            im.save(dst, "JPEG", quality=95)
        return dst
    # 原样复制（字节级一致），保留原扩展名，零重编码质量损失
    shutil.copy2(image, target)
    return target


def _make_ingest_handlers(
    *,
    sampling: str,
    scene_threshold: float,
    fps_target: float,
    min_frames: int,
    max_frames: int,
    black_threshold: int,
    white_threshold: int,
    size: int,
    image_size: int,
    hwaccel: str | None,
) -> dict[str, Callable[[Path, MediaItem], None]]:
    """构造 "媒体类型 -> 摄入处理器" 的分派表。

    处理器签名 ``(media: Path, item: MediaItem) -> None``：把媒体写入其条目
    （视频=ffmpeg 拆帧 center-crop 到工作区目录，图片=center-crop 为单文件）。
    后续新增媒体类型只需在此注册一个新 kind 的处理器，``extract_from_input``
    主循环不变。
    """

    def _video(media: Path, item: MediaItem) -> None:
        extract_frames(
            media,
            item.path,
            sampling=sampling,
            scene_threshold=scene_threshold,
            fps_target=fps_target,
            min_frames=min_frames,
            max_frames=max_frames,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
            size=size,
            hwaccel=hwaccel,
        )

    def _image(media: Path, item: MediaItem) -> None:
        # 图片：center-crop 为单文件条目（默认 MODEL_INPUT_SIZE=224 存 JPEG；0=字节复制）
        ingest_image(media, item.path, size=image_size)

    return {"video": _video, "image": _image}


def extract_from_input(
    input_path,
    frames_root: Path,
    sampling: str = config.DEFAULT_SAMPLING,
    scene_threshold: float = config.DEFAULT_SCENE_THRESHOLD,
    fps_target: float = config.FPS_TARGET,
    min_frames: int = config.MIN_FRAMES,
    max_frames: int = config.DEFAULT_MAX_FRAMES,
    black_threshold: int = config.BLACK_FRAME_MEAN,
    white_threshold: int = config.WHITE_FRAME_MEAN,
    size: int = config.EXTRACT_MAX_WIDTH,
    image_size: int = config.IMAGE_MAX_WIDTH,
    hwaccel: str | None = config.EXTRACT_HWACCEL,
    workers: int = config.EXTRACT_WORKERS,
    recursive: bool = True,
    progress=None,
) -> list[MediaItem]:
    """对媒体文件（视频或图片）、媒体文件夹或媒体路径列表进行处理。

    - 单个视频文件 -> ffmpeg 拆帧到 frames/video/{sanitized_stem}/
    - 单个图片文件 -> 摄入为单文件条目 frames/image/{sanitized_stem}.jpg
      （``ingest_image``，默认 center-crop 到 ``IMAGE_MAX_WIDTH``=224 存 JPEG）
    - 文件夹 -> 对其下每个媒体文件分别输出到对应子区
      （``recursive=True`` 时递归所有子目录）
    - 路径列表（list[Path]）-> 逐项处理（视频拆帧 / 图片摄入）

    ``workers``：并行处理的文件数（ffmpeg 为子进程，并行可缩短整批耗时；
    0/1=串行）。``progress`` 可选 ``progress((i, total), desc=...)``。

    Returns
    -------
    所有条目列表（``MediaItem``：视频=帧工作区目录，图片=单文件）。
    """
    frames_root = Path(frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)

    if isinstance(input_path, (list, tuple)):
        sources = [Path(p) for p in input_path]
    else:
        input_path = Path(input_path)
        if input_path.is_file():
            sources = [input_path]
        elif input_path.is_dir():
            sources = iter_media_files(input_path, recursive=recursive)
        else:
            raise ValueError(f"输入既不是文件也不是文件夹: {input_path}")

    # 同名媒体去重：分配 + 解析统一由 FramesNamer 处理。
    # 先单线程分配好所有条目（避免并行时同名文件竞态），再并行处理。
    # image_size 同时决定图片条目输出扩展名（缩放 → .jpg，见 FramesNamer）。
    namer = FramesNamer(frames_root, image_size=image_size)
    jobs = [(v, namer.assign(v)) for v in sources]
    total = len(jobs)
    done = [0]
    done_lock = threading.Lock()  # 串行化进度上报，避免并发调用 gr.Progress 产生多个进度条
    handlers = _make_ingest_handlers(
        sampling=sampling,
        scene_threshold=scene_threshold,
        fps_target=fps_target,
        min_frames=min_frames,
        max_frames=max_frames,
        black_threshold=black_threshold,
        white_threshold=white_threshold,
        size=size,
        image_size=image_size,
        hwaccel=hwaccel,
    )

    def _one(job):
        media, item = job
        try:
            handlers[item.kind](media, item)
            result = item
        except Exception as e:
            # 单个坏文件不应中断整批：记录并跳过
            print(f"[warn] 处理失败，跳过 {media.name}: {e}", file=sys.stderr)
            result = None
        with done_lock:
            done[0] += 1
            if progress is not None:
                progress((done[0], total), desc=f"拆帧 {media.name}")
        return result

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out_items = list(ex.map(_one, jobs))
    else:
        out_items = [_one(job) for job in jobs]
    out_items = [d for d in out_items if d is not None]

    # 持久化 manifest，供训练/推理解析 media_path -> 条目
    namer.save()
    return out_items
