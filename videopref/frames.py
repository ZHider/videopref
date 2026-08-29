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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config
from .ffmpeg import probe_duration
from .manifest import FramesNamer
from .paths import frame_filename, iter_video_files
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
    max_width: int = config.EXTRACT_MAX_WIDTH,
    hwaccel: str | None = config.EXTRACT_HWACCEL,
) -> Path:
    """对单个视频拆帧并写入 ``out_dir``，返回 out_dir。

    流程：按抽样策略取帧 -> 剔除纯黑/纯白 -> 零填充重新编号。
    过滤后不足 ``min_frames`` 时回退保留原始抽取结果，避免空目录。

    - ``max_width``：输出宽度上限（0=不缩放），默认 640，降低 CPU/磁盘。
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
            count = scene_extract(video, tmp_dir, scene_threshold, max_width=max_width, hwaccel=hwaccel)
            if count < 1:
                budget = adaptive_frame_budget(
                    video, fps_target, min_frames, max_frames, duration=duration
                )
                uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel, duration=duration)
        elif sampling == "keyframe":
            count = keyframe_sample(video, tmp_dir, max_width=max_width, hwaccel=hwaccel)
            if count < min_frames:
                # 关键帧过少则回退均匀采样，保证帧数足够
                budget = adaptive_frame_budget(
                    video, fps_target, min_frames, max_frames, duration=duration
                )
                uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel, duration=duration)
        else:  # uniform（默认）
            budget = adaptive_frame_budget(video, fps_target, min_frames, max_frames, duration=duration)
            uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel, duration=duration)

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
    max_width: int = config.EXTRACT_MAX_WIDTH,
    hwaccel: str | None = config.EXTRACT_HWACCEL,
    workers: int = config.EXTRACT_WORKERS,
    recursive: bool = True,
    progress=None,
) -> list[Path]:
    """对视频文件、视频文件夹或视频路径列表进行拆帧。

    - 单个视频文件 -> frames/{sanitized_stem}/
    - 文件夹 -> 对其中每个视频文件分别输出到其子文件夹（``recursive=True`` 时递归所有子目录）
    - 路径列表（list[Path]）-> 逐视频输出

    ``workers``：并行拆帧的视频数（ffmpeg 为子进程，并行可缩短整批耗时；
    0/1=串行）。``progress`` 可选 ``progress((i, total), desc=...)``。

    Returns
    -------
    所有输出文件夹列表。
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
            sources = iter_video_files(input_path, recursive=recursive)
        else:
            raise ValueError(f"输入既不是文件也不是文件夹: {input_path}")

    # 同名视频去重：分配 + 解析统一由 paths.FramesNamer 处理。
    # 先单线程分配好所有目录名（避免并行时同名视频的竞态），再并行拆帧。
    namer = FramesNamer(frames_root)
    jobs = [(v, namer.assign(v)) for v in sources]
    total = len(jobs)
    done = [0]

    def _one(job):
        v, out_dir = job
        try:
            extract_frames(
                v,
                out_dir,
                sampling=sampling,
                scene_threshold=scene_threshold,
                fps_target=fps_target,
                min_frames=min_frames,
                max_frames=max_frames,
                black_threshold=black_threshold,
                white_threshold=white_threshold,
                max_width=max_width,
                hwaccel=hwaccel,
            )
            result = out_dir
        except Exception as e:
            # 单个坏文件不应中断整批：记录并跳过
            print(f"[warn] 拆帧失败，跳过 {v.name}: {e}", file=sys.stderr)
            result = None
        done[0] += 1
        if progress is not None:
            progress((done[0], total), desc=f"拆帧 {v.name}")
        return result

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out_dirs = list(ex.map(_one, jobs))
    else:
        out_dirs = [_one(job) for job in jobs]
    out_dirs = [d for d in out_dirs if d is not None]

    # 持久化 manifest，供训练/推理解析 video_path -> 帧目录
    namer.save()
    return out_dirs
