"""抽帧策略：scene / keyframe / uniform 三种采样 + 帧预算计算 + 黑白帧过滤。

策略函数只负责构造并执行一条 ffmpeg 命令、返回写出的帧数；公共命令骨架统一
由 ``ffmpeg.build_ffmpeg_cmd`` 提供。``extract_frames``（frames.py）负责编排：
选策略 -> 过滤 -> 编号落盘。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import config
from .ffmpeg import build_ffmpeg_cmd, ffmpeg_path, probe_duration, run_ffmpeg, scale_filter


def uniform_sample_list(items: list, n: int) -> list:
    """从有序列表均匀抽出至多 n 个（不按时间，按序号均分）。n<=0 或 len<=n 时原样返回。"""
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[0]]
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    out, seen = [], set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def adaptive_frame_budget(
    video: Path | str,
    fps_target: float = config.FPS_TARGET,
    min_frames: int = config.MIN_FRAMES,
    max_frames: int = config.DEFAULT_MAX_FRAMES,
    duration: float | None = None,
) -> int:
    """按时长计算抽帧预算：``n = clip(round(dur × fps_target), min, max)``。

    ``duration`` 可由调用方预先探测一次并传入，避免对同一视频重复跑 ffprobe。
    """
    if duration is None:
        duration = probe_duration(video)
    if duration and duration > 0:
        n = int(round(duration * fps_target))
    else:
        n = (min_frames + max_frames) // 2
    return int(max(min_frames, min(max_frames, n)))


def scene_extract(
    video: Path | str,
    out_dir: Path,
    scene_threshold: float,
    max_width: int = 0,
    hwaccel: str | None = None,
) -> int:
    """ffmpeg 场景变化检测采样到 out_dir（cap_*.jpg）。返回写出的帧数。"""
    vf = ["select='gt(scene,{})'".format(scene_threshold)]
    s = scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = build_ffmpeg_cmd(
        ffmpeg_path(), video, out_dir / "cap_%06d.jpg", vf_parts=vf, hwaccel=hwaccel, fps_mode=True
    )
    res = run_ffmpeg(cmd)
    if res.returncode != 0:
        produced = len(list(out_dir.glob("cap_*.jpg")))
        # 场景检测未选出任何帧（非故障）：返回 0，交由调用方均匀采样回退
        if produced == 0 and "Nothing was written" in res.stderr:
            return 0
        raise RuntimeError(f"ffmpeg 场景检测失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def keyframe_sample(
    video: Path | str,
    out_dir: Path,
    max_width: int = 0,
    hwaccel: str | None = None,
) -> int:
    """关键帧抽帧：``-skip_frame nokey`` 只解码 I 帧，跳过 P/B 帧。

    解码量从"整段视频"骤降到"少数关键帧"，速度大幅提升、CPU 骤降。
    帧间隔依赖编码器 GOP，不完全均匀，属"粗糙但快"模式。
    """
    vf = []
    s = scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = build_ffmpeg_cmd(
        ffmpeg_path(), video, out_dir / "cap_%06d.jpg", vf_parts=vf, hwaccel=hwaccel,
        skip_frame="nokey", fps_mode=True,
    )
    res = run_ffmpeg(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 关键帧采样失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def uniform_sample(
    video: Path | str,
    out_dir: Path,
    n_frames: int,
    max_width: int = 0,
    hwaccel: str | None = None,
    duration: float | None = None,
) -> int:
    """均匀时间抽样：把视频时间轴均匀切成约 n_frames 帧。"""
    if duration is None:
        duration = probe_duration(video)
    if duration and duration > 0:
        fps = max(0.05, n_frames / duration)
    else:
        # 时长未知（如无法 ffprobe）：按名义 30s 估算，结果仍受后续上限约束
        fps = max(0.05, n_frames / 30.0)
    vf = [f"fps={fps:.6f}"]
    s = scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = build_ffmpeg_cmd(
        ffmpeg_path(), video, out_dir / "cap_%06d.jpg", vf_parts=vf, hwaccel=hwaccel, fps_mode=False
    )
    res = run_ffmpeg(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 均匀采样失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def is_black_or_white(frame_path: Path, black_thr: int, white_thr: int) -> bool:
    """基于灰度均值判定纯黑/纯白（打开后即关闭，避免文件句柄泄漏）。"""
    with Image.open(frame_path) as im:
        gray = im.convert("L")
        hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return True
    mean = sum(i * n for i, n in enumerate(hist)) / total
    return mean <= black_thr or mean >= white_thr
