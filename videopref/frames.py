"""3.1 视频预处理与关键帧采样。

抽帧策略（``sampling`` 参数）：
- ``uniform``（默认）：**时长自适应均匀时间抽样**。帧数预算
  ``n = clip(round(duration × fps_target), min_frames, max_frames)``，
  短视频少抽、长视频多抽、时间尽量平均，符合"整体观感"类偏好任务。
- ``scene``：ffmpeg 场景变化检测（``select=gt(scene,THRESH)``），按内容突变抽帧；
  若未选出帧则回退到均匀抽样。

两个模式都自动剔除纯黑/纯白帧（灰度均值阈值可配置），
输出按零填充编号命名（``0001.jpg`` ...），保证文件系统排序不丢失时序。

输出契约::

    frames/{sanitized_video_name}/
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from . import config
from .paths import FramesNamer, frame_filename, iter_video_files


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# ffmpeg 底层调用
# ---------------------------------------------------------------------------
# Windows 默认用 GBK 解码子进程输出，而 ffmpeg 输出含非 GBK 字节会抛 UnicodeDecodeError。
# 统一用 utf-8 + errors="replace"，保证任何情况下都不会因解码崩溃。
_SUBPROCESS_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _probe_duration(video: Path) -> float | None:
    """用 ffprobe 探测时长（秒）；失败返回 None。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            **_SUBPROCESS_TEXT,
            timeout=120,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except Exception:
        return None
    return None


def _run_ffmpeg(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """执行 ffmpeg；兼容新旧版本的 fps_mode/vsync 选项差异。"""
    # 定位 -vsync 参数（若有），转成 -fps_mode（新版本）；失败则回退旧写法
    if "-vsync" in cmd:
        idx = cmd.index("-vsync")
        mode = cmd[idx + 1]
        new_cmd = cmd[:idx] + ["-fps_mode", mode] + cmd[idx + 2 :]
        res = subprocess.run(new_cmd, capture_output=True, timeout=timeout, **_SUBPROCESS_TEXT)
        if res.returncode == 0 or "Unrecognized option 'fps_mode'" not in (res.stderr or ""):
            return res
        # 旧版本不支持 fps_mode，回退 -vsync
        return subprocess.run(cmd, capture_output=True, timeout=timeout, **_SUBPROCESS_TEXT)
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **_SUBPROCESS_TEXT)


def _scale_filter(max_width: int) -> str:
    """输出分辨率上限（640 等）：对分类无损，显著降低 JPEG 编码 CPU 与磁盘。0=不缩放。"""
    if max_width and max_width > 0:
        return f"scale='min({max_width},iw)':-2"
    return ""


def _uniform_sample_list(items: list, n: int) -> list:
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


def _scene_extract(
    video: Path,
    out_dir: Path,
    scene_threshold: float,
    max_width: int = 0,
    hwaccel: str | None = None,
) -> int:
    """ffmpeg 场景变化检测采样到 out_dir（cap_*.jpg）。返回写出的帧数。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH。")
    vf = ["select='gt(scene,{})'".format(scene_threshold)]
    s = _scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = [ffmpeg, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(video), "-vf", ",".join(vf), "-vsync", "vfr", "-q:v", "2", str(out_dir / "cap_%06d.jpg")]
    res = _run_ffmpeg(cmd)
    if res.returncode != 0:
        produced = len(list(out_dir.glob("cap_*.jpg")))
        # 场景检测未选出任何帧（非故障）：返回 0，交由调用方均匀采样回退
        if produced == 0 and "Nothing was written" in res.stderr:
            return 0
        raise RuntimeError(f"ffmpeg 场景检测失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def _keyframe_sample(
    video: Path,
    out_dir: Path,
    max_width: int = 0,
    hwaccel: str | None = None,
) -> int:
    """关键帧抽帧：``-skip_frame nokey`` 只解码 I 帧，跳过 P/B 帧。

    解码量从"整段视频"骤降到"少数关键帧"，速度大幅提升、CPU 骤降。
    帧间隔依赖编码器 GOP，不完全均匀，属"粗糙但快"模式。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH。")
    vf = []
    s = _scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = [ffmpeg, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-skip_frame", "nokey", "-i", str(video)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-vsync", "vfr", "-q:v", "2", str(out_dir / "cap_%06d.jpg")]
    res = _run_ffmpeg(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 关键帧采样失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def _uniform_sample(
    video: Path,
    out_dir: Path,
    n_frames: int,
    max_width: int = 0,
    hwaccel: str | None = None,
) -> int:
    """均匀时间抽样：把视频时间轴均匀切成约 n_frames 帧。"""
    ffmpeg = shutil.which("ffmpeg")
    duration = _probe_duration(video)
    if duration and duration > 0:
        fps = max(0.05, n_frames / duration)
    else:
        # 时长未知（如无法 ffprobe）：按名义 30s 估算，结果仍受后续上限约束
        fps = max(0.05, n_frames / 30.0)
    vf = [f"fps={fps:.6f}"]
    s = _scale_filter(max_width)
    if s:
        vf.append(s)
    cmd = [ffmpeg, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(video), "-vf", ",".join(vf), "-q:v", "2", str(out_dir / "cap_%06d.jpg")]
    res = _run_ffmpeg(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 均匀采样失败:\n{(res.stderr or '')[-2000:]}")
    return len(list(out_dir.glob("cap_*.jpg")))


def adaptive_frame_budget(
    video: Path,
    fps_target: float = config.FPS_TARGET,
    min_frames: int = config.MIN_FRAMES,
    max_frames: int = config.DEFAULT_MAX_FRAMES,
) -> int:
    """按时长计算抽帧预算：``n = clip(round(dur × fps_target), min, max)``。"""
    duration = _probe_duration(video)
    if duration and duration > 0:
        n = int(round(duration * fps_target))
    else:
        n = (min_frames + max_frames) // 2
    return int(max(min_frames, min(max_frames, n)))


def _is_black_or_white(frame_path: Path, black_thr: int, white_thr: int) -> bool:
    """基于灰度均值判定纯黑/纯白（打开后即关闭，避免文件句柄泄漏）。"""
    with Image.open(frame_path) as im:
        gray = im.convert("L")
        hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return True
    mean = sum(i * n for i, n in enumerate(hist)) / total
    return mean <= black_thr or mean >= white_thr


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if sampling == "scene":
            count = _scene_extract(video, tmp_dir, scene_threshold, max_width=max_width, hwaccel=hwaccel)
            if count < 1:
                budget = adaptive_frame_budget(video, fps_target, min_frames, max_frames)
                _uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel)
        elif sampling == "keyframe":
            count = _keyframe_sample(video, tmp_dir, max_width=max_width, hwaccel=hwaccel)
            if count < min_frames:
                # 关键帧过少则回退均匀采样，保证帧数足够
                budget = adaptive_frame_budget(video, fps_target, min_frames, max_frames)
                _uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel)
        else:  # uniform（默认）
            budget = adaptive_frame_budget(video, fps_target, min_frames, max_frames)
            _uniform_sample(video, tmp_dir, budget, max_width=max_width, hwaccel=hwaccel)

        raw_frames = sorted(tmp_dir.glob("cap_*.jpg"))
        kept = [p for p in raw_frames if not _is_black_or_white(p, black_threshold, white_threshold)]

        # 过滤后过少则回退保留原始帧，避免空/过少结果（防御）
        if len(kept) < min_frames and raw_frames:
            kept = raw_frames

        # 均匀抽到最多 max_frames 帧（不按时间，按序号均分），避免只取开头
        kept = _uniform_sample_list(kept, max_frames)
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
    from concurrent.futures import ThreadPoolExecutor

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
