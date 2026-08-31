"""ffmpeg 底层封装：命令构建与解码执行。

所有抽样策略（scene / keyframe / uniform）都构造形如
``[ffmpeg, -y, (-hwaccel), (-skip_frame), -i video, (-vf), (-vsync vfr), -q:v 2, out``
的命令，仅各自的 ``-vf`` 滤镜与 ``-skip_frame``/``-vsync`` 不同。本模块把公共
部分收敛为 ``build_ffmpeg_cmd``，策略模块只提供差异化参数。

Windows 下 ffmpeg 子进程输出用 ``utf-8 + errors="replace"`` 解码，避免中文路径
/ 非 GBK 字节触发 ``UnicodeDecodeError``（历史坑，勿回退）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Windows 默认用 GBK 解码子进程输出，而 ffmpeg 输出含非 GBK 字节会抛 UnicodeDecodeError。
# 统一用 utf-8 + errors="replace"，保证任何情况下都不会因解码崩溃。
_SUBPROCESS_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def ffmpeg_available() -> bool:
    """ffmpeg 是否在 PATH 中可用。"""
    return shutil.which("ffmpeg") is not None


def ffmpeg_path() -> str:
    """返回 ffmpeg 路径；不存在则抛出带安装提示的异常。"""
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH。")
    return p


def probe_duration(video: Path | str) -> float | None:
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


def run_ffmpeg(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
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


def scale_filter(size: int) -> str:
    """生成"center-crop 成 size×size 正方形"滤镜（先裁正方形再缩放到 size）。

    与 ``DINOv3ViTImageProcessor`` 的 resize 语义对齐：保持纵横比、中心裁切成
    正方形再缩放到 ``size``，喂模型时 processor 的 resize 退化为恒等（跳过昂贵
    缩放）。0=不缩放。

    注意：filter 链内 ``min(iw,ih)`` 的逗号必须转义为 ``\\,``，否则会被当成
    filter 分隔符。
    """
    if size and size > 0:
        s = str(size)
        crop = "crop=min(iw\\,ih):min(iw\\,ih):(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2"
        return f"{crop},scale={s}:{s}"
    return ""


def build_ffmpeg_cmd(
    ffmpeg: str,
    video: Path | str,
    out_pattern: Path | str,
    vf_parts: list[str] | None = None,
    hwaccel: str | None = None,
    skip_frame: str | None = None,
    fps_mode: bool = True,
) -> list[str]:
    """构建统一的抽帧 ffmpeg 命令。

    Parameters
    ----------
    ffmpeg : ffmpeg 可执行文件路径。
    video : 输入视频。
    out_pattern : 输出文件模式，如 ``out/cap_%06d.jpg``。
    vf_parts : ``-vf`` 滤镜片段（将用逗号拼接）；None 则省略 ``-vf``。
    hwaccel : 可选硬件解码（如 ``"cuda"``）。
    skip_frame : 可选 ``-skip_frame`` 值（如 ``"nokey"`` 只解 I 帧）。
    fps_mode : 是否追加 ``-vsync vfr``（uniform 模式不需要）。
    """
    cmd = [ffmpeg, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if skip_frame:
        cmd += ["-skip_frame", skip_frame]
    cmd += ["-i", str(video)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    if fps_mode:
        cmd += ["-vsync", "vfr"]
    cmd += ["-q:v", "2", str(out_pattern)]
    return cmd
