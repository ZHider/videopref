"""文件系统数据契约：路径解析、安全化命名、帧文件枚举。

- 纯路径/枚举工具在本模块；
- manifest 与 ``FramesNamer``/``frames_dir_for_video`` 见 ``manifest.py``。

全程使用 ``pathlib.Path``，禁止字符串拼接路径。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import config


def sanitize_name(name: str) -> str:
    """将视频名安全化：去特殊字符、空格与点替换为下划线。

    仅保留字母数字、`-`、`_`；空格、点(`.`)及其它符号统一替换为 ``_``，
    并压缩连续下划线。例如 ``"My Video! (Part 1).mp4"`` -> ``"My_Video_Part_1_mp4"``。
    """
    name = name.strip()
    # 替换非法字符（空格、点、符号）为下划线
    cleaned = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned


def path_key(video_path: Path) -> str:
    """稳定的视频路径 key（绝对规范化路径），用于 manifest 与 hash。"""
    return str(Path(video_path).resolve())


def short_hash(video_path: Path, length: int = 8) -> str:
    """基于完整路径的稳定短哈希，用于同名视频去重后缀。"""
    return hashlib.sha256(path_key(video_path).encode("utf-8")).hexdigest()[:length]


def list_frame_files(frames_dir: Path, ext: str = config.FRAME_EXT) -> list[Path]:
    """列出目录下全部帧文件，按文件名升序恢复时序。

    返回空列表表示目录为空或不存在（空文件夹异常由调用方决定是否防御）。
    """
    if not frames_dir.is_dir():
        return []
    return sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() == ext)


def iter_video_files(folder: Path, recursive: bool = True) -> list[Path]:
    """枚举文件夹下的视频文件；``recursive=True`` 时递归扫描所有子目录。"""
    folder = Path(folder)
    exts = config.VIDEO_EXTENSIONS
    if recursive:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def is_video(path: Path) -> bool:
    """按扩展名判断是否为视频文件。"""
    return Path(path).suffix.lower() in config.VIDEO_EXTENSIONS


def is_image(path: Path) -> bool:
    """按扩展名判断是否为图片文件。"""
    return Path(path).suffix.lower() in config.IMAGE_EXTENSIONS


def is_media(path: Path) -> bool:
    """按扩展名判断是否为媒体文件（视频或图片）。"""
    return Path(path).suffix.lower() in config.MEDIA_EXTENSIONS


def iter_media_files(folder: Path, recursive: bool = True) -> list[Path]:
    """枚举文件夹下的媒体文件（视频 + 图片）；``recursive=True`` 时递归扫描。"""
    folder = Path(folder)
    exts = config.MEDIA_EXTENSIONS
    if recursive:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def frame_filename(index: int, width: int = config.FRAME_FILENAME_WIDTH) -> str:
    """生成零填充编号文件名，如 ``0001.jpg``。"""
    return f"{index:0{width}d}{config.FRAME_EXT}"


def feature_cache_path(cache_dir: Path, video_key: str, suffix: str = ".pt") -> Path:
    """返回某视频对应的特征缓存文件路径。"""
    return (cache_dir / video_key).with_suffix(suffix)


def video_key_of(frames_dir: Path) -> str:
    """由媒体条目路径生成稳定缓存键。

    键为条目相对 ``frames/`` 根的子路径（如 ``video/foo``、``image/foo.png``），
    保证视频与图片同 basename 时不互相覆盖缓存。无法相对根时回退为
    ``父目录名/条目名``，避免自定义 frames_root 下 key 冲突。
    """
    p = Path(frames_dir)
    try:
        return p.relative_to(config.FRAMES_ROOT).as_posix()
    except ValueError:
        return f"{p.parent.name}/{p.name}"

