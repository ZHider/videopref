"""文件系统数据契约：路径解析、安全化命名、帧文件枚举。

全程使用 ``pathlib.Path``，禁止字符串拼接路径。
"""

from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------------------
# manifest：video_path -> frames 目录名（数据契约的一部分）
# ---------------------------------------------------------------------------
def manifest_path(frames_root: Path | None = None) -> Path:
    root = frames_root if frames_root is not None else config.FRAMES_ROOT
    return root / "_manifest.json"


def load_manifest(frames_root: Path | None = None) -> dict[str, str]:
    p = manifest_path(frames_root)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            return m if isinstance(m, dict) else {}
        except Exception:
            return {}
    return {}


def save_manifest(frames_root: Path | None, manifest: dict[str, str]) -> Path:
    p = manifest_path(frames_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return p


def frames_dir_for_video(
    video_path: Path,
    frames_root: Path | None = None,
    manifest: dict[str, str] | None = None,
) -> Path:
    """解析视频对应的 ``frames/`` 工作区目录。

    优先查 manifest（同名视频可能被加过 hash 后缀）；以多种 key 形式匹配
    （规范化绝对路径 / 绝对路径 / 原始字符串），降低 CWD 差异导致的失配；
    均未命中则回退为 ``frames/{sanitized_stem}/``。
    用于训练/标注时按 video_path 定位帧目录。
    """
    root = frames_root if frames_root is not None else config.FRAMES_ROOT
    if manifest is None:
        manifest = load_manifest(root)
    p = Path(video_path)
    for key in (path_key(p), str(p.absolute()), str(p)):
        name = manifest.get(key)
        if name is not None:
            return root / name
    return root / sanitize_name(p.stem)


class FramesNamer:
    """批量分配唯一帧目录名；同名视频自动追加路径短哈希（如 ``a_1a2b3c4``）。

    分配与解析（``frames_dir_for_video``）同处本模块，保证"视频 -> 目录名"
    契约的唯一实现。幂等：同一视频重复拆帧复用既有目录名。
    """

    def __init__(self, frames_root: Path):
        self.frames_root = Path(frames_root)
        self.manifest = load_manifest(self.frames_root)
        # 仅把 manifest 中已登记的目录名视为"已被其他视频占用"；
        # 磁盘上未登记的旧目录视为同视频既往产物，可复用（幂等）。
        self.used: set[str] = set(self.manifest.values())

    def assign(self, video_path: Path) -> Path:
        key = path_key(video_path)
        if key in self.manifest:
            return self.frames_root / self.manifest[key]
        base = sanitize_name(Path(video_path).stem)
        name = base
        if name in self.used:
            name = f"{base}_{short_hash(video_path)}"
        self.used.add(name)
        self.manifest[key] = name
        return self.frames_root / name

    def save(self) -> Path:
        return save_manifest(self.frames_root, self.manifest)


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


def frame_filename(index: int, width: int = config.FRAME_FILENAME_WIDTH) -> str:
    """生成零填充编号文件名，如 ``0001.jpg``。"""
    return f"{index:0{width}d}{config.FRAME_EXT}"


def feature_cache_path(cache_dir: Path, video_key: str, suffix: str = ".pt") -> Path:
    """返回某视频对应的特征缓存文件路径。"""
    return (cache_dir / video_key).with_suffix(suffix)


def video_key_of(frames_dir: Path) -> str:
    """由 frames 子目录名生成稳定缓存键（与 sanitize 保持一致）。"""
    return frames_dir.name
