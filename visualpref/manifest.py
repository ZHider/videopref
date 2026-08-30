"""manifest：``media_path -> 条目元数据`` 数据契约。

``frames/_manifest.json`` 把每个媒体（视频/图片）的真实路径映射到其在
``frames/`` 下的条目信息，元数据含类型以提升信息密度::

    {
      "F:/.../a.mp4": {"dir": "video/a",    "kind": "video", "ext": ".mp4"},
      "C:/.../b.png": {"dir": "image/b.png", "kind": "image", "ext": ".png"}
    }

- ``dir``：条目相对 ``frames/`` 根的子路径。视频=工作区目录（``video/foo``），
  图片=单文件（``image/foo.png``）。
- ``kind``：``"video"`` | ``"image"``。
- ``ext``：源文件扩展名。

``FramesNamer`` 负责幂等分配条目子路径（同名媒体加路径短哈希去重），
``frames_dir_for_video`` 负责反向解析。两者同处本模块，保证"媒体 -> 条目路径"
契约的唯一实现，也避免与 ``paths`` 形成循环导入。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .paths import is_image, path_key, sanitize_name, short_hash


def manifest_path(frames_root: Path | None = None) -> Path:
    root = frames_root if frames_root is not None else config.FRAMES_ROOT
    return Path(root) / "_manifest.json"


def load_manifest(frames_root: Path | None = None) -> dict[str, object]:
    p = manifest_path(frames_root)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            return m if isinstance(m, dict) else {}
        except Exception:
            return {}
    return {}


def save_manifest(frames_root: Path | None, manifest: dict[str, object]) -> Path:
    p = manifest_path(frames_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return p


def manifest_dir_of(value: object) -> str:
    """从 manifest 值里取出条目子路径（``video/foo`` / ``image/foo.png``）。

    兼容结构化 dict 与历史裸字符串两种形态。
    """
    if isinstance(value, dict):
        return value.get("dir") or value.get("key") or ""
    return value if isinstance(value, str) else ""


def frames_dir_for_video(
    video_path: Path,
    frames_root: Path | None = None,
    manifest: dict[str, object] | None = None,
) -> Path:
    """解析媒体对应的 ``frames/`` 条目路径（视频=目录，图片=单文件）。

    优先查 manifest（同名媒体可能被加过 hash 后缀）；以多种 key 形式匹配
    （规范化绝对路径 / 绝对路径 / 原始字符串），降低 CWD 差异导致的失配；
    均未命中则按类型回退：视频 -> ``video/{sanitized_stem}``，
    图片 -> ``image/{sanitized_stem}{ext}``。
    用于训练/标注时按 video_path 定位条目。
    """
    root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
    if manifest is None:
        manifest = load_manifest(root)
    p = Path(video_path)
    for key in (path_key(p), str(p.absolute()), str(p)):
        val = manifest.get(key)
        if val is not None and manifest_dir_of(val):
            return root / manifest_dir_of(val)
    if is_image(p):
        return root / config.FRAMES_IMAGE_SUBDIR / (sanitize_name(p.stem) + p.suffix.lower())
    return root / config.FRAMES_VIDEO_SUBDIR / sanitize_name(p.stem)


def frames_key_for(
    media_path: Path,
    frames_root: Path | None = None,
    manifest: dict[str, object] | None = None,
) -> str:
    """返回媒体条目相对 ``frames/`` 根的子路径 key（``video/foo`` / ``image/foo.png``）。

    供缺帧判定等场景与 ``FramesNamer``/扫描产出的 key 集合比对。
    """
    item = frames_dir_for_video(media_path, frames_root, manifest)
    root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
    return item.relative_to(root).as_posix()


class FramesNamer:
    """批量分配唯一媒体条目子路径；同名媒体自动追加路径短哈希（如 ``a_1a2b3c4``）。

    - 视频条目 -> ``video/{sanitized_stem}``（目录）
    - 图片条目 -> ``image/{sanitized_stem}{ext}``（单文件，保留原扩展名）

    分配与解析（``frames_dir_for_video``）同处本模块，保证契约唯一实现。
    幂等：同一媒体重复摄入复用既有条目路径。manifest 值为结构化 dict
    （含 ``dir``/``kind``/``ext``），提升元数据信息密度。
    """

    def __init__(self, frames_root: Path):
        self.frames_root = Path(frames_root)
        self.manifest = load_manifest(self.frames_root)
        # 仅把 manifest 中已登记的条目子路径视为"已被其他媒体占用"；
        # 磁盘上未登记的旧产物视为同媒体既往结果，可复用（幂等）。
        self.used: set[str] = set(
            d for v in self.manifest.values() if (d := manifest_dir_of(v))
        )

    def assign(self, media_path: Path) -> Path:
        key = path_key(media_path)
        if key in self.manifest:
            return self.frames_root / manifest_dir_of(self.manifest[key])
        p = Path(media_path)
        stem = sanitize_name(p.stem)
        ext = p.suffix.lower()
        if is_image(p):
            sub, base = config.FRAMES_IMAGE_SUBDIR, f"{stem}{ext}"
        else:
            sub, base = config.FRAMES_VIDEO_SUBDIR, stem
        value = f"{sub}/{base}"
        if value in self.used:
            value = f"{sub}/{stem}_{short_hash(media_path)}{ext if is_image(p) else ''}"
        self.used.add(value)
        self.manifest[key] = {
            "dir": value,
            "kind": "image" if is_image(p) else "video",
            "ext": ext,
        }
        return self.frames_root / value

    def save(self) -> Path:
        return save_manifest(self.frames_root, self.manifest)
