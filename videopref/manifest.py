"""manifest：``video_path -> frames 目录名`` 数据契约。

``frames/_manifest.json`` 把每个视频的路径映射到其 ``frames/`` 下的工作区目录名
（同名视频会加路径短哈希后缀去重）。``FramesNamer`` 负责幂等分配目录名，
``frames_dir_for_video`` 负责反向解析——两者同处本模块，保证"视频 -> 目录名"
契约的唯一实现，也避免与 ``paths`` 形成循环导入。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .paths import path_key, sanitize_name, short_hash


def manifest_path(frames_root: Path | None = None) -> Path:
    root = frames_root if frames_root is not None else config.FRAMES_ROOT
    return Path(root) / "_manifest.json"


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
    root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
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
