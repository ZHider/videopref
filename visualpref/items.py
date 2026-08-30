"""媒体条目（MediaItem）：视频=帧目录、图片=单文件的统一数据契约。

过去"条目"散落在多处隐式类型分支：``features.frames_dir_to_paths`` 用
``is_dir()`` 切换、``paths.video_key_of`` 从路径推缓存键、标注/批量推理/UI
各自扫描 ``frames/video`` 与 ``frames/image`` 两个子区。本模块把这一概念
显式化为 ``MediaItem``：

- ``key``：条目相对 ``frames/`` 根的子路径（``video/foo`` / ``image/foo.png``），
  同时是特征缓存键与 labels 的唯一键。
- ``kind``：``"video"`` | ``"image"``。
- ``path``：条目在磁盘上的路径（视频=目录，图片=单文件）。
- ``frame_paths()``：条目对应的全部帧图片（视频=枚举 ``*.jpg``，图片=单元素）。

扫描/解析的**唯一**实现也收敛在本模块：

- ``scan``：一次遍历两个子区，返回所有"有帧"的条目（供标注队列、批量推理
  缺帧判定、UI 下拉共用）。
- ``iter_entry_paths``：遍历两个子区全部条目路径（不做帧判定，供清空等场景）。
- ``from_entry_path``：由条目路径反向构造（供续标等场景）。

媒体路径 -> 条目的 manifest 解析见 ``manifest.resolve_item``（依赖 manifest，
放在 manifest 模块以避免循环导入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .paths import list_frame_files

__all__ = ["MediaItem"]


@dataclass(frozen=True)
class MediaItem:
    """媒体条目：视频=帧工作区目录，图片=单文件。

    Attributes
    ----------
    key : 条目相对 ``frames/`` 根的子路径（``video/foo`` / ``image/foo.png``）。
    kind : ``"video"`` | ``"image"``。
    path : 条目磁盘路径（视频=目录；图片=文件）。
    frames_root : 条目所属的 ``frames/`` 根目录。
    """

    key: str
    kind: str
    path: Path
    frames_root: Path = config.FRAMES_ROOT

    @classmethod
    def from_entry_path(
        cls,
        entry: Path,
        frames_root: Path | None = None,
    ) -> MediaItem:
        """由条目磁盘路径构造；kind 按子区名判定，未知子区回退 ``is_dir``。"""
        entry = Path(entry)
        root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
        try:
            key = entry.relative_to(root).as_posix()
        except ValueError:
            key = f"{entry.parent.name}/{entry.name}"
        if entry.parent.name == config.FRAMES_VIDEO_SUBDIR:
            kind = "video"
        elif entry.parent.name == config.FRAMES_IMAGE_SUBDIR:
            kind = "image"
        else:
            kind = "video" if entry.is_dir() else "image"
        return cls(key=key, kind=kind, path=entry, frames_root=root)

    # ------------------------------------------------------------------
    # 帧枚举
    # ------------------------------------------------------------------
    @property
    def frame_paths(self) -> list[Path]:
        """条目对应的全部帧图片路径（按文件名排序；图片条目为单元素）。"""
        if self.kind == "video":
            return list_frame_files(self.path)
        if self.path.is_file():
            return [self.path]
        return []

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------
    @classmethod
    def iter_entry_paths(cls, frames_root: Path | None = None) -> list[Path]:
        """遍历 ``frames/`` 两个子区的全部条目路径（视频目录 / 图片文件）。

        不做"有帧"判定，供清空记录等需要操作所有条目的场景使用。
        """
        root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
        if not root.is_dir():
            return []
        entries: list[Path] = []
        for sub in (config.FRAMES_VIDEO_SUBDIR, config.FRAMES_IMAGE_SUBDIR):
            subroot = root / sub
            if not subroot.is_dir():
                continue
            entries.extend(sorted(subroot.iterdir()))
        return entries

    @classmethod
    def scan(cls, frames_root: Path | None = None) -> list[MediaItem]:
        """扫描两个子区，返回所有**有帧**的媒体条目（按 key 排序）。

        判定与旧实现保持一致：视频目录至少含一张 ``*.jpg`` 帧；图片文件
        扩展名属于 ``IMAGE_EXTENSIONS`` 即收录。
        """
        root = Path(frames_root) if frames_root is not None else config.FRAMES_ROOT
        items: list[MediaItem] = []
        for entry in cls.iter_entry_paths(root):
            if entry.is_dir():
                if any(f.suffix.lower() == config.FRAME_EXT for f in entry.iterdir()):
                    items.append(cls.from_entry_path(entry, root))
            elif entry.suffix.lower() in config.IMAGE_EXTENSIONS:
                items.append(cls.from_entry_path(entry, root))
        return items
