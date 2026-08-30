"""标注工作流：扫描 ``frames/`` 下已摄入的媒体条目 -> UI 逐项标注 -> labels.json。

设计：
- 队列条目: ``{"media_path", "key", "frames", "n_frames", "kind"}``，``key`` 即
  条目相对 ``frames/`` 根的子路径（如 ``video/foo`` / ``image/foo.png``），
  作为 labels 的唯一键；``kind`` 记录类型（视频/图片）以提升元数据密度。
- 标注结果: ``{key: 0/1}``。
- 进度持久化到 ``data/label_progress.json``，支持中断续标。
- 拆帧/摄入由独立的「拆帧」入口（``frames.extract_frames``/``extract_from_input``）
  完成，本模块只负责从已有条目构建队列，不再拆帧。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .items import MediaItem
from .manifest import load_manifest, manifest_dir_of
from .paths import iter_media_files


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def parse_media_list(text: str) -> list[Path]:
    """解析每行一个媒体路径的文本（视频或图片）；去重、过滤不存在的文件。

    支持每行一个**文件**或一个**文件夹**：
    - 文件：直接收录（若存在；视频与图片均可）。
    - 文件夹：递归扫描其下所有媒体文件（视频 + 图片）。
    """
    paths: list[Path] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        p = Path(line)
        if p.is_dir():
            # 文件夹：递归展开为媒体文件
            for vp in iter_media_files(p, recursive=True):
                if str(vp) not in seen:
                    seen.add(str(vp))
                    paths.append(vp)
        elif p.is_file() and str(p) not in seen:
            seen.add(str(p))
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# 队列构建（从已拆帧目录，不重新拆帧）
# ---------------------------------------------------------------------------
def build_queue_from_frames(frames_root: Path) -> list[dict]:
    """扫描 ``frames/`` 下所有已摄入媒体条目，直接构建标注队列（不重新拆帧/摄入）。

    - 视频工作区 ``frames/video/*``（目录）与图片 ``frames/image/*``（单文件）都纳入。
    - 每条: ``{"media_path", "key", "frames", "n_frames", "kind"}``。
      ``key`` 为条目相对根的子路径；``media_path`` 优先取 manifest 映射的真实路径。
    """
    frames_root = Path(frames_root)
    manifest = load_manifest(frames_root)
    # 条目子路径 -> 可能的真实媒体路径列表
    reverse: dict[str, list[str]] = {}
    for vp, val in manifest.items():
        d = manifest_dir_of(val)
        if d:
            reverse.setdefault(d, []).append(vp)

    queue = []
    for item in MediaItem.scan(frames_root):
        vps = reverse.get(item.key, [])
        media_path = vps[0] if vps else item.key
        queue.append(
            {
                "media_path": media_path,
                "key": item.key,
                "frames": [str(p) for p in item.frame_paths],
                "n_frames": item.n_frames,
                "kind": item.kind,
            }
        )
    return queue


# ---------------------------------------------------------------------------
# 导出 / 进度持久化
# ---------------------------------------------------------------------------
def save_labels(labels_path: Path | str, queue: list[dict], labels: dict) -> int:
    """把已标注结果写为 labels.json：``[{media_path, label, kind}]``。返回条数。

    ``media_path`` 一律存为绝对路径，避免训练时 CWD 不同导致 manifest 解析失配；
    兼容读取历史队列条目的 ``video_path`` 字段；``kind`` 记录类型（video/image）。
    """
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "media_path": str(
                Path(e.get("media_path") or e.get("video_path") or e["key"]).resolve()
            ),
            "label": int(labels[e["key"]]),
            "kind": e.get("kind", "video"),
        }
        for e in queue
        if e["key"] in labels
    ]
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return len(items)


def progress_file() -> Path:
    return config.DATA_DIR / "label_progress.json"


def save_progress(state: dict) -> Path:
    """持久化标注会话（队列 + 已标 labels + 当前索引）。"""
    p = progress_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return p


def load_progress() -> dict | None:
    """读取上次标注会话；不存在或损坏返回 None。"""
    p = progress_file()
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict) and isinstance(state.get("queue"), list):
            return state
    except Exception:
        return None
    return None


def new_state() -> dict:
    return {"queue": [], "idx": 0, "labels": {}, "skipped": []}


def current_entry(state: dict) -> dict | None:
    idx = int(state.get("idx", 0))
    queue = state.get("queue", [])
    if 0 <= idx < len(queue):
        return queue[idx]
    return None


def progress_text(state: dict) -> str:
    queue = state.get("queue", [])
    idx = min(int(state.get("idx", 0)), len(queue))
    labeled = sum(1 for e in queue if e["key"] in state.get("labels", {}))
    return f"进度 {idx}/{len(queue)} ｜ 已标注 {labeled}"


# ---------------------------------------------------------------------------
# 标注会话状态机（纯逻辑，无 UI 依赖）
# ---------------------------------------------------------------------------
def render_state(state: dict) -> tuple[str, str, list]:
    """根据 state 渲染（标题, 进度文本, 帧图路径列表）。标题带类型图标。"""
    entry = current_entry(state)
    if entry is None:
        return "🎉 全部完成（或队列为空）", progress_text(state), []
    icon = "🖼" if entry.get("kind") == "image" else "🎬"
    return f"{icon} **{entry['key']}**　({entry['n_frames']} 帧)", progress_text(state), entry["frames"]


def sanitize_state(state: dict, frames_root: Path) -> dict:
    """续标时校验队列条目：重建帧列表、剔除已失效目录。"""
    frames_root = Path(frames_root)
    queue = []
    labels = {}
    for e in state.get("queue", []):
        item = MediaItem.from_entry_path(frames_root / e["key"], frames_root)
        frames = [str(p) for p in item.frame_paths]
        if not frames:
            continue
        e2 = dict(e)
        e2["frames"] = frames
        e2["n_frames"] = len(frames)
        queue.append(e2)
        if e["key"] in state.get("labels", {}):
            labels[e["key"]] = state["labels"][e["key"]]
    return {
        "queue": queue,
        "idx": min(int(state.get("idx", 0)), len(queue)),
        "labels": labels,
        "skipped": state.get("skipped", []),
    }


def advance(state: dict, label: int | None = None) -> dict:
    """记录标签（若提供）并前进；返回更新后的 state。"""
    entry = current_entry(state)
    if entry is not None and label is not None:
        state["labels"][entry["key"]] = label
    state["idx"] = int(state.get("idx", 0)) + 1
    return state
