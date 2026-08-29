"""标注工作流：视频路径列表 -> 一键拆帧 -> UI 逐视频标注 -> labels.json。

设计：
- 输入为文本（每行一个视频路径）。
- 拆帧采用时长自适应均匀抽样（见 ``frames.extract_frames``）。
- 队列条目: ``{"video_path", "key", "frames", "n_frames"}``，``key`` 即
  ``frames/`` 下的目录名（同名视频会带 hash 后缀），作为 labels 的唯一键。
- 标注结果: ``{key: 0/1}``。
- 进度持久化到 ``data/label_progress.json``，支持中断续标。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .frames import extract_from_input
from .paths import frames_dir_for_video, iter_video_files, list_frame_files


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def parse_video_list(text: str) -> list[Path]:
    """解析每行一个视频路径的文本；去重、过滤不存在的文件。

    支持每行一个**文件**或一个**文件夹**：
    - 文件：直接收录（若存在）。
    - 文件夹：递归扫描其下所有视频文件。
    """
    paths: list[Path] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        p = Path(line)
        if p.is_dir():
            # 文件夹：递归展开为视频文件
            for vp in iter_video_files(p, recursive=True):
                if str(vp) not in seen:
                    seen.add(str(vp))
                    paths.append(vp)
        elif p.is_file() and str(p) not in seen:
            seen.add(str(p))
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# 拆帧 + 队列构建
# ---------------------------------------------------------------------------
def extract_and_build_queue(
    video_paths: list[Path],
    frames_root: Path,
    sampling: str = config.DEFAULT_SAMPLING,
    scene_threshold: float = config.DEFAULT_SCENE_THRESHOLD,
    fps_target: float = config.FPS_TARGET,
    min_frames: int = config.MIN_FRAMES,
    max_frames: int = config.DEFAULT_MAX_FRAMES,
    black_threshold: int = config.BLACK_FRAME_MEAN,
    white_threshold: int = config.WHITE_FRAME_MEAN,
    progress=None,
) -> list[dict]:
    """逐个视频拆帧并构建标注队列。

    ``progress`` 可选回调 ``progress((i, total), desc=...)``。
    """
    frames_root = Path(frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)
    # 复用 extract_from_input：统一走同名去重 + manifest，队列 key 与训练解析一致
    out_dirs = extract_from_input(
        video_paths,
        frames_root,
        sampling=sampling,
        scene_threshold=scene_threshold,
        fps_target=fps_target,
        min_frames=min_frames,
        max_frames=max_frames,
        black_threshold=black_threshold,
        white_threshold=white_threshold,
        progress=progress,
    )
    queue: list[dict] = []
    for v, out_dir in zip(video_paths, out_dirs):
        frames = [str(p) for p in list_frame_files(out_dir)]
        queue.append(
            {
                "video_path": str(v),
                "key": out_dir.name,
                "frames": frames,
                "n_frames": len(frames),
            }
        )
    return queue


def queue_without_extraction(video_paths: list[Path], frames_root: Path) -> list[dict]:
    """对已拆帧的目录直接构建队列（不重新拆帧），用于续标/复查。"""
    queue = []
    for v in video_paths:
        out_dir = frames_dir_for_video(v, frames_root)
        frames = [str(p) for p in list_frame_files(out_dir)]
        if frames:
            queue.append(
                {
                    "video_path": str(v),
                    "key": out_dir.name,
                    "frames": frames,
                    "n_frames": len(frames),
                }
            )
    return queue


def build_queue_from_frames(frames_root: Path) -> list[dict]:
    """扫描 ``frames/`` 下所有含帧的子目录，直接构建标注队列（不重新拆帧）。

    - 每个子目录对应一个待标注视频。
    - ``video_path`` 优先取 manifest 中映射的真实路径（若有），否则用目录名
      （训练解析帧目录时，目录名经 sanitize 即可回落到自身，保证一致）。
    """
    from .paths import load_manifest

    frames_root = Path(frames_root)
    manifest = load_manifest(frames_root)
    # dir name -> 可能的真实 video_path 列表
    reverse: dict[str, list[str]] = {}
    for vp, dn in manifest.items():
        reverse.setdefault(dn, []).append(vp)

    queue = []
    if not frames_root.is_dir():
        return queue
    for d in sorted(frames_root.iterdir()):
        if not d.is_dir():
            continue
        frames = [str(p) for p in list_frame_files(d)]
        if not frames:
            continue
        vps = reverse.get(d.name, [])
        video_path = vps[0] if vps else d.name
        queue.append(
            {
                "video_path": video_path,
                "key": d.name,
                "frames": frames,
                "n_frames": len(frames),
            }
        )
    return queue


# ---------------------------------------------------------------------------
# 导出 / 进度持久化
# ---------------------------------------------------------------------------
def save_labels(labels_path: Path | str, queue: list[dict], labels: dict) -> int:
    """把已标注结果写为 labels.json：``[{video_path, label}]``。返回条数。

    ``video_path`` 一律存为绝对路径，避免训练时 CWD 不同导致 manifest 解析失配。
    """
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {"video_path": str(Path(e["video_path"]).resolve()), "label": int(labels[e["key"]])}
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
