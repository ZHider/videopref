#!/usr/bin/env python
"""数据契约回归测试：验证 MediaItem/manifest/labels/透传等重构后的行为等价性。

无 GPU、无网络依赖，秒级运行；在项目根创建临时目录 ``vp_verify_tmp`` 验证后自动清理。

用法::

    uv run python scripts/verify_contract.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import visualpref.train as train_mod
from visualpref.dataset import load_labels
from visualpref.frames import _make_ingest_handlers
from visualpref.items import MediaItem
from visualpref.labeling import (
    build_queue_from_frames,
    sanitize_state,
    save_labels,
)
from visualpref.manifest import FramesNamer, item_key_for, resolve_item
from visualpref.paths import feature_cache_path

ok = 0


def check(name: str, cond: bool) -> None:
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1
    print(f"  ok - {name}")


def main() -> int:
    root = PROJECT_ROOT / "vp_verify_tmp"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    try:
        _run_checks(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\nALL {ok} CHECKS PASSED")
    return 0


def _run_checks(root: Path) -> None:
    frames_root = root / "frames"
    video_sub = frames_root / "video"
    image_sub = frames_root / "image"
    video_sub.mkdir(parents=True)
    image_sub.mkdir(parents=True)

    # 构造条目：视频目录 2 个（一个有帧一个空）、图片 2 张 + 1 个非图片文件
    vid_a = video_sub / "a"
    vid_a.mkdir()
    (vid_a / "0001.jpg").write_bytes(b"f1")
    (vid_a / "0002.jpg").write_bytes(b"f2")
    (video_sub / "b").mkdir()  # 空目录 -> 不应出现在 scan
    img_a = image_sub / "p1.png"
    img_a.write_bytes(b"png-data")
    (image_sub / "p2.webp").write_bytes(b"webp-data")
    (image_sub / "note.txt").write_bytes(b"note")  # 非图片文件 -> 不应出现在 scan

    print("== MediaItem.from_entry_path ==")
    it = MediaItem.from_entry_path(vid_a, frames_root)
    check("video kind", it.kind == "video")
    check("video key", it.key == "video/a")
    check("video frame_paths", [p.name for p in it.frame_paths] == ["0001.jpg", "0002.jpg"])
    check("video n_frames", it.n_frames == 2)
    it = MediaItem.from_entry_path(img_a, frames_root)
    check("image kind", it.kind == "image")
    check("image key", it.key == "image/p1.png")
    check("image frame_paths 单元素", it.frame_paths == [img_a])

    print("== MediaItem.scan / iter_entry_paths ==")
    keys = [i.key for i in MediaItem.scan(frames_root)]
    check("scan 收录有帧条目", keys == ["video/a", "image/p1.png", "image/p2.webp"])
    entries = MediaItem.iter_entry_paths(frames_root)
    check(
        "iter_entry_paths 遍历两子区",
        sorted(str(p.name) for p in entries) == ["a", "b", "note.txt", "p1.png", "p2.webp"],
    )

    print("== manifest.resolve_item（含 hash 后缀与回退） ==")
    media_v = root / "src" / "my video.mp4"
    media_v.parent.mkdir(parents=True)
    media_v.write_bytes(b"v")
    media_img = root / "src" / "pic.PNG"
    media_img.write_bytes(b"i")

    it = resolve_item(media_v, frames_root)
    check("回退视频条目", it.key == "video/my_video" and it.kind == "video")
    it = resolve_item(media_img, frames_root)
    check("回退图片条目", it.key == "image/pic.png" and it.kind == "image")

    namer = FramesNamer(frames_root)
    it1 = namer.assign(media_v)
    it2 = namer.assign(media_img)
    check("分配视频条目", it1.key == "video/my_video" and it1.kind == "video")
    check("分配图片条目", it2.key == "image/pic.png" and it2.kind == "image")
    check("assign 幂等", namer.assign(media_v) == it1)
    media_v2 = root / "src2" / "my video.mp4"
    media_v2.parent.mkdir(parents=True)
    media_v2.write_bytes(b"v2")
    it3 = namer.assign(media_v2)
    check(
        "同名视频加 hash 去重",
        it3.key != it1.key and it3.key.startswith("video/my_video_") and it3.kind == "video",
    )
    check("manifest 结构化", namer.manifest[str(media_v)]["kind"] == "video")
    mf = namer.save()
    check(
        "manifest 保存",
        mf.is_file() and json.loads(mf.read_text(encoding="utf-8"))[str(media_v)]["dir"] == "video/my_video",
    )
    it = resolve_item(media_v, frames_root)
    check("manifest 命中", it.key == "video/my_video" and it.path == frames_root / "video/my_video")
    check("item_key_for", item_key_for(media_v2, frames_root) == it3.key)

    print("== labeling.build_queue_from_frames ==")
    queue = build_queue_from_frames(frames_root)
    qkeys = [e["key"] for e in queue]
    check("队列含全部有帧条目", "video/a" in qkeys and "image/p1.png" in qkeys)
    q_a = next(e for e in queue if e["key"] == "video/a")
    check(
        "队列字段",
        q_a["media_path"] == "video/a" and q_a["n_frames"] == 2
        and q_a["kind"] == "video" and len(q_a["frames"]) == 2,
    )

    print("== save_labels / load_labels（新旧字段兼容） ==")
    labels = {"video/a": 1, "image/p1.png": 0}
    labels_path = root / "labels.json"
    n = save_labels(labels_path, queue, labels)
    check("导出条数", n == 2)
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    check("导出字段 media_path", all("media_path" in d and "video_path" not in d for d in data))
    loaded = load_labels(labels_path)
    check("load_labels 新字段", all("media_path" in d for d in loaded) and loaded[0]["label"] in (0, 1))
    old = [{"video_path": "/x/a.mp4", "label": 1, "kind": "video"}]
    old_path = root / "old_labels.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    loaded_old = load_labels(old_path)
    check("load_labels 兼容旧 video_path", loaded_old[0]["media_path"] == "/x/a.mp4")

    print("== save_labels 兼容旧 progress 队列 ==")
    old_queue = [{"video_path": "/x/a.mp4", "key": "video/a", "frames": [], "n_frames": 2, "kind": "video"}]
    n = save_labels(root / "mixed.json", old_queue, {"video/a": 1})
    check(
        "旧队列字段导出",
        n == 1 and json.loads((root / "mixed.json").read_text(encoding="utf-8"))[0]["media_path"].endswith("a.mp4"),
    )

    print("== sanitize_state（续标） ==")
    st = {"queue": queue, "idx": 0, "labels": {"video/a": 1}, "skipped": []}
    st2 = sanitize_state(st, frames_root)
    check("续标保留标签与队列", st2["labels"] == {"video/a": 1} and len(st2["queue"]) == len(queue))

    print("== feature_cache_path ==")
    check("缓存路径按 key 分区", feature_cache_path(root / "cache", "video/a") == root / "cache" / "video" / "a.pt")
    check("图片缓存不与视频冲突", feature_cache_path(root / "cache", "image/a.png") != feature_cache_path(root / "cache", "video/a"))

    print("== app._pass_through int 白名单 ==")
    spec = importlib.util.spec_from_file_location("app_mod", str(PROJECT_ROOT / "app.py"))
    app_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(app_mod)
    kw = app_mod._pass_through(["--max-threads", "4", "--share", "--max-file-size", "50mb", "--auth", "me:secret"])
    check("int 白名单转换", kw["max_threads"] == 4 and isinstance(kw["max_threads"], int))
    check("str 回退", kw["max_file_size"] == "50mb")
    check("flag 与普通字符串", kw["share"] is True and kw["auth"] == "me:secret")
    kw2 = app_mod._pass_through(["--width=800"])
    check("= 形式 int 化", kw2["width"] == 800)

    print("== frames._make_ingest_handlers 分派 ==")
    handlers = _make_ingest_handlers(
        sampling="uniform", scene_threshold=0.3, fps_target=0.5,
        min_frames=2, max_frames=8, black_threshold=10, white_threshold=245,
        max_width=640, image_max_width=0, hwaccel=None,
    )
    check("分派表含 video/image", set(handlers) == {"video", "image"})
    src_img = root / "pic.png"
    src_img.write_bytes(b"PNGDATA")
    img_item = MediaItem.from_entry_path(frames_root / "image" / "pic.png", frames_root)
    handlers["image"](src_img, img_item)
    check("image 处理器字节级复制", img_item.path.read_bytes() == b"PNGDATA")

    print("== train._maybe_save_best 择优 ==")
    _saved = []

    class _DummyModel:
        def state_dict(self):
            return {}

    def _fake_save(path, model, cfg, lm, st):
        _saved.append((str(path), dict(st)))

    orig_save = train_mod.save_checkpoint
    try:
        train_mod.save_checkpoint = _fake_save
        out_dir = root / "out"
        ckpt_cfg = {"feature_dim": 768}
        lm = {"like": 1}
        bk, bp = train_mod._maybe_save_best(_DummyModel(), out_dir, ckpt_cfg, lm, 0.9, 0.1, 1, (-float("inf"), -float("inf")), None)
        check("更高 auc 保存最佳", bk == (0.9, -0.1) and len(_saved) == 1)
        bk2, bp2 = train_mod._maybe_save_best(_DummyModel(), out_dir, ckpt_cfg, lm, 0.7, 0.3, 2, bk, bp)
        check("更低 auc 不覆盖", bk2 == bk and bp2 == bp and len(_saved) == 1)
    finally:
        train_mod.save_checkpoint = orig_save


if __name__ == "__main__":
    raise SystemExit(main())
