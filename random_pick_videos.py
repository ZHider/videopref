#!/usr/bin/env python
"""从目录下随机选取若干视频文件，打印到标准输出或写入文件。

用法::

    python random_pick_videos.py <src_dir> [--count N] [--no-recursive] \\
        [--seed S] [--output FILE]

示例::

    python random_pick_videos.py "F:\\GV" --count 50
    python random_pick_videos.py "F:\\GV" --count 50 --seed 42 --output picked.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts", ".rmvb", ".rm",
}


def find_videos(root: Path, recursive: bool = True) -> list[Path]:
    """递归/非递归查找 root 下所有视频文件，返回绝对路径列表（升序）。"""
    root = root.resolve()
    if recursive:
        files = (p for p in root.rglob("*") if p.is_file())
    else:
        files = (p for p in root.iterdir() if p.is_file())
    return sorted(p for p in files if p.suffix.lower() in VIDEO_EXTS)


def pick(videos: list[Path], count: int, seed: int | None) -> list[Path]:
    """从视频列表中随机选取至多 count 个；seed 提供时结果可复现。"""
    rng = random.Random(seed)
    n = min(count, len(videos))
    return rng.sample(videos, n)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从目录下随机选取若干视频文件")
    p.add_argument("src", metavar="SRC_DIR", help="要扫描的源目录")
    p.add_argument("-n", "--count", type=int, default=50, help="随机选取数量（默认 50；超出总数时取全部）")
    p.add_argument("--no-recursive", action="store_true", help="不递归扫描子目录（默认递归）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（提供后结果可复现）")
    p.add_argument("-o", "--output", default=None, help="输出到文件（每行一个路径）；缺省打印到标准输出")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    root = Path(args.src)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    videos = find_videos(root, recursive=not args.no_recursive)
    print(f"共找到 {len(videos)} 个视频文件")

    if not videos:
        raise SystemExit("目录中没有找到任何视频文件")

    picked = pick(videos, args.count, args.seed)
    lines = [str(p) for p in picked]

    if args.output:
        out = Path(args.output)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"已随机选取 {len(picked)} 个视频 -> {out}")
    else:
        print(f"随机选取 {len(picked)} 个视频：")
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
