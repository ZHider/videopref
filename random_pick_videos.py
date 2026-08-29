"""随机从 F:\\GV 目录下选择 150 个视频文件。"""

import glob
import os
import random

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts", ".rmvb", ".rm",
}

SRC_DIR = r"F:\GV\Drugs & Underage"
PICK_COUNT = 50


def find_videos(root: str) -> list:
    """递归查找 root 下所有视频文件，返回绝对路径列表。"""
    videos = []
    for file in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(file) and os.path.splitext(file)[1].lower() in VIDEO_EXTS:
            videos.append(file)
    return videos


def main() -> None:
    if not os.path.isdir(SRC_DIR):
        raise SystemExit(f"目录不存在: {SRC_DIR}")

    videos = find_videos(SRC_DIR)
    print(f"共找到 {len(videos)} 个视频文件")

    if not videos:
        raise SystemExit("目录中没有找到任何视频文件")

    n = min(PICK_COUNT, len(videos))
    picked = random.sample(videos, n)

    print(f"随机选取 {n} 个视频：")
    for p in picked:
        print(p)


if __name__ == "__main__":
    main()
