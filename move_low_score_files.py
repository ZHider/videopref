#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据三列 CSV 文件，将喜好分数低于阈值的文件移动到指定文件夹。

CSV 格式（列顺序）：文件名,喜好分数(0-1.0),文件全路径
示例：
  photo1.jpg,0.3,C:\\Users\\me\\Pictures\\photo1.jpg
  photo2.jpg,0.8,C:\\Users\\me\\Pictures\\photo2.jpg

用法：
  python move_low_score_files.py <csv_path>
  python move_low_score_files.py <csv_path> -d D:\\目标文件夹
  python move_low_score_files.py <csv_path> -t 0.6 --dry-run

核心逻辑在 ``move_low_score_files`` 纯函数，CLI 与 Gradio 均复用它。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


def _unique_dest(dest_dir: Path, filename: str) -> Path:
    """目标路径；已存在同名文件时追加序号（a_1.ext, a_2.ext ...）。"""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    base, ext = os.path.splitext(filename)
    i = 1
    while (dest_dir / f"{base}_{i}{ext}").exists():
        i += 1
    return dest_dir / f"{base}_{i}{ext}"


def move_low_score_files(
    rows: Iterable[list],
    dest_dir,
    threshold: float = 0.5,
    dry_run: bool = True,
):
    """处理三列 CSV 行，把分数低于阈值的文件移到 dest_dir。

    Parameters
    ----------
    rows : 可迭代的 CSV 行（list）。
    dest_dir : 目标文件夹。
    threshold : 分数阈值，低于该值的文件被移动。
    dry_run : True 时仅预览（不实际移动）。

    Returns
    -------
    (results, stats)
        results : list[dict]，每行一条，含 ``row_num``/``status``/``detail``
          及（若涉及）``filename``/``score``/``src``/``dest``。
          status 取值：ignored_header / skipped_empty / skipped_cols /
          skipped_invalid / skipped_high / skipped_missing / failed_notfound /
          failed_move / moved。
        stats : dict ``{moved, skipped, failed}``（表头不计入）。
    """
    dest_dir = Path(dest_dir)
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    stats = {"moved": 0, "skipped": 0, "failed": 0}

    for row_num, row in enumerate(rows, start=1):
        # 空行
        if not row or all(not cell.strip() for cell in row):
            results.append({"row_num": row_num, "status": "skipped_empty", "detail": "空行"})
            stats["skipped"] += 1
            continue
        # 列数不足
        if len(row) < 3:
            results.append({"row_num": row_num, "status": "skipped_cols", "detail": f"列数不足: {row}"})
            stats["skipped"] += 1
            continue

        filename = row[0].strip()
        score_str = row[1].strip()
        full_path = row[2].strip()

        # 表头（不计入统计）
        if filename.lower() in ("文件名", "filename", "name"):
            results.append({"row_num": row_num, "status": "ignored_header", "detail": "表头"})
            continue

        # 分数解析
        try:
            score = float(score_str)
        except ValueError:
            results.append(
                {"row_num": row_num, "filename": filename, "status": "skipped_invalid", "detail": f"分数无效: {score_str!r}"}
            )
            stats["skipped"] += 1
            continue
        if score >= threshold:
            results.append(
                {"row_num": row_num, "filename": filename, "score": score, "status": "skipped_high", "detail": f"分数 {score:.4f} >= 阈值"}
            )
            stats["skipped"] += 1
            continue

        # 路径
        if not full_path:
            results.append({"row_num": row_num, "filename": filename, "score": score, "status": "skipped_missing", "detail": "缺少文件路径"})
            stats["skipped"] += 1
            continue
        src = Path(full_path)
        if not src.is_file():
            results.append({"row_num": row_num, "filename": filename, "score": score, "status": "failed_notfound", "detail": f"文件不存在: {src}"})
            stats["failed"] += 1
            continue

        dest = _unique_dest(dest_dir, filename)
        if dry_run:
            results.append(
                {"row_num": row_num, "filename": filename, "score": score, "status": "moved",
                 "src": str(src), "dest": str(dest), "detail": "（预览）"}
            )
            stats["moved"] += 1
            continue

        try:
            shutil.move(str(src), str(dest))
            results.append(
                {"row_num": row_num, "filename": filename, "score": score, "status": "moved",
                 "src": str(src), "dest": str(dest), "detail": ""}
            )
            stats["moved"] += 1
        except Exception as e:  # noqa: BLE001 - 单文件移动失败不应中断整批
            results.append({"row_num": row_num, "filename": filename, "score": score, "status": "failed_move", "detail": f"移动出错: {e}"})
            stats["failed"] += 1

    return results, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 CSV 文件中的喜好分数，把低于阈值的文件移动到指定文件夹。"
    )
    parser.add_argument("csv_path", metavar="CSV", help="三列 CSV 文件路径（文件名,喜好分数,文件全路径）")
    parser.add_argument("-d", "--dest", help="目标文件夹")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="喜好分数阈值，低于该值的文件会被移动，默认：0.5")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不真正移动文件")
    parser.add_argument("--no-progress", action="store_true", help="不显示进度条")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if not 0 <= args.threshold <= 1.0:
        raise SystemExit(f"错误: 阈值必须在 0~1.0 之间，收到 {args.threshold}")
    if not Path(args.csv_path).is_file():
        raise SystemExit(f"错误: CSV 文件不存在: {args.csv_path}")

    with open(args.csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if args.no_progress:
        results, stats = move_low_score_files(rows, args.dest, args.threshold, args.dry_run)
    else:
        results, stats = move_low_score_files(
            tqdm(rows, desc="处理中", unit="行"), args.dest, args.threshold, args.dry_run
        )

    print("-" * 50)
    action = "预览" if args.dry_run else "移动"
    print(f"完成: {action} {stats['moved']} 个, 跳过 {stats['skipped']} 个, 失败 {stats['failed']} 个")
    if args.dry_run:
        for r in results:
            if r["status"] == "moved":
                print(f"  将移动: {r['filename']} ({r['score']:.4f}) -> {r['dest']}")


if __name__ == "__main__":
    main()
