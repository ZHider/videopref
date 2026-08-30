"""CPU 流水线预取工具。

把"逐个 CPU 预处理（解码/缩放/归一化）-> 消费"这种生产者-消费者结构从业务
代码里抽出来。典型用途：特征提取时用后台线程做纯 CPU 预处理，与主线程的 GPU
前向重叠，避免 GPU 空等 CPU（背压使生产者限速，防止内存无限增长）。

后台线程只做纯 CPU 工作，绝不碰 CUDA，避免并发访问设备的风险。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def prefetch_map(fn: Callable[[T], R], items: Iterable[T], depth: int = 2) -> Iterable[R]:
    """惰性迭代 ``fn(item)`` 的结果，CPU 预处理在后台线程进行。

    - ``depth < 1`` 或 ``items`` 至多 1 个时，走同步路径（不启线程）。
    - 否则后台线程逐个调用 ``fn`` 并把结果放入容量为 ``depth`` 的有界队列；
      主线程在队列满时阻塞（背压），使 CPU 预处理与下游消费重叠。
    - 生产者结束后放入 ``None`` 哨兵，消费者据此终止。
    """
    items = list(items)
    if depth < 1 or len(items) <= 1:
        for item in items:
            yield fn(item)
        return

    q: queue.Queue = queue.Queue(maxsize=max(1, depth))

    def _producer() -> None:
        try:
            for item in items:
                q.put(fn(item))
        finally:
            q.put(None)  # 哨兵

    threading.Thread(target=_producer, daemon=True).start()

    got = 0
    while got < len(items):
        out = q.get()
        if out is None:
            break
        got += 1
        yield out
