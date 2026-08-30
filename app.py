#!/usr/bin/env python
"""Gradio 应用入口（六 Tab：拆帧/标注/推理/批量推理/工具/训练）。

用 argparse 捕获启动参数：``--server-name`` / ``--server-port`` 显式解析，
其余所有 ``--key value`` 一律**原样透传**为 ``**kwargs`` 交给 ``demo.launch``，
因此可直接使用 Gradio ``Blocks.launch`` 支持的任何参数（如 ``--share``、
``--inbrowser``、``--auth "user:pass"``、``--allowed-path <path>`` ...）。

已知需要整数的透传参数（``--max-threads``、``--width``、``--height``、
``--max-file-size``）会自动 ``int`` 化，其余保持字符串原样。

用法::

    python app.py                          # 默认 127.0.0.1:7860
    python app.py --server-port 7861 --share
    python app.py --auth "me:secret" --max-file-size "50mb"
    python app.py --allowed-path /some/dir
"""

from __future__ import annotations

import argparse

from visualpref.gradio_app import launch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="启动 visual-pref Gradio 界面")
    p.add_argument("--server-name", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    p.add_argument("--server-port", "--port", type=int, default=7860, help="监听端口（默认 7860）")
    return p


# 透传参数中需要 int 化的键（对应 gradio.launch 的整数参数）；
# 无法 int 化（如 "--max-file-size 50mb"）时回退为原字符串。
_INT_PASS_THROUGH_KEYS = frozenset({"max_threads", "width", "height", "max_file_size"})


def _coerce_passthrough(key: str, value) -> object:
    if key in _INT_PASS_THROUGH_KEYS and isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def _pass_through(extras: list[str]) -> dict:
    """把 ``parse_known_args`` 捕获的未知参数（``--key value``）转成 kwargs，原样透传。

    - ``--key=value``  -> ``key=value``
    - ``--key value``  -> ``key=value``
    - ``--flag``（无值）-> ``flag=True``
    - 连字符转下划线：``--max-file-size`` -> ``max_file_size``
    - 白名单键（见 ``_INT_PASS_THROUGH_KEYS``）自动 int 化。
    """
    kwargs: dict = {}
    i, n = 0, len(extras)
    while i < n:
        tok = extras[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            k, _, v = body.partition("=")
            kwargs[k.replace("-", "_")] = _coerce_passthrough(k.replace("-", "_"), v)
            i += 1
            continue
        key = body.replace("-", "_")
        if i + 1 < n and not extras[i + 1].startswith("--"):
            kwargs[key] = _coerce_passthrough(key, extras[i + 1])
            i += 2
        else:
            kwargs[key] = True
            i += 1
    return kwargs


def main(argv=None) -> None:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    kwargs = _pass_through(extras)
    launch(server_name=args.server_name, server_port=args.server_port, **kwargs)


if __name__ == "__main__":
    main()
