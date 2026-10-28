"""Bounded, read-only report loading and terminal-safe formatting; no ML imports."""

import json
import math
import unicodedata
from pathlib import Path


def text(value):
    return "".join(c if not unicodedata.category(c).startswith("C") else " " for c in str(value))


def read(path, warn=False, kind=dict):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("结果清单超过 16 MiB，请直接查看原文件")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, kind):
            raise ValueError("结果清单格式不正确")
        return value
    except (OSError, ValueError) as error:
        if warn:
            print(f"  无法读取 {text(path)}：{text(error)}")
        return kind()


def mapping(value):
    return value if isinstance(value, dict) else {}


def rows(value):
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def number(value, percent=False):
    if not finite(value):
        return "—（未记录/不适用）"
    return f"{value:.4f}" + (f"（{value * 100:.2f}%）" if percent else "")


def count(value):
    return str(value) if type(value) is int and value >= 0 else "未记录"


def named(value, names):
    return names.get(value, text(value)) if isinstance(value, str) else "未记录"
