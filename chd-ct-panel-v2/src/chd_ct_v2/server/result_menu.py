"""Two-level result navigation with bounded pages; no changes to running jobs."""

import shutil
import unicodedata
from pathlib import Path

from . import result_views, results
from .prompts import PanelBack, ask

PAGE_LINES = 20


def display_width(value):
    return sum(
        0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        for c in value
    )


def paginate(value, width=None):
    width = max(20, width or min(120, shutil.get_terminal_size((100, 24)).columns - 2))
    lines = []
    for original in value.splitlines():
        line, used = "", 0
        for character in original:
            size = display_width(character)
            if used + size > width:
                lines.append(line)
                line, used = "", 0
            line += character
            used += size
        lines.append(line)
    return [lines[i : i + PAGE_LINES] for i in range(0, len(lines), PAGE_LINES)] or [[]]


def section_menu(job, section):
    detail, page = "summary", 0
    pages = paginate(result_views.render(job, section, detail))
    while True:
        label = "简洁" if detail == "summary" else "详细"
        print(
            f"\n{results.title(job)} · {result_views.SECTIONS[section]} · {label} | 第 {page + 1}/{len(pages)} 页"
        )
        print("\n".join(pages[page]))
        print("s 简洁  d 详细  n 下一页  p 上一页  r 刷新  b/0 返回结果中心  q 退出面板")
        try:
            action = ask("请选择 [b]：").lower()
        except (PanelBack, KeyboardInterrupt, EOFError):
            return
        if action in ("", "0"):
            return
        if action in ("s", "d", "r"):
            if action != "r":
                detail = "summary" if action == "s" else "detailed"
            pages = paginate(result_views.render(job, section, detail))
            page = 0
        elif action == "n":
            page = min(page + 1, len(pages) - 1)
        elif action == "p":
            page = max(page - 1, 0)
        else:
            print("请选择页面中列出的操作。")


def browse_results(job):
    job = Path(job).expanduser().resolve()
    sections = list(result_views.SECTIONS)
    while True:
        result_views.header(job)
        for index, section in enumerate(sections, 1):
            print(f"{index} {result_views.SECTIONS[section]}")
        print("b/0 返回服务器面板  q 退出面板（后台任务继续）")
        try:
            action = ask("查看哪个部分 [b]：").lower()
        except (PanelBack, KeyboardInterrupt, EOFError):
            return
        if action in ("", "0"):
            return
        if action.isdigit() and 1 <= int(action) <= len(sections):
            section_menu(job, sections[int(action) - 1])
        else:
            print("请输入列表内的编号。")
