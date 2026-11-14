from pathlib import Path

from .artifacts.store import read_json


def compatible_code(source, target):
    if source == target:
        return True
    audit = read_json(Path(__file__).with_name("code_compatibility.json"))
    if audit["schema"] != 1:
        raise ValueError("代码兼容审计格式无效")
    # 只接收经过审查的完整源码摘要对，不按版本号或任意文件子集放行。
    return any(edge["source"] == source and edge["target"] == target for edge in audit["edges"])
