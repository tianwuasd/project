"""Typed cache incompatibility; raw-data and system errors remain ordinary failures."""

import json

from ..recovery import file_hash, read_json
from .common import validate_prepared


class CacheReuseError(ValueError):
    pass


def read_cache(path):
    try:
        result = read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CacheReuseError("缓存记录损坏：" + str(path)) from error
    if not isinstance(result, dict):
        raise CacheReuseError("缓存记录不是对象：" + str(path))
    return result


def changed_fields(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(set(old) | set(new)):
            label = prefix + "." + key if prefix else key
            if key not in old or key not in new:
                result.append(label)
            else:
                result.extend(changed_fields(old[key], new[key], label))
        return result
    return [prefix] if old != new else []


def inspect_sources(output, pairs, complete, options):
    """Check all existing case records before changing any partial-cache metadata."""
    sources, verified = {}, set()
    if complete is not None:
        try:
            validate_prepared(output, complete)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise CacheReuseError("已完成缓存清单无效：" + str(error)) from error
    for case, image, label in pairs:
        sidecar = output / ".resume" / (case + ".json")
        if not sidecar.is_file():
            continue
        record = read_cache(sidecar)
        if (
            not {"sources", "row", "excluded", "cache_sha256"}.issubset(record)
            or not isinstance(record.get("row"), dict)
            or not isinstance(record.get("sources"), dict)
            or not isinstance(record.get("excluded"), bool)
            or (not record.get("excluded") and not {"source_sha256", "cache"}.issubset(record.get("row", {})))
        ):
            raise CacheReuseError("病例缓存记录不完整：" + case)
        if record["row"].get("case_id") != case:
            raise CacheReuseError("病例缓存编号缺失或不符：" + case)
        if not record["excluded"]:
            candidate = dict(
                options,
                status="prepared",
                cases=[dict(record["row"])],
                resolution="resized"
                if options["size"]
                else "resampled"
                if options.get("resampling")
                else "native",
            )
            try:
                validate_prepared(output, candidate, require_files=False)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise CacheReuseError("病例缓存记录无效：" + case + "：" + str(error)) from error
        sources[case] = dict(image=file_hash(image), label=file_hash(label) if label else None)
        if record["sources"] != sources[case]:
            raise CacheReuseError(case + " 原始影像或标签已改变；旧缓存不能沿用")
        if not record["excluded"] and complete is not None:
            cached = output / (case + ".npz")
            if not cached.is_file() or file_hash(cached) != record["cache_sha256"]:
                raise CacheReuseError("已完成缓存缺失或损坏：" + case)
            verified.add(case)
    if complete is not None:
        if not isinstance(complete.get("cases"), list) or not isinstance(complete.get("excluded"), list):
            raise CacheReuseError("已完成缓存清单缺少 cases/excluded")
        for row in complete["cases"]:
            if not isinstance(row, dict) or not isinstance(row.get("cache"), str):
                raise CacheReuseError("已完成缓存清单中的病例记录损坏")
            path = output / row["cache"]
            if path.resolve().parent != output.resolve() or not path.is_file():
                raise CacheReuseError("已完成缓存文件缺失或路径无效：" + row["cache"])
    return sources, verified
