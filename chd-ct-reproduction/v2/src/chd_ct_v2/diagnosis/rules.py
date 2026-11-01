"""Candidate rule reference; missing evidence stays indeterminate."""

import math
import operator

import yaml

from ..paths import config_path

OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
}


def evaluate_node(node, features, thresholds, evidence):
    if "all" in node or "any" in node:
        key = "all" if "all" in node else "any"
        if not node[key]:
            raise ValueError("规则组不能为空")
        values = [evaluate_node(child, features, thresholds, evidence) for child in node[key]]
        if key == "all":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    key = node["feature"]
    actual = features.get(key)
    expected = thresholds.get(node["threshold"]) if "threshold" in node else node.get("value")
    if node.get("op") not in OPS:
        raise ValueError("无效规则运算符")
    missing = actual is None or expected is None
    if any(isinstance(x, (int, float)) and not math.isfinite(x) for x in (actual, expected)):
        missing = True
    result = None if missing else bool(OPS[node["op"]](actual, expected))
    evidence.append(
        {
            "feature": key,
            "observed": actual,
            "operator": node["op"],
            "threshold": expected,
            "satisfied": result,
        }
    )
    return result


def evaluate_rules(features, config):
    findings = {}
    for disease, node in config["rules"].items():
        evidence = []
        value = evaluate_node(node, features, config.get("thresholds", {}), evidence)
        findings[disease] = {
            "status": "indeterminate" if value is None else "positive" if value else "negative",
            "evidence": evidence,
        }
    for group in config.get("exclusive_groups", []):
        positive = [d for d in group if findings[d]["status"] == "positive"]
        if len(positive) > 1:
            for disease in positive:
                findings[disease].update(status="indeterminate", reason="conflicting_candidate_rules")
    return findings


def load_rules(path=None):
    from pathlib import Path

    return yaml.safe_load(Path(path or config_path("diagnosis-rules.yaml")).read_text(encoding="utf-8"))


def validate_evidence(features):
    from .features import RULE_KEYS

    if not isinstance(features, dict):
        raise ValueError("人工解剖证据 features 须为对象")
    for key, value in features.items():
        if key not in RULE_KEYS:
            raise ValueError("人工解剖证据含未知字段")
        if value is None:
            continue
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
            raise ValueError("人工解剖证据数值无效")
        if key.startswith("n_island_"):
            valid = not isinstance(value, bool) and value >= 0 and int(value) == value
        elif key == "ao_override_fraction":
            valid = not isinstance(value, bool) and 0 <= value <= 1
        elif key.startswith("posi_"):
            valid = not isinstance(value, bool)
        else:
            valid = value in (0, 1)
        if not valid:
            raise ValueError("人工解剖证据字段取值无效：" + key)
