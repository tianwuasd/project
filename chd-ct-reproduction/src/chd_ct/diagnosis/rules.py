"""Three-valued, inspectable rules. No eval(), learned probabilities, or invented thresholds."""

import math
import operator

from ..labels import DISEASES

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
            raise ValueError("Empty rule group")
        values = [evaluate_node(child, features, thresholds, evidence) for child in node[key]]
        if key == "all":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    key = node["feature"]
    actual = features.get(key)
    expected = thresholds.get(node["threshold"]) if "threshold" in node else node.get("value")
    if node.get("op") not in OPS:
        raise ValueError("Unknown rule operator")
    unknown = actual is None or expected is None
    if isinstance(actual, (int, float)) and not math.isfinite(actual):
        unknown = True
    if isinstance(expected, (int, float)) and not math.isfinite(expected):
        unknown = True
    value = None if unknown else bool(OPS[node["op"]](actual, expected))
    evidence.append(
        {"feature": key, "actual": actual, "op": node["op"], "expected": expected, "passed": value}
    )
    return value


def evaluate_rules(features, config):
    findings = {}
    for disease in DISEASES:
        evidence = []
        node = config.get("rules", {}).get(disease)
        value = evaluate_node(node, features, config.get("thresholds", {}), evidence) if node else None
        findings[disease] = {
            "status": "indeterminate" if value is None else "positive" if value else "negative",
            "evidence": evidence,
        }
    conflicts = []
    for group in config.get("exclusive_groups", []):
        positives = [key for key in group if findings[key]["status"] == "positive"]
        if len(positives) > 1:
            conflicts.append(positives)
            for key in positives:
                findings[key]["status"] = "indeterminate"
                findings[key]["reason"] = "conflicting positive rules within an exclusive disease group"
    return {
        "status": "research_only_unvalidated",
        "rule_set": config.get("name", "custom"),
        "findings": findings,
        "conflicts": conflicts,
        "note": "Candidate rule results, not clinical diagnoses. Missing evidence stays indeterminate.",
    }
