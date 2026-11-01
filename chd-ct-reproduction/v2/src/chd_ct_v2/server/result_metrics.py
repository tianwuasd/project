"""Summaries of saved scalar metrics only; never read images or recompute Dice."""

from .result_io import finite, mapping, rows


def mean(values):
    present = [v for v in values if finite(v)]
    return sum(present) / len(present) if present else None


def summary(saved, cases, labels):
    """Preserve explicit nulls and recorded values; only fill absent summary fields."""
    result = dict(saved)
    derived = False
    complete = "cases_evaluated" not in saved or (
        type(saved.get("cases_evaluated")) is int and saved["cases_evaluated"] == len(cases)
    )
    if not complete:
        result["dice_by_class"] = dict(mapping(saved.get("dice_by_class")))
        return result, False
    if cases:
        if "cases_evaluated" not in result:
            result["cases_evaluated"] = len(cases)
            derived = True
        if "mean_case_foreground_dice" not in result and all(
            "mean_foreground_dice" in r
            and (r["mean_foreground_dice"] is None or finite(r["mean_foreground_dice"]))
            for r in cases
        ):
            result["mean_case_foreground_dice"] = mean(r.get("mean_foreground_dice") for r in cases)
            derived = True
    by_class = dict(mapping(saved.get("dice_by_class")))
    for code in labels:
        if (
            code not in by_class
            and cases
            and all(
                code in mapping(r.get("dice")) and (r["dice"][code] is None or finite(r["dice"][code]))
                for r in cases
            )
        ):
            by_class[code] = mean(mapping(r.get("dice")).get(code) for r in cases)
            derived = True
    result["dice_by_class"] = by_class
    return result, derived


def fold_index(row):
    indices = row.get("fold_indices")
    if (
        row.get("inference_protocol") == "out_of_fold"
        and isinstance(indices, list)
        and len(indices) == 1
        and type(indices[0]) is int
        and indices[0] in range(3)
    ):
        return indices[0]
    return None


def fold_details(report, labels):
    cases = rows(report.get("cases"))
    saved = mapping(report.get("folds"))
    groups = {f"fold_{i}": [r for r in cases if fold_index(r) == i] for i in range(3)}
    protocols = report.get("protocols")
    has_oof = (isinstance(protocols, list) and "out_of_fold" in protocols) or any(groups.values())
    if not saved and not has_oof:
        return []
    details = []
    for name, members in groups.items():
        original = mapping(saved.get(name))
        value, derived = summary(original, members, labels)
        details.append((name, value, members, derived))
    return details


def denominator(cases, code, expected):
    """A partial/malformed detail list cannot establish the summary's denominator."""
    if (
        not cases
        or type(expected) is not int
        or len(cases) != expected
        or any(code not in mapping(r.get("dice")) for r in cases)
    ):
        return "未记录"
    values = [mapping(r.get("dice"))[code] for r in cases]
    if any(v is not None and not finite(v) for v in values):
        return "未记录"
    return f"{sum(finite(v) for v in values)}/{len(cases)}"
