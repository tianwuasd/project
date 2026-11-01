"""Offline source manifest generation/verification; never installs or starts tasks."""

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = "V2_FILE_MANIFEST.json"
EXCLUDED = {
    ".git",
    ".venv",
    ".codex",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "data",
    "results",
    "runtime",
    "build",
    "dist",
    "back",
    "07_experiments",
    "08_results",
    "09_reports",
    "model_inputs",
    "model_weights",
}


def inventory(root):
    result = {}
    for file in sorted(root.rglob("*")):
        relative = file.relative_to(root)
        if any(p in EXCLUDED or p.endswith(".egg-info") for p in relative.parts):
            continue
        if relative.parts[:2] == ("foundation_engine", "04_data") and relative.as_posix() != "foundation_engine/04_data/manifests/README.md":
            continue
        if file.is_symlink():
            raise ValueError("source links are not allowed: " + str(relative))
        if not file.is_file() or file.name == MANIFEST or file.suffix in (".pyc", ".pyo"):
            continue
        result[relative.as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help="maintainer: refresh the manifest after review")
    args = parser.parse_args()
    root = args.root.resolve()
    actual = inventory(root)
    target = root / MANIFEST
    if args.write:
        target.write_text(
            json.dumps(
                {"panel": "2.1.0", "algorithm": "CT 0.24.0 + STU-Net-S", "files": actual}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        expected = json.loads(target.read_text(encoding="utf-8"))["files"]
        if actual != expected:
            changed = [
                name
                for name in sorted(actual.keys() | expected.keys())
                if actual.get(name) != expected.get(name)
            ]
            raise SystemExit("Source manifest mismatch: " + ", ".join(changed))
    print(f"CT V2 source verified: {len(actual)} files")


if __name__ == "__main__":
    main()
