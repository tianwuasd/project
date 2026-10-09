"""Source-checkout and installed-package resource locations."""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = Path(__file__).resolve().parent.parent
IS_CHECKOUT = (SOURCE_ROOT / "pyproject.toml").is_file() and (SOURCE_ROOT / "src/chd_ct").resolve() == Path(
    __file__
).resolve().parent


def project_root():
    return SOURCE_ROOT if IS_CHECKOUT else Path.cwd()


def config_path(name):
    path = (
        (SOURCE_ROOT / "configs" / name)
        if IS_CHECKOUT
        else (Path(__file__).resolve().parent / "presets" / name)
    )
    if not path.is_file():
        raise FileNotFoundError("Missing bundled configuration: " + name)
    return path
