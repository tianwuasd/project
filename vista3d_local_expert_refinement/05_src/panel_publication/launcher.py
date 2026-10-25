from pathlib import Path
from server_data import write_json

ROOT=Path(__file__).resolve().parents[2]


def personal_path(value):
    return Path(value).expanduser().resolve()
