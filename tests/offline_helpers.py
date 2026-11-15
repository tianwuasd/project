from pathlib import Path

from chd_segmentation.preprocessing.offline import preprocess
from chd_segmentation.preprocessing.prepare import prepare


def prepare_offline(config, output):
    output = Path(output)
    if config["preprocessing"]["root"] is None:
        config["preprocessing"]["root"] = str(Path(config["work_root"]).parent / "offline-data")
    source = output.with_name("source-" + output.name)
    prepare(config, source)
    return preprocess(config, source, output)
