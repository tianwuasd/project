"""Model-independent task graph for localization, training and validation policy."""

from .config import STAGES

TASKS = ("crop64", "crop128", "roi", "all64", "all128", "blood2d", "blood_lstm", "calibration")
DEPENDENCIES = {
    "roi": ["crop64", "crop128"],
    "all64": ["roi"],
    "all128": ["roi"],
    "blood2d": ["roi"],
    "blood_lstm": ["roi", "blood2d"],
    "calibration": list(STAGES),
}


def execute(directory, rows, stage, config, output, mode, manifest_hash, resume=False):
    if stage == "roi":
        from .roi_inputs import prepare_rois

        return prepare_rois(directory, rows, config, output, resume)
    if stage == "calibration":
        from .calibration import calibrate

        return calibrate(directory, rows, config, output, mode, resume)
    from .train import train_stage

    return train_stage(directory, rows, stage, config, output, mode, manifest_hash, resume)
