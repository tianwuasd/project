"""Small, explicitly non-anatomical phantoms for end-to-end engineering tests."""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_demo_dataset(destination, seed=42):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.csv").exists():
        raise ValueError("Demo manifest already exists; choose a fresh output directory")
    rng = np.random.default_rng(seed)
    shape = (24, 24, 24)
    rows = []
    for i, split in enumerate(("train", "train", "val", "test")):
        label = np.zeros(shape, dtype=np.uint8)
        # Disjoint labeled cuboids, deliberately not a simulated clinical disease model.
        for c in range(1, 11):
            x = 2 + ((c - 1) % 3) * 7
            y = 2 + (((c - 1) // 3) % 2) * 10
            z = 3 + ((c - 1) // 6) * 11
            label[x : x + 4, y : y + 5, z : z + 6] = c
        initial = np.zeros_like(label)
        for c in (1, 2, 3, 4):
            initial[label == c] = c
        initial[label == 7] = 5
        initial[np.isin(label, [5, 6])] = 6
        initial[:, :, 10:] *= np.isin(initial[:, :, 10:], [1, 2, 3, 4, 5])
        image = rng.normal(-100, 20, shape).astype(np.float32)
        image[label > 0] += 350
        affine = np.diag([0.8, 1.0, 1.5, 1.0])
        names = {}
        for key, array in (("image", image), ("label", label), ("initial_label", initial)):
            names[key] = f"synthetic_{i:02d}_{key}.nii.gz"
            nib.save(nib.Nifti1Image(array, affine), destination / names[key])
        rows.append(
            {"case_id": f"synthetic_{i:02d}", "patient_id": f"phantom_{i:02d}", "split": split, **names}
        )
    with (destination / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "DATASET_INFO.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "clinical_data": False,
                "purpose": "Software verification only; no anatomical realism or performance interpretation",
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination / "manifest.csv"
