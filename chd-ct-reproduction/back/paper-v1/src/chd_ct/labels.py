"""Internal labels are a project convention, not an asserted author encoding."""

ANATOMY = ("BG", "LV", "RV", "LA", "RA", "AO", "PA", "MYO", "SVC", "IVC", "PV")
LABELS = {name: i for i, name in enumerate(ANATOMY)}
INITIAL = ("BG", "LV", "RV", "LA", "RA", "MYO", "INITIAL_VESSELS")
DISEASES = (
    "VSD",
    "SV",
    "ASD",
    "SA",
    "ToF",
    "DORV",
    "TGA",
    "TAC",
    "PuA",
    "IAA",
    "AAH",
    "CoA",
    "RAA",
    "APVC",
    "PS",
    "DSVC",
    "PDA",
)
BLOOD_IDS = (1, 2, 3, 4, 5, 6, 8, 9, 10)
STAGES = ("crop64", "crop128", "all64", "all128", "init64", "init128", "blood2d", "blood_lstm")
