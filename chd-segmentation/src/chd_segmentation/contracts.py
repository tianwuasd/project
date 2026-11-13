from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    num_classes: int
    patch: tuple
    strides: tuple
    kernels: tuple = ((3, 3, 3),) * 6


@dataclass(frozen=True)
class StageSpec:
    name: str
    attempts: int
    batch: int
    loss_recipe: str


@dataclass(frozen=True)
class ModelBundle:
    checkpoint: str
    identity: str
    model_spec: dict


@dataclass
class RunState:
    attempts: int = 0
    updates: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image: str
    label: str | None
    spacing_xyz: tuple
    roles: tuple
    unknown_classes: tuple
    geometry_policy: str
    patient: str | None = None


@dataclass(frozen=True)
class PreparedPlan:
    path: str
    plan: dict
    identity: str
    scope: str
