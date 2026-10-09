import numpy as np
import pytest

from chd_ct.diagnosis import evaluate_rules
from chd_ct.features import extract_features
from chd_ct.features.graph import vessel_graph, widest_path


def test_sheared_affine_and_nonfinite_contact_are_rejected():
    labels = np.zeros((6, 6, 6), dtype=np.uint8)
    affine = np.eye(4)
    affine[0, 1] = 0.3
    with pytest.raises(ValueError, match="orthogonal"):
        extract_features(labels, (1, np.sqrt(1.09), 1), affine=affine)
    with pytest.raises(ValueError, match="finite"):
        extract_features(labels, (1, 1, 1), contact_mm=float("nan"))


def test_missing_anatomy_is_unknown():
    labels = np.zeros((12, 12, 12), dtype=np.uint8)
    features = extract_features(labels, (1, 1, 2))
    assert features["conn_LV_RV"] is None
    assert features["n_island_AO"] is None
    assert features["ratio_PA_to_AO"] is None


def test_contact_and_physical_volume():
    labels = np.zeros((12, 12, 12), dtype=np.uint8)
    labels[1:4, 1:4, 1:4] = 1
    labels[4:7, 1:4, 1:4] = 2
    features = extract_features(labels, (1, 2, 3), contact_mm=1.0)
    assert features["conn_LV_RV"] is True
    assert features["volume_LV_mm3"] == 162
    features = extract_features(labels, (2, 2, 3), contact_mm=1.0)
    assert features["conn_LV_RV"] is False


def test_graph_disconnected_and_radius():
    mask = np.zeros((15, 15, 25), dtype=bool)
    mask[5:10, 5:10, 2:23] = True
    graph = vessel_graph(mask, (1, 1, 2))
    assert len(graph) > 1
    radii = [d["radius_mm"] for _, d in graph.nodes(data=True)]
    assert max(radii) == 3.0
    graph.add_node((-1, -1, -1), radius_mm=1)
    assert widest_path(graph, next(iter(graph)), (-1, -1, -1)) == []


def test_three_valued_rules_and_groups():
    config = {
        "rules": {
            "VSD": {"all": [{"feature": "connection", "op": "eq", "value": True}]},
            "SV": {"all": [{"feature": "ratio", "op": "gt", "threshold": "sv"}]},
        },
        "thresholds": {"sv": None},
        "exclusive_groups": [["VSD", "SV"]],
    }
    out = evaluate_rules({"connection": True, "ratio": 0.9}, config)
    assert out["findings"]["VSD"]["status"] == "positive"
    assert out["findings"]["SV"]["status"] == "indeterminate"
    config["thresholds"]["sv"] = 0.5
    out = evaluate_rules({"connection": True, "ratio": 0.9}, config)
    assert out["findings"]["VSD"]["status"] == "indeterminate"
    assert out["findings"]["SV"]["status"] == "indeterminate"
    assert out["conflicts"] == [["VSD", "SV"]]
