"""Tests for the steward agent's pure decision logic (no index/API needed)."""

import pytest

from steward_agent import (
    HIGH_PRIORITY_INCIDENT_TYPES,
    _build_dashboard_payload,
    _coerce_incident_json,
    _decide_verdict,
    _extract_features,
    run_steward_agent,
)


def test_coerce_rejects_non_object_json():
    with pytest.raises(ValueError):
        _coerce_incident_json("[1, 2, 3]")
    with pytest.raises(ValueError):
        _coerce_incident_json("not json")


def test_features_from_high_g_incident():
    features = _extract_features(
        {
            "lateral_g": 5.2,
            "braking_force": 0.9,
            "apex_clearance": 1.2,
            "incident_type": "collision",
            "incident_description": "Contact at turn entry",
        }
    )
    assert features["high_lateral_load"] is True
    assert features["hard_braking"] is True
    assert features["low_clearance"] is True
    assert features["collision_signal"] is True


def test_features_never_invent_collision_from_clean_data():
    features = _extract_features(
        {"lateral_g": 1.0, "incident_type": "normal_telemetry"}
    )
    assert features["collision_signal"] is False
    assert features["off_track_signal"] is False


def test_verdict_thresholds():
    assert _decide_verdict(_extract_features({"lateral_g": 5.0, "incident_type": "collision", "incident_description": "crash"}))[0] == "PENALTY"
    # Hard braking + tight clearance = moderate evidence, investigation not penalty.
    assert _decide_verdict(_extract_features({"lateral_g": 1.0, "incident_type": "review", "braking_force": 0.9, "apex_clearance": 1.5}))[0] == "INVESTIGATION"
    assert _decide_verdict(_extract_features({"lateral_g": 0.5, "incident_type": "review"}))[0] == "NO_FURTHER_ACTION"


def test_confidence_is_bounded_and_evidence_scaled():
    low = _decide_verdict(_extract_features({"lateral_g": 0.5, "incident_type": "review"}))[2]
    high = _decide_verdict(
        _extract_features({"lateral_g": 5.0, "incident_type": "collision", "incident_description": "crash"})
    )[2]
    assert 0.5 <= low < high <= 0.95


def test_normal_telemetry_short_circuits_without_index(tmp_path):
    """The no-action path must not require the FAISS index or any API key."""
    result = run_steward_agent(
        query="status ping",
        incident_json={"driver": "VER", "speed_kph": 210.0, "lateral_g": 1.2, "incident_type": "normal_telemetry"},
        index_dir=tmp_path / "does-not-exist.index",
    )
    assert result["ruling"] == "NO_FURTHER_ACTION"
    assert result["retrieved_articles"] == []


def test_unknown_incident_type_is_not_high_priority():
    assert "normal_telemetry" not in HIGH_PRIORITY_INCIDENT_TYPES
    result = run_steward_agent(
        query="status",
        incident_json={"driver": "VER", "incident_type": "formation_lap_delta"},
        index_dir="unused",
    )
    assert result["ruling"] == "NO_FURTHER_ACTION"


def test_dashboard_payload_single_shape():
    payload = _build_dashboard_payload(
        {"id": "x", "driver": "VER", "incident_type": "high_g_event", "lap": 58},
        "query",
        ruling="PENALTY",
        reasons=["r1"],
        confidence=0.9,
        article_cited="Article 33.3",
        rule_summary="summary",
        retrieved_docs=[],
        judicial_verdict="Judicial Verdict: PENALTY.",
    )
    assert payload["verdict"] == payload["ruling"] == "PENALTY"
    assert payload["driver"] == "VER"
    assert payload["retrieved_articles"] == []
    assert "confidence_score" in payload and "judicial_verdict" in payload


def test_index_mismatch_is_rejected(tmp_path, monkeypatch):
    """A truncated index must fail loudly with rebuild instructions."""
    import json

    import faiss
    import numpy as np

    index_file = tmp_path / "bad.index"
    index = faiss.IndexFlatL2(8)
    index.add(np.zeros((3, 8), dtype=np.float32))  # 3 vectors...
    faiss.write_index(index, str(index_file))
    (tmp_path / "bad_metadata.json").write_text(
        json.dumps({"texts": ["a"] * 5, "metadatas": [{}] * 5})  # ...5 texts
    )

    from steward_agent import _load_vector_store

    with pytest.raises(ValueError, match="Rebuild"):
        _load_vector_store(index_file)
