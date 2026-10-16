from reflex.critic_actor import Diagnosis


def test_diagnosis_accepts_schema_echo_with_values_under_properties():
    obj = {
        "type": "object",
        "required": [
            "failure_modes",
            "evidence_regions",
            "root_cause",
            "suggested_fix_type",
            "confidence",
        ],
        "properties": {
            "failure_modes": ["no_tip_height_growth"],
            "evidence_regions": ["tip_height_trajectory_panel"],
            "root_cause": "The controller does not build momentum.",
            "suggested_fix_type": "control_law_change",
            "confidence": 0.9,
        },
    }

    diagnosis = Diagnosis.from_json(obj)

    assert diagnosis.failure_modes == ["no_tip_height_growth"]
    assert diagnosis.evidence_regions == ["tip_height_trajectory_panel"]
    assert diagnosis.root_cause == "The controller does not build momentum."
    assert diagnosis.suggested_fix_type == "control_law_change"
    assert diagnosis.confidence == 0.9


def test_diagnosis_does_not_treat_unchanged_schema_as_values():
    obj = {
        "type": "object",
        "properties": {
            "failure_modes": {"type": "array"},
            "evidence_regions": {"type": "array"},
            "root_cause": {"type": "string"},
            "suggested_fix_type": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    diagnosis = Diagnosis.from_json(obj)

    assert diagnosis.failure_modes == []
    assert diagnosis.root_cause == ""
    assert diagnosis.confidence == 0.0
