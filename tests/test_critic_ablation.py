from pathlib import Path

from reflex.critic_actor import MDCOrchestrator
from reflex.memory import SkillMemory
from reflex.operators import Operators
from reflex.population import Individual


class _RoleFakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("usage_role") == "critic":
            return (
                '{"failure_modes":["phase lag"],'
                '"evidence_regions":["phase portrait"],'
                '"root_cause":"late switching",'
                '"suggested_fix_type":"logic_change","confidence":0.8}'
            )
        return (
            "{targeted phase switch}\n"
            "```python\n"
            "def choose_action(x):\n"
            "    return 1\n"
            "```"
        )


def _operator(tmp_path: Path, mode: str):
    llm = _RoleFakeLLM()
    memory = SkillMemory(str(tmp_path / f"{mode}.json"), llm=llm)
    mdc = MDCOrchestrator(llm)
    ops = Operators(
        "test task",
        "def choose_action(x):\n    return 0",
        mdc,
        memory,
        diagnostic_mode=mode,
    )
    parent = Individual(
        code="def choose_action(x):\n    return 0",
        concept="baseline",
        score=-1.0,
        image_b64="aW1hZ2U=",
        be_summary={"mean_steps": 100.0},
    )
    return llm, ops, parent


def test_actor_only_uses_no_critic_call(tmp_path):
    llm, ops, parent = _operator(tmp_path, "actor_only")
    child = ops.mutate_diagnostic(parent)

    assert child.code
    assert [c.get("usage_role") for c in llm.calls] == ["actor"]
    assert child.diagnosis["confidence"] == 0.0
    assert "No Critic call" in child.diagnosis["root_cause"]


def test_text_critic_keeps_critic_role_without_image(tmp_path):
    llm, ops, parent = _operator(tmp_path, "text_critic")
    child = ops.mutate_diagnostic(parent)

    assert child.code
    assert [c.get("usage_role") for c in llm.calls] == ["critic", "actor"]
    critic_call = llm.calls[0]
    assert critic_call["image_b64s"] is None
    assert critic_call["force_vision_model"] is True
