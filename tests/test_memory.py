"""Smoke tests for SkillMemory self-evolution.

Run with:  python -m pytest REFLEX/tests/test_memory.py -q
or         python REFLEX/tests/test_memory.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from reflex.memory import Skill, SkillMemory  # noqa: E402


class FakeLLM:
    """Deterministic fake LLM for tests (avoids any network call)."""

    class _U:
        calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        failed = 0

    usage = _U()

    def __init__(self):
        self.responses: list[str] = []

    def stats(self):
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failed": 0}

    def chat(self, *a, **k) -> str:
        if self.responses:
            return self.responses.pop(0)
        return '{"id_hint":"steer-clip","trigger":"clip steering when off-track","snippet":"action[0] = max(min(action[0], 1), -1)","tags":["clip","steering"]}'


def test_add_retrieve_promote_demote():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "skills.json")
        llm = FakeLLM()
        mem = SkillMemory(store_path=path, llm=llm, tau_add=0.0, debug=False)

        sk = mem.add_from_elite(
            task_description="control task",
            elite_code="def f(): pass",
            elite_concept="clip steering",
            delta_fitness=2.0,
            diagnosis={
                "failure_modes": ["oscillation"],
                "root_cause": "steering too aggressive",
                "suggested_fix_type": "parameter_tweak",
                "evidence_regions": [],
                "confidence": 0.7,
            },
            target_hook="choose_action",
        )
        assert sk is not None
        assert len(mem) == 1

        retrieved = mem.retrieve("oscillation steering", k=2)
        assert retrieved and retrieved[0].id == sk.id
        assert retrieved[0].n_retrieved == 1

        mem.update_after_eval([sk.id], helped=True, magnitude=0.5)
        assert mem._skills[sk.id].utility > 1.0
        assert mem._skills[sk.id].n_helped == 1

        mem.update_after_eval([sk.id], helped=False, magnitude=2.0)
        assert mem._skills[sk.id].n_hurt == 1

        mem.save()
        mem2 = SkillMemory(store_path=path, llm=llm)
        assert sk.id in mem2._skills
        print("[ok] add/retrieve/promote/demote/persist")


def test_eviction():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "skills.json")
        llm = FakeLLM()
        mem = SkillMemory(
            store_path=path,
            llm=llm,
            tau_add=0.0,
            tau_evict_utility=0.0,
            evict_min_retrievals=2,
        )
        sk = mem.add_from_elite(
            task_description="t",
            elite_code="def f(): pass",
            elite_concept="c",
            delta_fitness=1.0,
        )
        assert sk is not None
        for _ in range(3):
            mem.retrieve("clip steering", k=1)
        mem.update_after_eval([sk.id], helped=False, magnitude=4.0)
        mem.update_after_eval([sk.id], helped=False, magnitude=4.0)
        # utility should be deeply negative; eviction should fire
        assert sk.id not in mem._skills or mem._skills[sk.id].archived
        print("[ok] eviction")


def test_merge():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "skills.json")
        llm = FakeLLM()
        # First call abstracts skill A, second abstracts skill B with very
        # similar text (should trigger Jaccard merge), third call (merge LLM)
        # returns the merged JSON.
        llm.responses = [
            '{"id_hint":"steer-clip","trigger":"clip steering when off-track","snippet":"action[0] = max(-1, min(1, action[0]))","tags":["clip","steering"]}',
            '{"id_hint":"steer-clamp","trigger":"clip steering when off-track aggressively","snippet":"action[0] = max(-1, min(1, action[0]))","tags":["clip","steering"]}',
            '{"id_hint":"steer-clip-merged","trigger":"clip steering bounds in [-1,1]","snippet":"action[0] = max(-1, min(1, action[0]))","tags":["clip","steering","bounds"]}',
        ]
        mem = SkillMemory(store_path=path, llm=llm, tau_merge=0.5)
        a = mem.add_from_elite(
            "t", "def f(): pass", "concept A", delta_fitness=1.0,
            target_hook="choose_action",
        )
        b = mem.add_from_elite(
            "t", "def g(): pass", "concept B", delta_fitness=1.0,
            target_hook="choose_action",
        )
        # After insertion of B, merge logic should have collapsed the bank.
        active = mem.all()
        assert len(active) == 1, f"expected 1 active skill after merge, got {len(active)}"
        assert active[0].id not in {a.id, b.id}
        print("[ok] merge")


def test_retrieval_ablation_modes():
    with tempfile.TemporaryDirectory() as tmp:
        skills = {
            "phase": Skill(
                id="phase",
                trigger="phase aligned energy pumping",
                snippet="torque = sign(velocity)",
                utility=3.0,
            ),
            "brake": Skill(
                id="brake",
                trigger="brake near upright",
                snippet="torque = -velocity",
                utility=1.0,
            ),
        }

        none = SkillMemory(
            os.path.join(tmp, "none.json"),
            retrieval_mode="none",
        )
        none._skills = dict(skills)
        none._rebuild_index()
        assert none.retrieve("phase energy", k=1) == []
        assert none.top_by_utility(1) == []
        assert none.retrieval_trace()[-1]["mode"] == "none"

        constant = SkillMemory(
            os.path.join(tmp, "constant.json"),
            retrieval_mode="bm25_constant",
        )
        constant._skills = dict(skills)
        constant._rebuild_index()
        picked = constant.retrieve("phase energy", k=1)
        assert picked and picked[0].id == "phase"
        before = picked[0].utility
        constant.update_after_eval(["phase"], helped=True, magnitude=1.0)
        assert constant._skills["phase"].utility == before

        random_a = SkillMemory(
            os.path.join(tmp, "random-a.json"),
            retrieval_mode="random",
            retrieval_seed=17,
        )
        random_b = SkillMemory(
            os.path.join(tmp, "random-b.json"),
            retrieval_mode="random",
            retrieval_seed=17,
        )
        random_a._skills = dict(skills)
        random_b._skills = dict(skills)
        random_a._rebuild_index()
        random_b._rebuild_index()
        assert [s.id for s in random_a.retrieve("anything", k=2)] == [
            s.id for s in random_b.retrieve("anything", k=2)
        ]

        dense = SkillMemory(
            os.path.join(tmp, "dense-lsa.json"),
            retrieval_mode="dense_lsa",
        )
        dense._skills = dict(skills)
        dense._rebuild_index()
        dense_picked = dense.retrieve("phase energy", k=1)
        assert dense_picked and dense_picked[0].id == "phase"
        trace = dense.retrieval_trace()[-1]
        assert trace["mode"] == "dense_lsa"
        assert trace["picked"][0]["score"] is not None


def test_memory_object_metadata_and_frozen_bank():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "textual.json")
        payload = {
            "version": 1,
            "skills": [
                Skill(
                    id="phase",
                    trigger="phase aligned energy pumping",
                    snippet="Pump in phase with angular velocity.",
                    tags=["phase", "energy"],
                    artifact_type="textual_strategy",
                    retrieval_text="phase aligned energy pumping phase energy",
                ).to_dict()
            ],
        }
        import json

        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        mem = SkillMemory(path, llm=FakeLLM(), frozen=True)
        picked = mem.retrieve("phase energy", k=1)
        assert picked[0].artifact_type == "textual_strategy"
        assert picked[0].doc_text() == "phase aligned energy pumping phase energy"
        assert mem.add_from_elite(
            "task",
            "def f(): pass",
            "concept",
            delta_fitness=1.0,
        ) is None
        assert mem.stats()["frozen"] is True


if __name__ == "__main__":
    test_add_retrieve_promote_demote()
    test_eviction()
    test_merge()
    print("All memory tests passed.")
