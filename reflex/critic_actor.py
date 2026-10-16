"""MDC: Multi-turn Diagnostic Chain orchestration.

Three roles share the same LLM endpoint but with strictly disjoint contexts:

    BE image  --[Critic]--> Diagnosis JSON
    Diagnosis + parent code + retrieved skills --[Actor]--> child code
    parent BE summary + child BE summary --[Referee]--> accept / revert

The decoupling matters: empirically, mixing image-grounded diagnosis with
code-generation in a single prompt makes the LLM hedge and copy. Splitting
the two roles forces a structured, auditable trajectory we can log,
evaluate, and ablate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .llm_client import LLMClient
from .utils import extract_python_block, get_function_name, safe_json_loads


@dataclass
class Diagnosis:
    failure_modes: list[str] = field(default_factory=list)
    evidence_regions: list[str] = field(default_factory=list)
    root_cause: str = ""
    suggested_fix_type: str = "parameter_tweak"
    confidence: float = 0.0
    raw: str = ""

    @classmethod
    def from_json(cls, obj: dict, raw: str = "") -> "Diagnosis":
        # Some VLMs echo the requested JSON Schema wrapper and place the
        # diagnosis values directly under its ``properties`` key. Accept that
        # provider-specific serialization only when the nested values have
        # the expected result types; do not mistake an unchanged schema for a
        # diagnosis.
        nested = obj.get("properties")
        if (
            not isinstance(obj.get("root_cause"), str)
            and isinstance(nested, dict)
            and isinstance(nested.get("failure_modes"), list)
            and isinstance(nested.get("evidence_regions"), list)
            and isinstance(nested.get("root_cause"), str)
            and isinstance(nested.get("suggested_fix_type"), str)
            and isinstance(nested.get("confidence"), (int, float))
        ):
            obj = nested
        return cls(
            failure_modes=list(obj.get("failure_modes", [])),
            evidence_regions=list(obj.get("evidence_regions", [])),
            root_cause=str(obj.get("root_cause", "")),
            suggested_fix_type=str(obj.get("suggested_fix_type", "parameter_tweak")),
            confidence=float(obj.get("confidence", 0.0)),
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_modes": self.failure_modes,
            "evidence_regions": self.evidence_regions,
            "root_cause": self.root_cause,
            "suggested_fix_type": self.suggested_fix_type,
            "confidence": self.confidence,
        }


@dataclass
class RefereeVerdict:
    verdict: str = "neutral"
    reason: str = ""

    @classmethod
    def from_json(cls, obj: dict) -> "RefereeVerdict":
        v = str(obj.get("verdict", "neutral")).lower()
        if v not in {"accept", "revert", "neutral"}:
            v = "neutral"
        return cls(verdict=v, reason=str(obj.get("reason", "")))


SKILLS_USED_RE = re.compile(r"\[skills_used:\s*([^\]]+)\]", re.IGNORECASE)
CONCEPT_RE = re.compile(r"\{([^{}]{4,400})\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def extract_concept(resp: str, default: str = "") -> str:
    """Extract the {concept} line, ignoring braces inside code fences.

    Searching the raw response used to pick up dict literals / docstring
    braces from the generated code (e.g. '"iters": int, "scale": float').
    """
    prose = _CODE_FENCE_RE.sub("", resp)
    m = CONCEPT_RE.search(prose)
    return m.group(1).strip() if m else default


class MDCOrchestrator:
    """Run the Critic -> Actor -> (optional) Referee chain."""

    def __init__(self, llm: LLMClient, debug: bool = False):
        self.llm = llm
        self.debug = debug

    # ---------------- Critic ----------------
    def diagnose(
        self,
        task_description: str,
        be_image_b64: str | None,
        parent_score: float | None = None,
        parent_metrics: dict | None = None,
        use_image: bool = True,
    ) -> Diagnosis:
        user = prompts.critic_prompt(task_description, parent_score, parent_metrics)
        resp = self.llm.chat(
            user=user,
            system=prompts.CRITIC_SYSTEM,
            image_b64s=[be_image_b64] if use_image and be_image_b64 else None,
            temperature=0.3,
            usage_role="critic",
            force_vision_model=True,
        )
        obj = safe_json_loads(resp) or {}
        if self.debug:
            print(f"[Critic raw]\n{resp[:600]}\n[Parsed]\n{obj}")
        return Diagnosis.from_json(obj, raw=resp)

    # ---------------- Actor -----------------
    def patch(
        self,
        task_description: str,
        template_function: str,
        parent_code: str,
        parent_concept: str,
        diagnosis: Diagnosis,
        retrieved_skills: list[dict] | None = None,
        temperature: float = 0.85,
        parent_score: float | None = None,
        best_score: float | None = None,
        best_concept: str | None = None,
        stagnation_hint: str | None = None,
        parent_metrics: dict | None = None,
        failed_attempts: list[str] | None = None,
        numeric_history: str | None = None,
        diagnosis_source: str = "vision_critic",
    ) -> dict[str, Any]:
        user = prompts.actor_prompt(
            task_description=task_description,
            template_function=template_function,
            parent_code=parent_code,
            parent_concept=parent_concept,
            diagnosis=diagnosis.to_dict(),
            retrieved_skills=retrieved_skills or [],
            parent_score=parent_score,
            best_score=best_score,
            best_concept=best_concept,
            stagnation_hint=stagnation_hint,
            parent_metrics=parent_metrics,
            failed_attempts=failed_attempts,
            numeric_history=numeric_history,
            diagnosis_source=diagnosis_source,
        )
        resp = self.llm.chat(
            user=user,
            system=prompts.ACTOR_SYSTEM,
            temperature=temperature,
            usage_role="actor",
        )
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp)
        skills_used: list[str] = []
        m2 = SKILLS_USED_RE.search(resp)
        if m2:
            skills_used = [s.strip() for s in m2.group(1).split(",") if s.strip()]

        fn_name = get_function_name(code) if code else None
        if self.debug:
            print(
                f"[Actor] fn={fn_name}, concept={concept[:80]!r}, "
                f"skills_used={skills_used}"
            )
        return {
            "code": code,
            "concept": concept,
            "skills_used": skills_used,
            "function_name": fn_name,
            "raw": resp,
        }

    # ---------------- Referee ---------------
    def referee(
        self,
        task_description: str,
        parent_summary: dict,
        child_summary: dict,
        parent_score: float,
        child_score: float,
    ) -> RefereeVerdict:
        user = prompts.referee_prompt(
            task_description=task_description,
            parent_summary=parent_summary,
            child_summary=child_summary,
            parent_score=parent_score,
            child_score=child_score,
        )
        # Referee just emits strict JSON; thinking mode wastes tokens here.
        resp = self.llm.chat(
            user=user,
            system=prompts.REFEREE_SYSTEM,
            temperature=0.2,
            extra_body={"enable_thinking": False},
            usage_role="referee",
        )
        obj = safe_json_loads(resp) or {}
        return RefereeVerdict.from_json(obj)
