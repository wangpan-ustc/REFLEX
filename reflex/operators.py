"""REFLEX evolutionary operators.

Thin wrappers around MDCOrchestrator + SkillMemory. Each operator returns
a fresh `Individual` (uncompiled, unscored). The core loop is responsible
for evaluation and population insertion.
"""
from __future__ import annotations

import base64
import io
import random
import threading
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any

from . import prompts
from .critic_actor import Diagnosis, MDCOrchestrator, extract_concept
from .memory import SkillMemory
from .population import Individual
from .utils import extract_python_block


def _apply_be_noise(b64_png: str, level: str, rng: random.Random) -> str:
    """Corrupt a base64 PNG BE image for the Critic stress test.

    clean:    no modification
    mild:     Gaussian blur sigma=1.0
    moderate: Gaussian blur sigma=3.0 + 10% random occlusion patches
    severe:   pixelate 4x + 30% occlusion
    """
    if level == "clean" or not b64_png:
        return b64_png
    try:
        from PIL import Image, ImageFilter, ImageDraw
    except ImportError:
        return b64_png
    try:
        raw = base64.b64decode(b64_png)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if level == "mild":
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif level == "moderate":
            img = img.filter(ImageFilter.GaussianBlur(radius=3.0))
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            n_patches = max(1, int(w * h * 0.10 / (40 * 40)))
            for _ in range(n_patches):
                x0 = rng.randint(0, max(0, w - 40))
                y0 = rng.randint(0, max(0, h - 40))
                pw = rng.randint(20, 60)
                ph = rng.randint(20, 60)
                draw.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(0, 0, 0))
            img = draw_img
        elif level == "severe":
            small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
            img = small.resize((w, h), Image.NEAREST)
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            n_patches = max(2, int(w * h * 0.30 / (40 * 40)))
            for _ in range(n_patches):
                x0 = rng.randint(0, max(0, w - 60))
                y0 = rng.randint(0, max(0, h - 60))
                pw = rng.randint(30, 80)
                ph = rng.randint(30, 80)
                draw.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(0, 0, 0))
            img = draw_img
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return b64_png


class Operators:
    def __init__(
        self,
        task_description: str,
        template_function_str: str,
        mdc: MDCOrchestrator,
        skills: SkillMemory,
        rng: random.Random | None = None,
        diagnostic_mode: str = "full",
        noise_level: str = "clean",
    ):
        self.task_description = task_description
        self.template_function_str = template_function_str
        self.mdc = mdc
        self.skills = skills
        self.rng = rng or random.Random()
        if diagnostic_mode not in {"full", "actor_only", "text_critic"}:
            raise ValueError(f"Unsupported diagnostic_mode: {diagnostic_mode}")
        self.diagnostic_mode = diagnostic_mode
        self.noise_level = noise_level
        # Best-so-far context, set by the runner before each operator call.
        self.best_score: float | None = None
        self.best_concept: str | None = None
        # Consecutive steps without a new best (set by the runner); drives
        # temperature escalation and fix-type escalation.
        self.plateau_count: int = 0
        # Optional task-specific guidance injected into the Actor prompt
        # when the search plateaus near the best score.
        self.stagnation_hint: str | None = None
        # Optional design ↔ score history table (set by the runner each
        # round) giving the Actor numeric-optimizer-style feedback.
        self.numeric_feedback: str | None = None
        # Relative margin defining "parent is near the best" for fix-type
        # escalation; scale-free so it works for NWS (~1) and antenna (~25).
        self.escalation_rel_margin: float = 0.02
        # Failure memory: parent iid -> recent failed mutation summaries
        # ("(Δ-0.42) tried X"). Injected into Actor prompts so the LLM stops
        # re-proposing ideas that already regressed on the same parent.
        self._failures: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=8))
        self._fail_lock = threading.Lock()
        # Diagnosis cache: parent iid -> (Diagnosis, use_count). One Critic
        # call is amortized over up to `diag_reuse` Actor samples, since
        # m_diag children are the highest-quality ones and the diagnosis of
        # an unchanged parent does not change between calls.
        self.diag_reuse: int = 2
        self._diag_cache: dict[str, tuple[Any, int]] = {}

    # -------------------- failure memory --------------------
    def note_failure(self, parent_iid: str, concept: str, delta: float) -> None:
        if not parent_iid or not concept:
            return
        with self._fail_lock:
            self._failures[parent_iid].append(f"(Δ{delta:+.3f}) {concept[:140]}")

    def failed_attempts(self, parent_iid: str | None) -> list[str]:
        if not parent_iid:
            return []
        with self._fail_lock:
            return list(self._failures.get(parent_iid, ()))

    # -------------------- init (zero-shot / informed) --------------------
    def init_zero_shot(self) -> Individual:
        # Informed init: when the skill bank has proven snippets, show the
        # top ones. Blind zero-shot inits in warm-started runs score far
        # below the population and waste budget.
        skill_block = ""
        try:
            top = self.skills.top_by_utility(3)
        except Exception:
            top = []
        if top:
            skill_block = (
                "\n## Proven skills from previous campaigns (adapt freely)\n"
            )
            for sk in top:
                skill_block += (
                    f"\n### {sk.id} (utility={sk.utility:.2f})\n"
                    f"When to use: {sk.trigger}\n"
                )
                if sk.artifact_type == "textual_strategy":
                    skill_block += f"Strategy reflection: {sk.snippet}\n"
                else:
                    skill_block += f"```python\n{sk.snippet}\n```\n"
        best_block = ""
        if self.best_score is not None and self.best_concept:
            best_block = (
                f"\nCurrent population best: score={self.best_score:.4f}, "
                f"concept: {self.best_concept}\n"
                "Design something DIFFERENT from it, not a copy.\n"
            )
        user = (
            f"## Task\n{self.task_description}\n"
            f"{skill_block}{best_block}\n"
            f"Design ONE policy from scratch. Template signature:\n\n"
            f"```python\n{self.template_function_str}\n```\n\n"
            f"{prompts.OUTPUT_FORMAT_BLOCK}"
        )
        resp = self.mdc.llm.chat(user=user, temperature=0.9, usage_role="actor")
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp)
        return Individual(code=code, concept=concept, operator="i1", parent_id=None)

    # -------------------- diagnostic mutation (MDC) --------------------
    def mutate_diagnostic(self, parent: Individual) -> Individual:
        if self.diagnostic_mode == "full" and not parent.image_b64:
            return self.mutate_text_only(parent)
        with self._fail_lock:
            cached = self._diag_cache.get(parent.iid)
        if cached is not None and cached[1] < self.diag_reuse:
            # Copy: escalation below mutates the object in place, and the
            # cached original must stay pristine for the next reuse.
            diag = replace(cached[0])
            with self._fail_lock:
                self._diag_cache[parent.iid] = (cached[0], cached[1] + 1)
        elif self.diagnostic_mode == "actor_only":
            compact_metrics = {
                k: v
                for k, v in (parent.be_summary or {}).items()
                if isinstance(v, (int, float, str, bool))
            }
            diag = Diagnosis(
                failure_modes=[],
                evidence_regions=[],
                root_cause=(
                    "No Critic call in this ablation. Use the parent code, "
                    f"concept, and metrics directly: {compact_metrics}"
                ),
                suggested_fix_type="parameter_tweak",
                confidence=0.0,
            )
        else:
            be_img = parent.image_b64
            if self.noise_level != "clean" and be_img and self.diagnostic_mode == "full":
                be_img = _apply_be_noise(be_img, self.noise_level, self.rng)
            diag = self.mdc.diagnose(
                task_description=self.task_description,
                be_image_b64=be_img,
                parent_score=parent.score,
                parent_metrics=parent.be_summary,
                use_image=self.diagnostic_mode == "full",
            )
            with self._fail_lock:
                self._diag_cache[parent.iid] = (diag, 1)
        retrieval_query = " ".join(
            diag.failure_modes + [diag.root_cause, diag.suggested_fix_type]
        )
        retrieved = self.skills.retrieve(retrieval_query, k=3)
        retrieved_dicts = [
            {
                "id": s.id,
                "trigger": s.trigger,
                "snippet": s.snippet,
                "utility": s.utility,
                "artifact_type": s.artifact_type,
            }
            for s in retrieved
        ]
        # If the search has plateaued and the parent already sits near the
        # best, escalate the fix type to force a structural rewrite rather
        # than yet another parameter tweak. The margin is relative to the
        # best score's magnitude, so it is task-scale independent.
        near_best = (
            self.best_score is not None
            and parent.score
            >= self.best_score - self.escalation_rel_margin * max(abs(self.best_score), 1.0)
        )
        if (
            near_best
            and self.plateau_count >= 2
            and diag.suggested_fix_type == "parameter_tweak"
        ):
            diag.suggested_fix_type = "structural_rewrite"
            diag.root_cause += (
                " (NOTE: parameter tweaks have not improved the best score "
                "recently. A structural rewrite of the policy — a genuinely "
                "different control/design strategy, not new constants — is "
                "now required.)"
            )

        temp = 1.0 if self.plateau_count >= 2 else 0.85

        out = self.mdc.patch(
            task_description=self.task_description,
            template_function=self.template_function_str,
            parent_code=parent.code,
            parent_concept=parent.concept,
            diagnosis=diag,
            retrieved_skills=retrieved_dicts,
            temperature=temp,
            parent_score=parent.score,
            best_score=self.best_score,
            best_concept=self.best_concept,
            stagnation_hint=(
                self.stagnation_hint if (near_best or self.plateau_count >= 2) else None
            ),
            parent_metrics=parent.be_summary,
            failed_attempts=self.failed_attempts(parent.iid),
            numeric_history=self.numeric_feedback,
            diagnosis_source=(
                "none"
                if self.diagnostic_mode == "actor_only"
                else self.diagnostic_mode
            ),
        )
        retrieved_ids = {s.id for s in retrieved}
        return Individual(
            code=out["code"],
            concept=out["concept"] or parent.concept,
            operator="m_diag",
            parent_id=parent.iid,
            diagnosis=diag.to_dict(),
            skill_ids_used=[sid for sid in out["skills_used"] if sid in retrieved_ids],
        )

    # -------------------- text-only mutation (fallback) ---------------
    def mutate_text_only(self, parent: Individual) -> Individual:
        failed = self.failed_attempts(parent.iid)
        failed_block = ""
        if failed:
            failed_block = (
                "## Mutations already tried on this parent that FAILED "
                "(do NOT repeat these ideas)\n"
                + "".join(f"- {fa}\n" for fa in failed[-8:])
                + "\n"
            )
        numeric_block = ""
        if self.numeric_feedback:
            numeric_block = (
                "## Design ↔ score history (numeric feedback; use it like an "
                "optimizer)\n"
                f"{self.numeric_feedback.strip()}\n\n"
            )
        user = (
            f"## Task\n{self.task_description}\n\n"
            f"## Parent (score={parent.score:.4f})\n"
            f"Concept: {parent.concept}\n"
            f"```python\n{parent.code}\n```\n\n"
            f"{failed_block}{numeric_block}"
            "Improve this policy with a TARGETED change: keep the parts that "
            "make it earn its current score, modify what limits it. "
            "Template signature:\n\n"
            f"```python\n{self.template_function_str}\n```\n\n"
            f"{prompts.OUTPUT_FORMAT_BLOCK}"
        )
        resp = self.mdc.llm.chat(user=user, temperature=0.85, usage_role="actor")
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp, default=parent.concept)
        return Individual(
            code=code, concept=concept, operator="m_text", parent_id=parent.iid
        )

    # -------------------- diversity crossover -------------------------
    def crossover_diversity(self, parents: list[Individual]) -> Individual:
        if not parents:
            return self.init_zero_shot()
        block = ""
        for i, p in enumerate(parents, 1):
            block += (
                f"### Parent {i} (score={p.score:.4f})\n"
                f"Concept: {p.concept}\n```python\n{p.code}\n```\n\n"
            )
        user = (
            f"## Task\n{self.task_description}\n\n"
            f"You are given {len(parents)} parent policies. Create ONE new policy "
            f"that combines the strongest mechanisms of the parents (or explores "
            f"a direction they all miss). It must be a complete, runnable "
            f"function.\n\n{block}"
            f"Template signature:\n\n"
            f"```python\n{self.template_function_str}\n```\n\n"
            f"{prompts.OUTPUT_FORMAT_BLOCK}"
        )
        resp = self.mdc.llm.chat(user=user, temperature=1.0, usage_role="actor")
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp)
        return Individual(
            code=code,
            concept=concept,
            operator="x_div",
            parent_id=",".join(p.iid for p in parents),
        )

    # -------------------- cross-domain inspiration ---------------------
    def mutate_inspire(self, parent: Individual) -> Individual:
        """Map a mechanism from a random unrelated domain onto the task.

        Escapes same-domain local optima that plain mutation keeps circling:
        every m_diag/m_text child stays in the parent's conceptual family,
        whereas m_inspire forces a structurally foreign mechanism in.
        """
        domain_name, domain_hint = self.rng.choice(prompts.INSPIRATION_DOMAINS)
        user = prompts.inspire_prompt(
            task_description=self.task_description,
            template_function=self.template_function_str,
            parent_code=parent.code,
            parent_concept=parent.concept,
            parent_score=parent.score,
            domain_name=domain_name,
            domain_hint=domain_hint,
            best_score=self.best_score,
            numeric_history=self.numeric_feedback,
            failed_attempts=self.failed_attempts(parent.iid),
        )
        resp = self.mdc.llm.chat(user=user, temperature=1.0, usage_role="actor")
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp)
        child = Individual(
            code=code,
            concept=f"[{domain_name}] {concept}" if concept else concept,
            operator="m_inspire",
            parent_id=parent.iid,
        )
        return child

    # -------------------- reflective revision (child regressed) --------
    def revise(self, parent: Individual, child: Individual) -> Individual:
        """One-shot Reflexion-style revision after a regressing child.

        Feeds the parent/child score gap and the child's metrics back to
        the Actor so it can undo the harmful part of its edit immediately,
        instead of waiting for the failure-memory hint on some future
        mutation of the same parent.
        """
        user = prompts.revise_prompt(
            task_description=self.task_description,
            template_function=self.template_function_str,
            parent_code=parent.code,
            parent_score=parent.score,
            child_code=child.code,
            child_score=child.score,
            child_metrics=child.be_summary,
            diagnosis=child.diagnosis,
        )
        resp = self.mdc.llm.chat(user=user, temperature=0.6, usage_role="revision")
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp, default=child.concept)
        return Individual(
            code=code or child.code,
            concept=concept,
            operator=child.operator + "_rev",
            parent_id=child.parent_id,
            diagnosis=child.diagnosis,
            skill_ids_used=list(child.skill_ids_used),
        )

    # -------------------- error repair (self-debug) --------------------
    def repair(self, child: Individual, error: str) -> Individual:
        """One-shot self-debug: fix a child whose evaluation crashed."""
        user = prompts.repair_prompt(
            task_description=self.task_description,
            broken_code=child.code,
            error=error,
            template_function=self.template_function_str,
        )
        resp = self.mdc.llm.chat(
            user=user,
            system=prompts.REPAIR_SYSTEM,
            temperature=0.3,
            usage_role="repair",
        )
        code = extract_python_block(resp) or ""
        concept = extract_concept(resp, default=child.concept)
        fixed = Individual(
            code=code or child.code,
            concept=concept,
            operator=child.operator + "_fix",
            parent_id=child.parent_id,
            diagnosis=child.diagnosis,
            skill_ids_used=list(child.skill_ids_used),
        )
        return fixed
