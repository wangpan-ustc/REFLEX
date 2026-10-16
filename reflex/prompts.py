"""Prompt templates for REFLEX (MDC: Critic / Actor / Referee + Memory ops).

All prompts are train-free; the LLM is never fine-tuned. They are designed
so the *Critic* never sees code-generation requests and the *Actor* never
sees the raw image, breaking the diagnosis-repair entanglement of MLES.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Critic role: looks at BE image, emits structured JSON. NO code allowed.
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = (
    "You are the Critic in a Reflective Evolutionary Search system. "
    "You analyse Behavioral Evidence (BE) images of executed code policies "
    "and emit a strictly structured JSON diagnosis. "
    "You MUST NOT propose code, patches, or implementation details. "
    "You diagnose only."
)

CRITIC_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "failure_modes",
        "evidence_regions",
        "root_cause",
        "suggested_fix_type",
        "confidence",
    ],
    "properties": {
        "failure_modes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short snake_case tags naming the failure you "
            "actually observe in THIS task's evidence (e.g. a control task "
            "might yield 'no_amplitude_growth' or 'late_braking'; an antenna "
            "task 'high_first_sidelobe'). Derive tags from the image — do "
            "NOT copy these examples verbatim.",
        },
        "evidence_regions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which panel of the BE image triggered the conclusion, "
            "e.g. 'trajectory_panel:turn_3', 'reward_curve:t=120-150'.",
        },
        "root_cause": {
            "type": "string",
            "description": "One sentence explaining the underlying logical cause.",
        },
        "suggested_fix_type": {
            "type": "string",
            "enum": [
                "parameter_tweak",
                "control_law_change",
                "perception_change",
                "state_estimation_change",
                "structural_rewrite",
            ],
        },
        "confidence": {
            "type": "number",
            "description": "Self-rated confidence in [0,1].",
        },
    },
}


def critic_prompt(
    task_description: str,
    parent_score: float | None,
    parent_metrics: dict | None = None,
) -> str:
    schema = json.dumps(CRITIC_JSON_SCHEMA, indent=2)
    score_line = (
        f"The current policy achieved a score of {parent_score:.4f}.\n"
        if parent_score is not None
        else ""
    )
    if parent_metrics:
        compact = {
            k: v for k, v in parent_metrics.items()
            if isinstance(v, (int, float, str, bool))
        }
        if compact:
            score_line += (
                "Structured evaluation metrics (ground your diagnosis in these "
                "numbers as well as the image):\n"
                f"{json.dumps(compact, indent=2, ensure_ascii=False, default=str)}\n"
            )
    return f"""## Task context
{task_description}

{score_line}You are shown a Behavioral Evidence image with multiple panels
(trajectory, action timeline, score curve, code-line attribution).

## Your job
Diagnose what is wrong with the current policy by analysing the image only.
Do NOT write code. Do NOT propose specific numeric values. Only diagnose.

## Output
Return EXACTLY one JSON object that conforms to this schema:

```json
{schema}
```

The JSON must be self-contained and parseable. No prose before or after.
"""


# ---------------------------------------------------------------------------
# Actor role: reads diagnosis JSON + parent code, emits a patched function.
# Never sees the BE image directly (decoupled).
# ---------------------------------------------------------------------------
ACTOR_SYSTEM = (
    "You are the Actor in a Reflective Evolutionary Search system. "
    "You receive a structured diagnosis (JSON) and a parent function. "
    "You produce ONE improved Python function that addresses the diagnosis. "
    "You do NOT see the raw image: trust the diagnosis."
)


def actor_prompt(
    task_description: str,
    template_function: str,
    parent_code: str,
    parent_concept: str,
    diagnosis: dict,
    retrieved_skills: list[dict] | None = None,
    parent_score: float | None = None,
    best_score: float | None = None,
    best_concept: str | None = None,
    stagnation_hint: str | None = None,
    parent_metrics: dict | None = None,
    failed_attempts: list[str] | None = None,
    numeric_history: str | None = None,
    diagnosis_source: str = "vision_critic",
) -> str:
    diag_str = json.dumps(diagnosis, indent=2, ensure_ascii=False)
    numeric_block = ""
    if numeric_history:
        numeric_block = (
            "\n## Design ↔ score history (numeric feedback)\n"
            "Use this table like an optimizer would: infer which parameter\n"
            "directions raise the score, and propose a design that moves along\n"
            "the promising directions instead of guessing blindly.\n"
            f"{numeric_history.strip()}\n"
        )
    metrics_block = ""
    if parent_metrics:
        compact = {
            k: v for k, v in parent_metrics.items()
            if isinstance(v, (int, float, str, bool))
        }
        if compact:
            metrics_block = (
                "\n## Parent evaluation metrics\n"
                f"{json.dumps(compact, indent=2, ensure_ascii=False, default=str)}\n"
            )
    failed_block = ""
    if failed_attempts:
        failed_block = (
            "\n## Mutations already tried on this parent that FAILED\n"
            "Each line is `(score delta) concept`. Do NOT propose the same or a\n"
            "trivially similar idea again — pick a genuinely different angle.\n"
        )
        for fa in failed_attempts[-8:]:
            failed_block += f"- {fa}\n"
    target_block = ""
    if best_score is not None:
        target_block = (
            f"\n## Performance target\n"
            f"Parent score: {parent_score if parent_score is not None else 'n/a'}\n"
            f"Current population best score: {best_score:.4f}\n"
        )
        if best_concept:
            target_block += f"Best-so-far concept: {best_concept}\n"
        target_block += (
            "Your child MUST aim to **beat** the current best score.\n"
        )
        if stagnation_hint:
            target_block += (
                "\n## Anti-stagnation rules (READ CAREFULLY)\n"
                f"{stagnation_hint.strip()}\n"
            )
    skill_block = ""
    if retrieved_skills:
        skill_block = "\n## Reusable memory objects retrieved for this diagnosis\n"
        for i, sk in enumerate(retrieved_skills, 1):
            skill_block += (
                f"\n### Skill {i} (id={sk['id']}, utility={sk['utility']:.2f})\n"
                f"**When to use**: {sk['trigger']}\n"
            )
            if sk.get("artifact_type") == "textual_strategy":
                skill_block += (
                    "**Textual strategy reflection**: "
                    f"{sk['snippet']}\n"
                )
            else:
                skill_block += (
                    "**Executable program fragment**:\n"
                    f"```python\n{sk['snippet']}\n```\n"
                )
        skill_block += (
            "\nUse these memory objects only if they directly address the "
            "diagnosis. You MUST emit `[skills_used: id1,id2]` when any object "
            "influences the child, or `[skills_used: none]` otherwise, so "
            "downstream fitness can update utility without ambiguous credit.\n"
        )
    if diagnosis_source == "none":
        diagnosis_heading = "## Neutral mutation context (no Critic call)"
        diagnosis_instruction = (
            "No Critic was used in this condition. Treat the context below as "
            "a neutral marker and reason from the parent code and metrics."
        )
    elif diagnosis_source == "text_critic":
        diagnosis_heading = "## Text-only Critic diagnosis (no BE image)"
        diagnosis_instruction = (
            "The Critic saw only scalar/text metrics, not the BE image."
        )
    else:
        diagnosis_heading = "## Vision Critic diagnosis"
        diagnosis_instruction = (
            "The Actor cannot see the image; use the structured diagnosis."
        )
    return f"""## Task
{task_description}

## Parent policy
Concept: {parent_concept}
```python
{parent_code}
```
{metrics_block}
{diagnosis_heading}
{diagnosis_instruction}
```json
{diag_str}
```
{failed_block}{numeric_block}{target_block}{skill_block}
## Edit discipline (IMPORTANT)
The parent already earns its current score — its working structure is
valuable. Unless the diagnosis explicitly calls for a structural_rewrite,
make a TARGETED modification: keep the control/design logic that works and
change only what the diagnosis identifies as broken. Wholesale rewrites of
working parents almost always regress.

## Your output
1. One sentence describing the new core idea, in curly braces: {{your idea}}.
2. A required `[skills_used: id1,id2]` line if memory influenced the child,
   otherwise `[skills_used: none]`.
3. The complete improved Python function inside a ```python ...``` block,
   conforming to this template signature:

```python
{template_function}
```

Constraints:
- The function MUST be self-contained and importable.
- Only standard scientific libs (numpy, math, cv2 if already used).
- No print statements, no global state.
"""


# ---------------------------------------------------------------------------
# Shared output-format instructions for generation prompts (init / crossover /
# text mutation). Spelled out explicitly so the LLM never mimics placeholder
# text like "```python full function```" back at us.
# ---------------------------------------------------------------------------
OUTPUT_FORMAT_BLOCK = (
    "## Output format (follow EXACTLY)\n"
    "Line 1: your one-sentence design concept wrapped in curly braces, e.g.\n"
    "{energy-pumping swing-up with PD stabilization near upright}\n"
    "Then: ONE fenced code block starting with ```python containing the\n"
    "complete, runnable function (same signature as the template).\n"
    "No other prose, no placeholder text.\n"
)


# ---------------------------------------------------------------------------
# Cross-domain inspiration operator (m_inspire): map a mechanism from an
# unrelated field onto the current problem to escape same-domain local optima.
# ---------------------------------------------------------------------------
INSPIRATION_DOMAINS: list[tuple[str, str]] = [
    ("quasicrystals & aperiodic tilings",
     "Penrose tilings and quasicrystals diffract with sharp peaks but no "
     "periodic grating lobes. Map: replace periodic/uniform structure with "
     "deterministic aperiodic sequences (Fibonacci, golden-angle, low-"
     "discrepancy) to break coherent interference or resonance."),
    ("phase-locked loops & synchronization",
     "A PLL locks onto a signal by feeding back the phase error to a "
     "controlled oscillator. Map: estimate the system's natural phase/"
     "frequency online and inject your control/design in phase (or "
     "deliberately in anti-phase) with it."),
    ("optics apodization & holography",
     "Telescope pupils use smooth apodization windows (Gaussian, Kaiser, "
     "prolate spheroidal) to trade resolution for contrast; holograms encode "
     "structure in phase, not amplitude. Map: move information from "
     "amplitude into phase, or use an optimal-window taper family."),
    ("number theory & difference sets",
     "Costas arrays and perfect difference sets have flat autocorrelation "
     "sidelobes. Map: choose discrete positions/timings so pairwise "
     "differences are maximally non-repeating."),
    ("biological central pattern generators",
     "Animal locomotion uses coupled oscillators with phase offsets, not "
     "feedback on position error. Map: drive the system with a rhythmic, "
     "phase-offset pattern and only modulate amplitude/frequency slowly."),
    ("resonance & parametric pumping",
     "A child on a swing pumps energy by modulating at twice the natural "
     "frequency. Map: identify the dominant natural frequency and act "
     "periodically at a resonant (or parametric, 2x) relationship to it."),
    ("simulated annealing & metallurgy",
     "Metals reach low-energy crystal states via slow cooling with occasional "
     "reheating. Map: schedule your aggressiveness — large exploratory "
     "changes early or when stuck, tiny refinements near a good solution."),
    ("portfolio theory & risk hedging",
     "Diversification minimizes variance for a target return by combining "
     "anti-correlated assets. Map: combine two mechanisms whose failure "
     "modes are anti-correlated, weighting them by conditions."),
    ("immune systems & negative selection",
     "Immune systems maintain diversity and explicitly avoid previously seen "
     "failures. Map: enumerate the failure signatures observed so far and "
     "construct a solution explicitly orthogonal to all of them."),
    ("music theory & counterpoint",
     "Consonance comes from small-integer frequency ratios; counterpoint "
     "interleaves independent voices that never collide. Map: set your "
     "sub-components' periods/spacings at small-integer ratios, or "
     "deliberately at irrational ratios to avoid coincidence."),
]


def inspire_prompt(
    task_description: str,
    template_function: str,
    parent_code: str,
    parent_concept: str,
    parent_score: float,
    domain_name: str,
    domain_hint: str,
    best_score: float | None = None,
    numeric_history: str | None = None,
    failed_attempts: list[str] | None = None,
) -> str:
    failed_block = ""
    if failed_attempts:
        failed_block = (
            "\n## Already-failed directions (do not repeat)\n"
            + "".join(f"- {fa}\n" for fa in failed_attempts[-6:])
        )
    numeric_block = ""
    if numeric_history:
        numeric_block = (
            f"\n## Design ↔ score history\n{numeric_history.strip()}\n"
        )
    best_line = (
        f"Population best score: {best_score:.4f}. Your goal is to BEAT it.\n"
        if best_score is not None else ""
    )
    return f"""## Task
{task_description}

## Current parent policy (score={parent_score:.4f})
Concept: {parent_concept}
```python
{parent_code}
```
{failed_block}{numeric_block}
## Cross-domain inspiration: {domain_name}
{domain_hint}

{best_line}
## Your job
1. Think about what structural mechanism from "{domain_name}" maps onto THIS
   problem. The mapping must be concrete and mechanistic, not a metaphor.
2. Design a policy that implements that mechanism. It may reuse working
   pieces of the parent, but the inspired mechanism must be load-bearing —
   not a cosmetic comment.

Template signature:

```python
{template_function}
```

{OUTPUT_FORMAT_BLOCK}"""


REPAIR_SYSTEM = (
    "You are a senior Python engineer. You fix a broken policy function so "
    "it runs, changing as little of the intended logic as possible."
)


def revise_prompt(
    task_description: str,
    template_function: str,
    parent_code: str,
    parent_score: float,
    child_code: str,
    child_score: float,
    child_metrics: dict | None = None,
    diagnosis: dict | None = None,
) -> str:
    metrics_block = ""
    if child_metrics:
        compact = {k: v for k, v in child_metrics.items()
                   if isinstance(v, (int, float, str, bool))}
        if compact:
            metrics_block = (
                "\nChild evaluation metrics:\n"
                f"{json.dumps(compact, indent=2, ensure_ascii=False, default=str)}\n"
            )
    diag_block = ""
    if diagnosis:
        diag_block = (
            "\nOriginal diagnosis the child tried to address:\n"
            f"{json.dumps(diagnosis, indent=2, ensure_ascii=False)}\n"
        )
    return f"""## Task
{task_description}

## What happened
A mutation of the parent policy REGRESSED instead of improving:
- Parent score: {parent_score:.4f}
- Child score:  {child_score:.4f}  (worse)

## Parent policy (working baseline)
```python
{parent_code}
```

## Child policy (your previous attempt — regressed)
```python
{child_code}
```
{metrics_block}{diag_block}
## Your job
Produce ONE revised policy. Diagnose which specific change in the child hurt
performance, undo or fix exactly that, and keep whatever parts of the child's
idea are sound. Do not simply return the parent unchanged.

Template signature:

```python
{template_function}
```

{OUTPUT_FORMAT_BLOCK}"""


def repair_prompt(
    task_description: str,
    broken_code: str,
    error: str,
    template_function: str,
) -> str:
    return f"""## Task
{task_description}

## Broken policy code
```python
{broken_code}
```

## Error when executing it
```
{error.strip()[:1500]}
```

Fix the code so it executes correctly, preserving the intended design as
much as possible. Common causes: syntax errors, wrong function signature,
undefined names, returning the wrong type/shape.

The function must conform to this template signature:

```python
{template_function}
```

{OUTPUT_FORMAT_BLOCK}"""


# ---------------------------------------------------------------------------
# Referee role: compares parent vs child BE summaries, decides accept/revert.
# Used as a sanity gate on top of the scalar fitness, especially when scores
# are close (within ε).
# ---------------------------------------------------------------------------
REFEREE_SYSTEM = (
    "You are the Referee. You compare two Behavioral Evidence summaries "
    "(parent vs child) and decide whether the child should be accepted "
    "into the population. You output a strict JSON verdict."
)

REFEREE_JSON_SCHEMA = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revert", "neutral"]},
        "reason": {"type": "string"},
    },
}


def referee_prompt(
    task_description: str,
    parent_summary: dict,
    child_summary: dict,
    parent_score: float,
    child_score: float,
) -> str:
    schema = json.dumps(REFEREE_JSON_SCHEMA, indent=2)
    return f"""## Task
{task_description}

## Parent (score = {parent_score:.4f})
{json.dumps(parent_summary, indent=2, ensure_ascii=False)}

## Child  (score = {child_score:.4f})
{json.dumps(child_summary, indent=2, ensure_ascii=False)}

The scalar scores are close. Decide whether the qualitative behaviour
of the child is genuinely better (accept), worse (revert), or
indistinguishable (neutral). Reply with JSON conforming to:

```json
{schema}
```
"""


# ---------------------------------------------------------------------------
# Memory ops: skill abstraction (when promoting an elite into a Skill) and
# skill merging (when two skills overlap too much).
# ---------------------------------------------------------------------------

ABSTRACT_SKILL_SYSTEM = (
    "You are a code librarian. Given an elite policy and the diagnosis it "
    "cured, you extract ONE reusable, minimal code snippet plus a textual "
    "trigger ('when to use'). You output strict JSON."
)


def abstract_skill_prompt(
    task_description: str, code: str, concept: str, diagnosis: dict | None
) -> str:
    diag_str = (
        json.dumps(diagnosis, indent=2, ensure_ascii=False)
        if diagnosis
        else "<<no diagnosis available>>"
    )
    schema = {
        "id_hint": "<3-6 word slug>",
        "trigger": "<one sentence describing when this skill is useful>",
        "snippet": "<a SHORT, self-contained python snippet, <=25 lines>",
        "tags": ["<tag1>", "<tag2>"],
    }
    return f"""## Task
{task_description}

## Elite policy concept
{concept}

## Elite policy code
```python
{code}
```

## Diagnosis it cured
```json
{diag_str}
```

Extract ONE reusable skill that captures the essential trick of this
elite. Keep the snippet minimal (<=25 lines), focused, and self-contained.
Return strict JSON conforming to:

```json
{json.dumps(schema, indent=2)}
```
"""


MERGE_SKILL_SYSTEM = (
    "You merge two overlapping skill snippets into one cleaner snippet "
    "without losing functionality. You output strict JSON."
)


def merge_skill_prompt(skill_a: dict, skill_b: dict) -> str:
    schema = {
        "id_hint": "<merged slug>",
        "trigger": "<merged trigger>",
        "snippet": "<merged snippet>",
        "tags": ["<merged tags>"],
    }
    return f"""## Skill A (utility={skill_a['utility']:.2f})
trigger: {skill_a['trigger']}
```python
{skill_a['snippet']}
```

## Skill B (utility={skill_b['utility']:.2f})
trigger: {skill_b['trigger']}
```python
{skill_b['snippet']}
```

Produce ONE merged skill that subsumes both. Prefer A's structure if a
choice has to be made. Return strict JSON:

```json
{json.dumps(schema, indent=2)}
```
"""
