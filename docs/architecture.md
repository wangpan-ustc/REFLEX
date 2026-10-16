# REFLEX architecture

This document summarizes the core components of REFLEX. Task-specific
evaluators and CLI options are described in [experiments.md](experiments.md).

Each task runner orchestrates the evolutionary loop using the modular
components in `reflex/`: the LLM client, Critic/Actor orchestrator, elitist
population, Skill Memory, and UCB1 operator bandit.

## High-level loop

```
init population (template + classical seeds, optionally)
repeat until budget exhausted or target reached:
    op  ← UCB1.select()          # bandit over {i1, m_diag, m_text, x_div, m_inspire}
    if stagnated for N steps:
        op ← forced exploration (m_inspire / i1 / x_div / m_diag burst)
    parent ← Population.tournament()          # x_div uses diverse_sample
    child  ← Operators[op](parent, skills=SkillMemory.retrieve(diagnosis))
    score, metrics, BE_image = Evaluator(child)   # code-hash eval cache
    if child crashed:      child ← Operators.repair(child, traceback)   # 1 retry
    if elite parent regressed (near miss, ≤ budget/10 times):
        also evaluate Operators.revise(parent, child)                   # Reflexion
    Population.add(child)                     # rejects duplicate code
    if child beat parent (ΔF > 0):
        SkillMemory.add_from_elite(child)
        Operators.note_failure(...) otherwise # per-parent failure memory
    SkillMemory.update_after_eval(child)      # promote / demote / evict
    UCB1.update(op, clip(ΔF)/Δmax)            # bounded reward, adaptive Δmax
    back-credit ancestors' operators with reward * 0.5^depth  # lineage credit
```

All learning is non-parametric: skill bank + population are the only
state that crosses iterations. No model weights are updated.

## Multi-turn Diagnostic Chain (MDC)

`reflex/critic_actor.py::MDCOrchestrator` exposes three calls:

1. `diagnose(parent, BE_image)` →
   ```json
   {
     "failure_modes":     ["..."],
     "evidence_regions":  ["..."],
     "root_cause":        "...",
     "suggested_fix_type":"parameter_tweak | structural_rewrite | reset",
     "confidence":        0.0–1.0
   }
   ```
   Vision LLM sees only the BE image + task prompt + parent metrics.
   It never writes code.

2. `patch(parent, diagnosis, skills, perf_context)` →
   ```python
   def choose_action(...): ...   # full child program
   # + a one-sentence concept describing the change
   ```
   Code LLM sees only text: parent code, diagnosis JSON, top-K skills,
   and a perf-context string ("best so far = ...; beat it"). It never
   sees the BE image.

3. `referee(parent_summary, child_summary)` →
   `accept | reject | neutral`. Cheap, optional. Disabled by default
   when the population already gates by score.

The split lets you swap the vision and code models independently
(e.g. Qwen-VL-Max + Qwen3.6-Plus, or GPT-4o-mini + Claude-3.5).

## Behavioral Evidence (BE)

Each task supplies its own BE renderer. The framework only cares that
the renderer returns a base64-encoded PNG. Existing tasks render:

- **Lunar Lander** — trajectory map, reward curve, action timeline,
  contact / crash markers.
- **Acrobot / Pendulum** — joint-angle traces, phase portrait, action
  timeline.
- **CCAA Antenna** — element layout coloured by amplitude, elevation
  AF cut with SLL/FNBW markers, per-element amplitude bar chart,
  per-ring statistics.

## Skill Memory

`reflex/memory.py::SkillMemory` is a JSON-backed bank of code snippets:

```json
{
  "id": "skill_alt_braking",
  "trigger": "altitude-gated braking when descending fast above pad",
  "snippet": "if vy < -0.7 and y > 0.05: action = 2",
  "utility": 1.4,
  "tags": ["lunar_lander", "control", "PD"]
}
```

Lifecycle:
- **Added** whenever a child beats its parent (ΔF > 0) and a
  skill-extraction prompt finds a self-contained snippet.
- **Promoted** (utility += min(|ΔF|, 1)) when a child re-using the skill
  beats its parent.
- **Demoted** (utility -= 0.5 · min(|ΔF|, 1)) when re-use regresses.
- **Merged** when two skills exceed Jaccard similarity 0.85 on the same
  target hook (LLM rewrites them into one).
- **Evicted** (archived) when utility falls below -1.0 after ≥ 3
  retrievals; capacity cap 256 drops lowest-utility first.

Retrieval uses BM25 over `trigger + tags`, with TF–IDF fallback if
`rank_bm25` is missing. The Critic's diagnosis becomes the query.

## UCB1 + lineage credit + stagnation burst

`reflex/bandit.py` runs a 5-arm UCB1 bandit over
`{i1, m_diag, m_text, x_div, m_inspire}`. The reward is the raw fitness
delta clipped to `[0, Δmax]` and normalized to `[0, 1]`, where `Δmax`
adapts online (running max of observed positive deltas), so one bandit
works across score scales (NWS ~1, Pendulum ~-1, antenna ~25). Failed
LLM calls / crashed evaluations count as reward 0. Selections handed to
concurrent workers are tracked as pending so parallel proposals spread
across arms.

**Lineage credit**: when a child improves on its parent, the operators
that produced its ancestors receive `reward * 0.5^depth` (up to depth 3).
This gives seeding operators credit when their descendants improve.

**`m_inspire` (cross-domain analogy)**: samples one of 10 inspiration
domains (quasicrystals, PLLs, optics apodization, difference sets,
central pattern generators, parametric pumping, annealing, portfolio
hedging, immune negative selection, counterpoint) and asks the Actor for
a concrete mechanistic mapping onto the task.

When the global best has not improved for `--stagnation-threshold`
steps, the loop forces an exploration burst:
`m_inspire + i1 + x_div + m_diag` (diagnosis escalated to
`structural_rewrite`, temperature raised to 1.0).

## Recovery loops and caching (v4.1)

- **Repair** (`Operators.repair`): a child whose evaluation crashed gets
  one self-debug LLM call (code + traceback) and a re-evaluation.
- **Reflective revision** (`Operators.revise`): when a mutation of an
  elite-frontier parent regresses by a small margin, the Actor
  immediately gets the parent/child score gap and metrics and produces
  one revision. Gated: near-miss regressions only (within 50% of the
  parent's score), at most `max(3, budget/10)` times per campaign —
  this limits the number of additional revision requests).
- **Diagnosis reuse**: one Critic call serves up to 2 Actor samples on
  the same parent (m_diag children are the highest-quality ones; the
  diagnosis of an unchanged parent does not change).
- **Eval cache + population dedup**: evaluations are cached by code
  hash; the population rejects exact-duplicate code.
- **Failure memory**: per-parent list of failed mutation concepts (with
  score deltas) injected into subsequent Actor prompts on that parent.
- **Informed init**: `i1` shows the top-3 skills by utility when the
  bank is non-empty.

## Evaluation and constant refinement

- `param_polish.py` provides optional Cross-Entropy Method (CEM) optimization
  over numeric constants in elite policies (`--polish-mode cem`, disabled by
  default).
- Reproducible evaluators reset policy state and per-episode Python/NumPy RNGs
  while preserving the outer search RNG state.

## Train-free guarantee

REFLEX never calls `optimizer.step()`, never fine-tunes the LLM,
and never trains an embedding model. The retriever is BM25 (lexical)
and the bandit is UCB1 (closed-form). Skill utility updates are
arithmetic counters.
