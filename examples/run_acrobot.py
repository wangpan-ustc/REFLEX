"""REFLEX runner for the MLES Acrobot benchmark.

Mirrors `run_lander.py` exactly — same concurrent + Critic/Actor + SkillMemory
pipeline — but evaluates on Acrobot-v1 with a custom multi-panel BE image
(joint angles, phase portrait, action timeline).

Requires: `pip install gymnasium` (no box2d needed for Acrobot).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.acrobot_evaluator import evaluate_acrobot_code
from examples.acrobot_reproducible import evaluate_acrobot_code_reproducible
from examples.acrobot_template import ACROBOT_TASK_DESCRIPTION, ACROBOT_TEMPLATE
from reflex.bandit import UCB1
from reflex.critic_actor import MDCOrchestrator
from reflex.llm_client import LLMCallBudgetExceeded, LLMClient
from reflex.memory import SkillMemory
from reflex.operators import Operators
from reflex.param_polish import polish_constants_cem
from reflex.population import Individual, Population
from reflex.randomness import environment_seeds, resolve_seed, seed_argument
from reflex.utils import dump_json, ensure_dir, short_hash


def _record(ind: Individual) -> dict[str, Any]:
    return {
        "iid": ind.iid,
        "score": ind.score,
        "concept": ind.concept,
        "operator": ind.operator,
        "gen": ind.gen,
        "parent_id": ind.parent_id,
        "skill_ids_used": list(ind.skill_ids_used),
        "be_summary": ind.be_summary,
        "code": ind.code,
        "error": ind.extra.get("error", ""),
        "generated_call_start": ind.extra.get("generated_call_start"),
        "generated_call_end": ind.extra.get("generated_call_end"),
        "committed_at_call": ind.extra.get("committed_at_call"),
    }


def run_campaign(
    args: argparse.Namespace,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    if getattr(args, "reproducible_evaluation", False) and args.num_workers != 1:
        raise ValueError("Policy-RNG evaluation v2 requires one sampling worker")
    print(
        f"[REFLEX-Acrobot] start | budget={args.budget} pop={args.pop_size} "
        f"workers={args.num_workers} "
        f"condition={args.diagnostic_mode} llm_call_budget={args.llm_call_budget}",
        flush=True,
    )
    args.seed = resolve_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    ensure_dir(args.log_dir)
    training_seeds = environment_seeds(args.training_seeds, count=3)
    print(f"[REFLEX-Acrobot] seed={args.seed} training_seeds={list(training_seeds)}", flush=True)

    llm = llm_client or LLMClient(
        host=args.llm_host, key=args.llm_key, model=args.llm_model,
        vision_model=args.vision_model,
        vision_host=args.vision_host,
        vision_key=args.vision_key,
        extra_body={
            **(
                {"enable_thinking": True}
                if args.enable_thinking
                else ({"enable_thinking": False} if args.disable_thinking else {})
            ),
            **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}),
            **({"verbosity": args.verbosity} if args.verbosity else {}),
        },
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        max_tokens=args.max_tokens,
        call_budget=args.llm_call_budget,
        trust_env=not args.llm_no_env_proxy,
        provider=args.llm_provider,
        debug=args.debug,
    )
    if llm_client is not None:
        llm.call_budget = args.llm_call_budget
    skills = SkillMemory(
        args.skill_bank,
        llm=llm,
        retrieval_mode=args.retrieval_mode,
        retrieval_seed=args.seed,
        frozen=args.freeze_skill_bank,
        debug=args.debug,
    )
    mdc = MDCOrchestrator(llm, debug=args.debug)
    ops = Operators(
        ACROBOT_TASK_DESCRIPTION,
        ACROBOT_TEMPLATE,
        mdc,
        skills,
        rng=rng,
        diagnostic_mode=args.diagnostic_mode,
        noise_level=getattr(args, "noise_level", "clean"),
    )
    pop = Population(capacity=args.pop_size, rng=rng)
    history: list[dict[str, Any]] = []
    bandit: dict[str, list[float]] = {}
    ucb = UCB1(["m_diag", "m_text", "x_div", "i1", "m_inspire"], c=args.ucb_c)
    n_evaluated = 0
    n_failed = 0
    budget_exhausted = False
    state_lock = threading.Lock()
    # Evaluation cache: code hash -> result tuple.
    eval_cache: dict[str, tuple] = {}
    cache_lock = threading.Lock()
    evaluator_trace: list[dict[str, Any]] = []

    def evaluate_code(code: str, origin: str):
        if getattr(args, "reproducible_evaluation", False):
            result = evaluate_acrobot_code_reproducible(
                code, training_seeds, render_be=origin != "inner_cem",
            )
        else:
            result = evaluate_acrobot_code(code, training_seeds)
        score, _, error, _ = result
        with cache_lock:
            evaluator_trace.append({"index": len(evaluator_trace) + 1, "origin": origin,
                                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                                    "score": float(score) if np.isfinite(score) else None,
                                    "error": bool(error), "at_attempted_call": llm.usage.attempted})
        return result

    def mutate_parent(parent: Individual, requested: str, *, forced: bool = False):
        del forced
        return (
            ops.mutate_diagnostic(parent)
            if requested == "m_diag"
            else ops.mutate_text_only(parent)
        )

    def evaluate_only(ind: Individual) -> Individual:
        ind.iid = ind.iid or f"g{ind.gen:03d}_{short_hash(ind.code, 8)}"
        key = short_hash(ind.code, 16)
        with cache_lock:
            hit = eval_cache.get(key)
        if hit is not None:
            score, metrics, err, img_b64 = hit
            ind.extra["cache_hit"] = True
        else:
            score, metrics, err, img_b64 = evaluate_code(ind.code, "outer")
            with cache_lock:
                eval_cache[key] = (score, metrics, err, img_b64)
        ind.score = score
        ind.be_summary = metrics
        if err:
            ind.extra["error"] = err
        elif img_b64:
            ind.image_b64 = img_b64
        return ind

    def commit_ind(ind: Individual) -> Individual:
        nonlocal n_evaluated, n_failed
        with state_lock:
            n_evaluated += 1
            if ind.extra.get("error"):
                n_failed += 1
            pop.add(ind)
            ind.extra["committed_at_call"] = llm.usage.attempted
            history.append(_record(ind))
            best = pop.best()
            best_s = best.score if best else float("-inf")
            mr = (ind.be_summary or {}).get("solve_rate", float("nan"))
            sr = (ind.be_summary or {}).get("mean_steps_to_goal", float("nan"))
            flag = "ERR" if ind.extra.get("error") else "OK "
            try:
                calls_n, n_skills = llm.usage.attempted, len(skills)
            except Exception:
                calls_n, n_skills = 0, 0
            print(
                f"[{n_evaluated:>3d}/{args.budget}] op={ind.operator:<10s} {flag} | "
                f"score={ind.score:+.4f} solve_rate={mr:.2f} steps={sr:.0f} | "
                f"best={best_s:+.4f} | pop={len(pop)} skills={n_skills} calls={calls_n}",
                flush=True,
            )
        return ind

    def score_ind(ind: Individual) -> Individual:
        return commit_ind(evaluate_only(ind))

    template_ind = Individual(
        code=ACROBOT_TEMPLATE,
        concept="MLES default Acrobot template",
        operator="template", gen=0,
    )
    score_ind(template_ind)

    n_workers = max(1, int(args.num_workers))
    pop_fill_target = max(2, args.pop_size // 2)
    _t = pop.best().score if pop.best() else 0.0
    min_keep_score = _t - max(1.0, 0.5 * abs(_t))

    while (
        len(pop) < pop_fill_target
        and n_evaluated < args.budget
        and not budget_exhausted
        and (llm.remaining_calls() is None or llm.remaining_calls() > 0)
    ):
        with state_lock:
            best = pop.best()
            ops.best_score = best.score if best else None
            ops.best_concept = best.concept if best else None
        if best is None:
            break
        if best.score >= args.target_score and not args.run_to_call_budget:
            print(f"[REFLEX-Acrobot] target {args.target_score} reached during pop fill — stopping.", flush=True)
            break
        batch_n = min(n_workers, pop_fill_target - len(pop), args.budget - n_evaluated)
        if batch_n <= 0:
            break
        def _make_child(_=None, _parent=best):
            call_start = llm.usage.attempted + 1
            try:
                use_diag = bool(_parent.image_b64) and (rng.random() < 0.6)
                if use_diag:
                    c = mutate_parent(_parent, "m_diag")
                    c.operator = "i1_diag"
                else:
                    c = mutate_parent(_parent, "m_text")
                    c.operator = "i1_text"
                c.extra["generated_call_start"] = call_start
                c.extra["generated_call_end"] = llm.usage.attempted
                return c, ""
            except LLMCallBudgetExceeded:
                return None, "__LLM_CALL_BUDGET_EXHAUSTED__"
            except Exception as e:
                return None, str(e)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_make_child) for _ in range(batch_n)]
            results = [f.result() for f in as_completed(futs)]
        for child, err_str in results:
            if child is None:
                if err_str == "__LLM_CALL_BUDGET_EXHAUSTED__":
                    budget_exhausted = True
                    print("[REFLEX-Acrobot] LLM call budget exhausted during population fill.", flush=True)
                    continue
                with state_lock:
                    n_failed += 1
                print(f"[i1 mutation failed] {err_str}", flush=True)
                continue
            child.gen = 0
            try:
                child = evaluate_only(child)
            except Exception as e:
                with state_lock:
                    n_failed += 1
                print(f"[i1 eval failed] {e}", flush=True)
                continue
            if np.isfinite(child.score) and child.score >= min_keep_score:
                commit_ind(child)
            else:
                with state_lock:
                    n_evaluated += 1
                    child.extra["committed_at_call"] = llm.usage.attempted
                    history.append(_record(child))
                    if child.extra.get("error"):
                        n_failed += 1
                    print(
                        f"[{n_evaluated:>3d}/{args.budget}] op={child.operator:<10s} "
                        f"DROP| score={child.score:+.4f} (<{min_keep_score:.3f}) — kept out of pop",
                        flush=True,
                    )

    stagnant = 0
    best_score = pop.best().score if pop.best() else float("-inf")
    # iid -> (base operator, parent_id); used for discounted lineage credit.
    lineage: dict[str, tuple[str, str | None]] = {}
    n_revised = 0
    n_polished = 0

    def _propose_one(force_op: str | None = None):
        with state_lock:
            best = pop.best()
            ops.best_score = best.score if best else None
            ops.best_concept = best.concept if best else None
            ops.plateau_count = stagnant
            parent = pop.tournament(k=3) or best
            parents_xdiv = pop.diverse_sample(3) if len(pop) >= 2 else []
            parents_topk = pop.topk(min(5, len(pop))) if len(pop) >= 2 else []
            pop_len = len(pop)
        before = parent.score if parent else float("-inf")
        call_start = llm.usage.attempted + 1
        if force_op is not None:
            op_name = f"{force_op}_forced"
        else:
            # UCB1 bandit over {m_diag, m_text, x_div, i1} (paper Eq. 2).
            op_name = ucb.select()
            if op_name == "x_div" and pop_len < 2:
                op_name = "m_diag"
            if op_name == "m_diag" and not (parent and parent.image_b64):
                op_name = "m_text"
        try:
            if op_name == "x_div":
                child = ops.crossover_diversity(parents_xdiv)
            elif op_name == "x_div_forced":
                child = ops.crossover_diversity(parents_topk)
            elif op_name in ("m_diag", "m_diag_forced"):
                child = mutate_parent(parent, "m_diag", forced=force_op is not None)
            elif op_name in ("m_inspire", "m_inspire_forced"):
                child = ops.mutate_inspire(parent)
            elif op_name in ("i1", "i1_forced"):
                child = ops.init_zero_shot()
            else:
                child = mutate_parent(parent, "m_text", forced=force_op is not None)
            child.extra["generated_call_start"] = call_start
            child.extra["generated_call_end"] = llm.usage.attempted
            return op_name, child, before, ""
        except LLMCallBudgetExceeded:
            return op_name, None, before, "__LLM_CALL_BUDGET_EXHAUSTED__"
        except Exception as e:
            return op_name, None, before, str(e)

    def _process_batch(force_ops):
        nonlocal n_failed, best_score, stagnant, n_revised, n_polished, budget_exhausted
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_propose_one, fo) for fo in force_ops]
            results = [f.result() for f in as_completed(futs)]
        evaluated = []
        for op_name, child, before, err_str in results:
            if child is None:
                if err_str == "__LLM_CALL_BUDGET_EXHAUSTED__":
                    budget_exhausted = True
                    print("[REFLEX-Acrobot] LLM call budget exhausted.", flush=True)
                    continue
                with state_lock:
                    n_failed += 1
                    bandit.setdefault(op_name, []).append(-1.0)
                ucb.update(op_name, None)
                print(f"[{op_name} failed] {err_str}", flush=True)
                continue
            child.gen = n_evaluated
            try:
                child = evaluate_only(child)
            except Exception as e:
                with state_lock:
                    n_failed += 1
                ucb.update(op_name, None)
                print(f"[{op_name} eval failed] {e}", flush=True)
                continue
            if child.extra.get("error") and child.code:
                try:
                    fixed = ops.repair(child, child.extra["error"])
                    if fixed.code and fixed.code != child.code:
                        fixed.gen = n_evaluated
                        fixed = evaluate_only(fixed)
                        if not fixed.extra.get("error"):
                            child = fixed
                except Exception:
                    pass
            # Optional CEM constant polishing on near-frontier candidates.
            if (getattr(args, "polish_mode", "none") != "none"
                    and not child.extra.get("error") and np.isfinite(child.score)):
                with state_lock:
                    near_frontier = child.score >= best_score - 0.3 * max(1.0, abs(best_score))
                if near_frontier:
                    def _pol_eval(c):
                        s, _, e, _ = evaluate_code(c, "inner_cem")
                        return float("-inf") if e else float(s)
                    try:
                        pre_polish = child.score
                        new_code, new_score = polish_constants_cem(
                            child.code, _pol_eval, seed=args.seed)
                        if new_score > child.score and new_code != child.code:
                            child.code = new_code
                            child = evaluate_only(child)
                            child.extra["polished"] = True
                            child.extra["pre_polish_score"] = float(pre_polish)
                            with state_lock:
                                n_polished += 1
                            print(
                                f"[polish] {child.iid} tuned constants: "
                                f"{pre_polish:+.4f} -> {new_score:+.4f}",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[polish failed] {e}", flush=True)
            # Reflexion-style one-shot revision: if a mutation of an
            # elite-frontier parent regressed, let the Actor immediately
            # undo the harmful part (costs 1 LLM call + 1 eval).
            with state_lock:
                near_best = np.isfinite(before) and before >= best_score - 0.05 * max(1.0, abs(best_score))
            regressed = (
                np.isfinite(child.score) and np.isfinite(before)
                and child.score < before - 0.02 * max(1.0, abs(before))
                # catastrophic children carry no salvageable edit; skip them
                and child.score > before - 0.5 * max(1.0, abs(before))
            )
            with state_lock:
                rev_budget_ok = n_revised < max(3, args.budget // 10)
            if (near_best and regressed and rev_budget_ok
                    and not child.extra.get("error")
                    and child.parent_id and "," not in child.parent_id):
                with state_lock:
                    parent_obj = next((p for p in pop if p.iid == child.parent_id), None)
                if parent_obj is not None:
                    try:
                        with state_lock:
                            n_revised += 1
                        revised = ops.revise(parent_obj, child)
                        if revised.code and revised.code != child.code:
                            revised.gen = n_evaluated
                            revised = evaluate_only(revised)
                            evaluated.append((op_name, revised, before))
                    except Exception:
                        pass
            evaluated.append((op_name, child, before))
        for op_name, child, before in evaluated:
            commit_ind(child)
            with state_lock:
                reward = (child.score - before
                          if np.isfinite(child.score) and np.isfinite(before) else -1.0)
                bandit.setdefault(op_name, []).append(float(reward))
                improved_parent = np.isfinite(child.score) and reward > 0
                if not improved_parent and child.parent_id and "," not in child.parent_id:
                    ops.note_failure(child.parent_id, child.concept, reward if np.isfinite(reward) else float("-1.0"))
                # Promote / demote retrieved skills based on the child outcome
                # (paper Sec. 3.4: utility update + eviction).
                if child.skill_ids_used:
                    skills.update_after_eval(
                        child.skill_ids_used,
                        helped=bool(improved_parent),
                        magnitude=min(abs(reward), 1.0),
                    )
                # Extract a new skill whenever the child beats its parent
                # (paper Alg. 1: ΔF > 0), not only on a new global best.
                if improved_parent:
                    try:
                        skills.add_from_elite(
                            task_description=ACROBOT_TASK_DESCRIPTION,
                            elite_code=child.code, elite_concept=child.concept,
                            delta_fitness=max(0.0, reward), diagnosis=child.diagnosis,
                            parent_iid=child.parent_id or "", target_hook="choose_action",
                        )
                    except Exception:
                        pass
                if child.score > best_score:
                    stagnant = 0
                    best_score = child.score
                else:
                    stagnant += 1
            ucb.update(op_name, reward)
            with state_lock:
                pid = child.parent_id if (child.parent_id and "," not in child.parent_id) else None
                lineage[child.iid] = (op_name, pid)
                # Discounted lineage credit: exploratory operators (e.g.
                # m_inspire) often pay off via descendants, not direct hits.
                if reward > 0:
                    anc, depth = pid, 1
                    while anc and depth <= 3:
                        entry = lineage.get(anc)
                        if not entry:
                            break
                        anc_op, anc_parent = entry
                        ucb.update(anc_op, reward * (0.5 ** depth))
                        anc, depth = anc_parent, depth + 1

    while (
        n_evaluated < args.budget
        and not budget_exhausted
        and (llm.remaining_calls() is None or llm.remaining_calls() > 0)
    ):
        with state_lock:
            best = pop.best()
        if best and best.score >= args.target_score and not args.run_to_call_budget:
            break
        if stagnant >= args.stagnation_threshold and len(pop) >= 2:
            print(
                f"[REFLEX-Acrobot] ** STAGNATION {stagnant} steps → forced burst **",
                flush=True,
            )
            with state_lock:
                stagnant = 0
            _process_batch(["m_inspire", "i1", "x_div", "m_diag"][:max(2, n_workers)])
            continue
        batch_n = min(n_workers, max(1, args.budget - n_evaluated))
        _process_batch([None] * batch_n)

    skills.save()
    best = pop.best()
    payload = {
        "task": "acrobot",
        "seed": args.seed,
        "training_seeds": list(training_seeds),
        "condition": args.diagnostic_mode,
        "evaluation_budget": args.budget,
        "population_size": args.pop_size,
        "ucb_c": args.ucb_c,
        "stagnation_threshold": args.stagnation_threshold,
        "diagnostic_mode": args.diagnostic_mode,
        "retrieval_mode": args.retrieval_mode,
        "freeze_skill_bank": args.freeze_skill_bank,
        "llm_call_budget": args.llm_call_budget,
        "max_tokens": args.max_tokens,
        "thinking_mode": (
            "enabled"
            if args.enable_thinking
            else ("disabled" if args.disable_thinking else "server_default")
        ),
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "run_to_call_budget": args.run_to_call_budget,
        "budget_exhausted": budget_exhausted or llm.remaining_calls() == 0,
        "target_score": args.target_score,
        "best": _record(best) if best else None,
        "population": [_record(ind) for ind in pop],
        "history": history,
        "llm_usage": llm.stats(),
        "llm_call_trace": llm.call_trace(),
        "ucb": ucb.stats(),
        "skills": skills.stats(),
        "retrieval_trace": skills.retrieval_trace(),
        "n_evaluated": n_evaluated, "n_failed": n_failed,
        "n_polished": n_polished,
        "evaluator_protocol": "acrobot-policy-rng-v2" if getattr(args, "reproducible_evaluation", False) else "legacy_v1",
        "polish_mode": getattr(args, "polish_mode", "none"),
        "evaluator_invocations": len(evaluator_trace),
        "inner_cem_evaluator_invocations": sum(r["origin"] == "inner_cem" for r in evaluator_trace),
        "evaluator_trace": evaluator_trace,
    }
    dump_json(os.path.join(args.log_dir, "history.json"), payload)
    if best:
        Path(args.log_dir, "best_policy.py").write_text(best.code + "\n", encoding="utf-8")
    print("\n--- Final report (Acrobot) ---", flush=True)
    print(f"  best score={best.score if best else 'n/a'}", flush=True)
    print(f"  best concept={best.concept if best else 'n/a'}", flush=True)
    print(f"  ucb={ucb.stats()}", flush=True)
    print(f"  skills={skills.stats()}", flush=True)
    print(f"  llm_usage={llm.stats()}", flush=True)
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-provider", choices=("openai",), default="openai")
    p.add_argument("--llm-host", default=os.environ.get("LLM_HOST", ""))
    p.add_argument("--llm-key", default=os.environ.get("LLM_KEY", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    p.add_argument("--vision-model", default=os.environ.get("VISION_MODEL") or None)
    p.add_argument("--vision-host", default=os.environ.get("VISION_HOST") or None,
                   help="Separate OpenAI-compatible endpoint for the vision model (defaults to --llm-host).")
    p.add_argument("--vision-key", default=os.environ.get("VISION_KEY") or None,
                   help="API key for --vision-host (defaults to --llm-key).")
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable reasoning through provider and vLLM chat-template controls.",
    )
    p.add_argument("--reasoning-effort", default=None,
                   help="Optional OpenAI reasoning effort, e.g. low/medium/high.")
    p.add_argument("--verbosity", default=None,
                   help="Optional OpenAI response verbosity, e.g. low/medium/high.")
    p.add_argument("--llm-timeout", type=float, default=300.0)
    p.add_argument("--llm-max-retries", type=int, default=2)
    p.add_argument(
        "--llm-no-env-proxy",
        action="store_true",
        help="Do not inherit HTTP(S)_PROXY for the model endpoint.",
    )
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Generation cap per LLM call (includes thinking tokens).")
    p.add_argument("--training-seeds", default="random",
                   help="Comma-separated environment seeds, or random (default; drawn values are logged).")
    p.add_argument("--budget", type=int, default=80)
    p.add_argument(
        "--llm-call-budget",
        type=int,
        default=None,
        help="Hard cap on successful + failed LLM generation attempts.",
    )
    p.add_argument(
        "--diagnostic-mode",
        choices=("full", "actor_only", "text_critic"),
        default="full",
        help="Critic diagnostic mode (full, actor_only, or text_critic).",
    )
    p.add_argument("--reproducible-evaluation", action="store_true",
                   help="Seed per-episode policy RNGs and reset state for each episode.")
    p.add_argument("--polish-mode", choices=("none", "cem"), default="none",
                   help="Optional CEM constant polishing on elite policies.")
    p.add_argument(
        "--retrieval-mode",
        choices=(
            "bm25_utility",
            "bm25_constant",
            "random",
            "none",
            "dense_embedding",
            "dense_lsa",
        ),
        default="bm25_utility",
        help="Skill retrieval mode.",
    )
    p.add_argument(
        "--freeze-skill-bank",
        action="store_true",
        help="Disable skill addition/merge while retaining fitness-based utility updates.",
    )
    p.add_argument(
        "--run-to-call-budget",
        action="store_true",
        help="Ignore task target for stopping and continue until the LLM-call cap.",
    )
    p.add_argument("--pop-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--target-score", type=float, default=-0.20)
    p.add_argument("--ucb-c", type=float, default=1.2,
                   help="UCB1 exploration constant for operator selection.")
    p.add_argument("--stagnation-threshold", type=int, default=12)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--skill-bank", default="runs/acrobot/skill_bank.json")
    p.add_argument("--seed", type=seed_argument, default=None,
                   help="Campaign seed, or random (default; the drawn value is logged).")
    p.add_argument(
        "--noise-level",
        choices=("clean", "mild", "moderate", "severe"),
        default="clean",
        help="Inject noise into BE images for the Critic stress test.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)
    if args.llm_provider == "openai" and (
        not args.llm_host or not args.llm_key or not args.llm_model
    ):
        p.error(
            "OpenAI-compatible provider requires host/key/model via flags or "
            "LLM_HOST, LLM_KEY, and LLM_MODEL."
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    main()
