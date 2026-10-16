"""REFLEX runner for the compact CCAA antenna synthesis task."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.antenna_physics import CCAAConfig, evaluate_design
from examples.antenna_seeds import CLASSIC_SEEDS
from examples.antenna_template import (
    ANTENNA_STAGNATION_HINT,
    ANTENNA_TASK_DESCRIPTION,
    ANTENNA_TEMPLATE,
)
from reflex.bandit import UCB1
from reflex.critic_actor import MDCOrchestrator
from reflex.randomness import resolve_seed, seed_argument
from reflex.llm_client import LLMClient
from reflex.memory import SkillMemory
from reflex.operators import Operators
from reflex.population import Individual, Population
from reflex.utils import dump_json, ensure_dir, short_hash


REGIMES: dict[str, CCAAConfig] = {
    "compact": CCAAConfig(ring_counts=(5, 7, 9, 11), fnbw_target_deg=28.0),
    "hauptmatch": CCAAConfig(ring_counts=(8, 12, 16, 20), fnbw_target_deg=18.0),
}


def cfg_to_spec(cfg: CCAAConfig) -> dict[str, Any]:
    return {
        "num_rings": cfg.num_rings,
        "ring_counts": tuple(cfg.ring_counts),
        "num_elements": cfg.num_elements,
        "design_dim": cfg.design_dim,
        "design_dim_rot": cfg.design_dim_rot,
        "design_dim_full": cfg.design_dim_full,
        "amp_min": cfg.amp_min,
        "min_spacing": cfg.min_spacing,
        "extra_spacing": cfg.extra_spacing,
        "fnbw_target_deg": cfg.fnbw_target_deg,
    }


def _safe_exec_design(code: str, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    ns: dict[str, Any] = {"np": np, "numpy": np, "math": math}
    exec(code, ns)
    fn = ns.get("design_array")
    if not callable(fn):
        raise ValueError("code did not define design_array(spec)")
    out = fn(spec)
    if isinstance(out, dict):
        x = out.get("x", out.get("design", out.get("array")))
        hint = dict(out.get("refine", {}) or {})
        if x is None:
            raise ValueError("design dict did not contain x/design/array")
        return np.asarray(x, dtype=float).reshape(-1), hint
    return np.asarray(out, dtype=float).reshape(-1), {}


def _polish_design(x: np.ndarray, cfg: CCAAConfig, hint: dict[str, Any]) -> np.ndarray:
    iters = int(min(max(hint.get("iters", 80), 0), 200))
    if iters <= 0:
        return x
    rng = np.random.default_rng(hint.get("seed"))
    best_x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
    best = float(evaluate_design(best_x, cfg)["score"])
    scale = float(hint.get("scale", 0.035))
    n_amp = cfg.num_elements
    # Per-block perturbation scales: amps full scale; radii tighter;
    # rotations/phases (if present in the extended design vector) tighter
    # still, since the score is very sensitive to them.
    sigmas = np.full(best_x.size, 0.5 * scale)
    sigmas[:n_amp] = scale
    if best_x.size >= cfg.design_dim:
        sigmas[n_amp : n_amp + cfg.num_rings] = 0.7 * scale
    for i in range(iters):
        cand = best_x + rng.normal(0.0, 1.0, best_x.size) * sigmas * (0.985 ** i)
        cand = np.clip(cand, 0.0, 1.0)
        score = float(evaluate_design(cand, cfg)["score"])
        if score > best:
            best = score
            best_x = cand
    return best_x


def evaluate_code(code: str, cfg: CCAAConfig, spec: dict[str, Any]) -> tuple[float, dict[str, Any], str, np.ndarray | None]:
    try:
        x, hint = _safe_exec_design(code, spec)
        if hint:
            x = _polish_design(x, cfg, hint)
        metrics = evaluate_design(x, cfg)
        return float(metrics["score"]), metrics, "", x
    except Exception:
        return float("-inf"), {}, traceback.format_exc(limit=4), None


def render_antenna_be(x: np.ndarray, cfg: CCAAConfig, score: float) -> str:
    """Render an antenna-specific BE image and return it as base64 PNG."""
    import base64
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.antenna_physics import decode_design, elevation_cut_db

    dec = decode_design(x, cfg)
    amps = dec["amps"]
    phases = dec["phases"]
    el_r = dec["element_radii"]
    el_phi = dec["element_phi"]
    radii_lambda = dec["radii"]
    theta_deg, af_db = elevation_cut_db(amps, el_r, el_phi, n_theta=361, phases=phases)

    sll = float(np.max(af_db[(theta_deg < -3.0) | (theta_deg > 3.0)]))
    fnbw_tgt = float(cfg.fnbw_target_deg)

    fig = plt.figure(figsize=(11.5, 7.5))
    ax_layout = fig.add_subplot(2, 2, 1)
    xs = el_r * np.cos(el_phi)
    ys = el_r * np.sin(el_phi)
    sc = ax_layout.scatter(xs, ys, c=amps, cmap="viridis", s=28, vmin=0.0, vmax=1.0)
    for r in radii_lambda:
        circ = plt.Circle((0.0, 0.0), float(r), fill=False, color="0.6", lw=0.5, ls="--")
        ax_layout.add_patch(circ)
    ax_layout.set_aspect("equal")
    ax_layout.set_title("element layout (color = amplitude)")
    ax_layout.set_xlabel("x / λ"); ax_layout.set_ylabel("y / λ")
    fig.colorbar(sc, ax=ax_layout, fraction=0.045, pad=0.04)

    ax_cut = fig.add_subplot(2, 2, 2)
    ax_cut.plot(theta_deg, af_db, lw=1.2)
    ax_cut.axhline(sll, color="r", ls="--", lw=0.8, label=f"SLL = {sll:+.2f} dB")
    ax_cut.axhline(-20.0, color="0.6", ls=":", lw=0.7, label="-20 dB target")
    ax_cut.axvline(+fnbw_tgt / 2, color="g", ls=":", lw=0.7, label=f"FNBW target ±{fnbw_tgt/2:.1f}°")
    ax_cut.axvline(-fnbw_tgt / 2, color="g", ls=":", lw=0.7)
    ax_cut.set_ylim(-50, 1); ax_cut.set_xlim(-90, 90)
    ax_cut.set_xlabel("θ (deg)"); ax_cut.set_ylabel("|AF| (dB)")
    ax_cut.set_title(f"elevation cut  (score = {score:+.3f})")
    ax_cut.legend(loc="lower center", fontsize=8)
    ax_cut.grid(alpha=0.3)

    ax_amp = fig.add_subplot(2, 1, 2)
    ax_amp.bar(np.arange(amps.size), amps, color="steelblue", edgecolor="0.2", lw=0.4)
    cum = 0
    for k, count in enumerate(cfg.ring_counts):
        ax_amp.axvline(cum - 0.5, color="r", ls="--", lw=0.5)
        ax_amp.text(cum + count / 2 - 0.5, 1.02, f"ring {k}\nr={radii_lambda[k]:.2f}λ",
                    ha="center", va="bottom", fontsize=8)
        cum += count
    ax_amp.axvline(cum - 0.5, color="r", ls="--", lw=0.5)
    ax_amp.set_ylim(0, 1.15); ax_amp.set_xlim(-0.5, amps.size - 0.5)
    ax_amp.set_xlabel("element index (grouped by ring)")
    ax_amp.set_ylabel("amplitude")
    ax_amp.set_title("per-element amplitude profile")
    ax_amp.grid(alpha=0.3, axis="y")
    if np.any(np.abs(phases) > 1e-9):
        ax_ph = ax_amp.twinx()
        ax_ph.plot(np.arange(phases.size), phases, "o-", color="darkorange",
                   ms=3, lw=0.8, alpha=0.8, label="phase (rad)")
        ax_ph.set_ylim(-np.pi * 1.05, np.pi * 1.05)
        ax_ph.set_ylabel("excitation phase (rad)", color="darkorange")
        ax_ph.tick_params(axis="y", labelcolor="darkorange")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _design_row(ind: Individual, cfg: CCAAConfig) -> str | None:
    """One compact table line: score metrics + per-ring decoded parameters."""
    x = ind.extra.get("x")
    if x is None:
        return None
    from examples.antenna_physics import decode_design

    dec = decode_design(np.asarray(x, dtype=float), cfg)
    amps = dec["amps"]
    phases = dec["phases"]
    ring_amps, ring_phase = [], []
    cum = 0
    for count in cfg.ring_counts:
        ring_amps.append(float(np.mean(amps[cum:cum + count])))
        ring_phase.append(float(np.mean(np.abs(phases[cum:cum + count]))))
        cum += count
    radii = [float(r) for r in dec["radii"]]
    m = ind.be_summary or {}
    return (
        f"| {ind.score:+.3f} | {m.get('sll_db', float('nan')):+.2f} "
        f"| {m.get('fnbw_deg', float('nan')):.1f} "
        f"| {m.get('directivity_db', float('nan')):.2f} "
        f"| {', '.join(f'{a:.3f}' for a in ring_amps)} "
        f"| {', '.join(f'{r:.3f}' for r in radii)} "
        f"| {', '.join(f'{p:.2f}' for p in ring_phase)} |"
    )


def build_numeric_feedback(pop: Population, history_inds: list[Individual],
                           cfg: CCAAConfig, top_k: int = 6, recent_k: int = 4) -> str | None:
    """Design ↔ score table: top-K elites + most recent attempts."""
    header = (
        "| score | SLL(dB) | FNBW(deg) | D(dB) | ring mean amps (inner→outer) "
        "| ring radii (λ) | ring mean |phase| (rad) |\n|---|---|---|---|---|---|---|"
    )
    rows: list[str] = []
    seen: set[str] = set()
    for ind in list(pop.topk(top_k)) + list(reversed(history_inds[-recent_k:])):
        if ind.iid in seen or not np.isfinite(ind.score):
            continue
        row = _design_row(ind, cfg)
        if row:
            rows.append(row)
            seen.add(ind.iid)
    if not rows:
        return None
    return header + "\n" + "\n".join(rows)


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": float(metrics.get("score", float("-inf"))),
        "sll_db": float(metrics.get("sll_db", float("nan"))),
        "fnbw_deg": float(metrics.get("fnbw_deg", float("nan"))),
        "directivity_db": float(metrics.get("directivity_db", float("nan"))),
    }


def _record(ind: Individual) -> dict[str, Any]:
    return {
        "iid": ind.iid,
        "score": ind.score,
        "concept": ind.concept,
        "operator": ind.operator,
        "gen": ind.gen,
        "parent_id": ind.parent_id,
        "skill_ids_used": list(ind.skill_ids_used),
        "be_summary": _summary_metrics(ind.be_summary),
        "code": ind.code,
        "error": ind.extra.get("error", ""),
    }


def _print_final(
    *,
    best: Individual | None,
    population: Population,
    skills: SkillMemory,
    llm: LLMClient,
    n_evaluated: int,
    n_failed: int,
    bandit: dict[str, list[float]],
    log_dir: str,
    ucb=None,
) -> None:
    ops = {}
    for name, rewards in sorted(bandit.items()):
        ops[name] = {
            "n": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }
    active = skills.all()
    mean_utility = float(np.mean([s.utility for s in active])) if active else 0.0
    best_summary = _summary_metrics(best.be_summary) if best else {}
    report = {
        "best_score": best.score if best else float("-inf"),
        "best_concept": best.concept if best else "",
        "be_summary": best_summary,
        "n_evaluated": n_evaluated,
        "n_failed": n_failed,
        "bandit": ops,
        "skills": {"n_active": len(active), "mean_utility": mean_utility},
        "llm_usage": llm.stats(),
        "log_dir": log_dir,
    }
    if ucb is not None:
        report["ucb"] = ucb.stats()
    print("\n--- Final report ---")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    print(
        f"[REFLEX] starting campaign: regime={args.regime} "
        f"budget={args.budget} pop_size={args.pop_size} "
        f"target_score={args.target_score} log_dir={args.log_dir}",
        flush=True,
    )
    print(
        f"[REFLEX] models: text='{args.llm_model}' vision='{args.vision_model}' "
        f"thinking={args.enable_thinking}",
        flush=True,
    )
    print(f"[REFLEX] skill bank: {args.skill_bank}", flush=True)
    args.seed = resolve_seed(args.seed)
    print(f"[REFLEX] seed={args.seed}", flush=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    ensure_dir(args.log_dir)

    cfg = REGIMES[args.regime]
    spec = cfg_to_spec(cfg)
    llm = LLMClient(
        host=args.llm_host,
        key=args.llm_key,
        model=args.llm_model,
        vision_model=args.vision_model,
        vision_host=args.vision_host,
        vision_key=args.vision_key,
        extra_body={"enable_thinking": bool(args.enable_thinking)},
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        max_tokens=args.max_tokens,
        debug=args.debug,
    )
    skills = SkillMemory(args.skill_bank, llm=llm, debug=args.debug)
    mdc = MDCOrchestrator(llm, debug=args.debug)
    ops = Operators(ANTENNA_TASK_DESCRIPTION, ANTENNA_TEMPLATE, mdc, skills, rng=rng)
    ops.stagnation_hint = ANTENNA_STAGNATION_HINT
    pop = Population(capacity=args.pop_size, rng=rng)
    history: list[dict[str, Any]] = []
    evaluated_inds: list[Individual] = []
    bandit: dict[str, list[float]] = {}
    ucb = UCB1(["m_diag", "m_text", "x_div", "i1", "m_inspire"], c=args.ucb_c)
    n_evaluated = 0
    n_failed = 0
    state_lock = threading.Lock()
    # Evaluation cache: code hash -> result tuple. The LLM regenerates
    # byte-identical policies surprisingly often; re-running the rollout
    # wastes wall-clock and floods the population with duplicates.
    eval_cache: dict[str, tuple] = {}
    cache_lock = threading.Lock()

    def evaluate_only(ind: Individual) -> Individual:
        """Pure CPU work: run code, build BE image. NO shared state mutation."""
        ind.iid = ind.iid or f"g{ind.gen:03d}_{short_hash(ind.code, 8)}"
        key = short_hash(ind.code, 16)
        with cache_lock:
            hit = eval_cache.get(key)
        if hit is not None:
            score, metrics, err, x_dec = hit
            ind.extra["cache_hit"] = True
        else:
            score, metrics, err, x_dec = evaluate_code(ind.code, cfg, spec)
            with cache_lock:
                eval_cache[key] = (score, metrics, err, x_dec)
        ind.score = score
        ind.be_summary = metrics
        if err:
            ind.extra["error"] = err
        elif x_dec is not None and np.isfinite(score):
            ind.extra["x"] = [float(v) for v in x_dec]
            try:
                ind.image_b64 = render_antenna_be(x_dec, cfg, score)
            except Exception as e:
                if args.debug:
                    print(f"[BE render failed] {e}", flush=True)
        return ind

    def commit_ind(ind: Individual) -> Individual:
        """Acquire lock, integrate scored individual into shared state, log."""
        nonlocal n_evaluated, n_failed
        with state_lock:
            n_evaluated += 1
            if ind.extra.get("error"):
                n_failed += 1
            pop.add(ind)
            history.append(_record(ind))
            evaluated_inds.append(ind)
            best = pop.best()
            best_s = best.score if best else float("-inf")
            sll = (ind.be_summary or {}).get("sll_db", float("nan"))
            flag = "ERR" if ind.extra.get("error") else "OK "
            try:
                calls_n = llm.usage.calls
                n_skills = len(skills)
            except Exception:
                calls_n, n_skills = 0, 0
            print(
                f"[{n_evaluated:>3d}/{args.budget}] op={ind.operator:<10s} {flag} | "
                f"score={ind.score:+.4f} sll={sll:+.2f} | best={best_s:+.4f} | "
                f"pop={len(pop)} skills={n_skills} calls={calls_n}",
                flush=True,
            )
        return ind

    def score_ind(ind: Individual) -> Individual:
        return commit_ind(evaluate_only(ind))

    for i, seed in enumerate(CLASSIC_SEEDS):
        score_ind(
            Individual(
                code=seed["code"],
                concept=seed.get("concept", seed["name"]),
                operator="seed",
                gen=0,
                iid=f"seed_{i}_{seed['name']}",
            )
        )
    score_ind(Individual(code=ANTENNA_TEMPLATE, concept="default template", operator="template", gen=0))

    while len(pop) < args.pop_size and n_evaluated < args.budget:
        try:
            child = ops.init_zero_shot()
        except Exception as e:
            n_failed += 1
            print(f"[init_zero_shot failed] {e}", flush=True)
            break
        child.gen = 0
        score_ind(child)

    stagnant = 0
    best_score = pop.best().score if pop.best() else float("-inf")
    n_workers = max(1, int(args.num_workers))
    # iid -> (base operator, parent_id); used for discounted lineage credit.
    lineage: dict[str, tuple[str, str | None]] = {}
    n_revised = 0

    def _propose_one(force_op: str | None = None) -> tuple[str, Individual | None, float, str]:
        """Generate ONE child (LLM call). Returns (op_name, child, parent_score, error_str).

        Reads pop snapshot under lock, then runs LLM call WITHOUT lock so
        multiple workers can do concurrent LLM requests.
        """
        with state_lock:
            best = pop.best()
            ops.best_score = best.score if best else None
            ops.best_concept = best.concept if best else None
            ops.plateau_count = stagnant
            try:
                ops.numeric_feedback = build_numeric_feedback(pop, evaluated_inds, cfg)
            except Exception:
                ops.numeric_feedback = None
            parent = pop.tournament(k=3) or best
            parents_xdiv = pop.sample(3) if len(pop) >= 2 else []
            parents_topk = pop.topk(min(5, len(pop))) if len(pop) >= 2 else []
            pop_len = len(pop)
        before = parent.score if parent else float("-inf")
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
                child = ops.mutate_diagnostic(parent)
            elif op_name in ("m_inspire", "m_inspire_forced"):
                child = ops.mutate_inspire(parent)
            elif op_name in ("i1", "i1_forced"):
                child = ops.init_zero_shot()
            else:
                child = ops.mutate_text_only(parent)
            return op_name, child, before, ""
        except Exception as e:
            return op_name, None, before, str(e)

    def _process_batch(force_ops: list[str | None]) -> int:
        """Generate len(force_ops) candidates concurrently, evaluate, commit.

        Returns the count of children that beat best_score (for stagnation
        bookkeeping). Logging happens inside commit_ind under lock.
        """
        nonlocal n_failed, best_score, stagnant, n_revised
        improved = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_propose_one, fo): fo for fo in force_ops}
            results: list[tuple[str, Individual | None, float, str]] = []
            for fut in as_completed(futs):
                results.append(fut.result())
        evaluated: list[tuple[str, Individual, float]] = []
        for op_name, child, before, err_str in results:
            if child is None:
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
                reward = (
                    child.score - before
                    if np.isfinite(child.score) and np.isfinite(before)
                    else -1.0
                )
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
                            task_description=ANTENNA_TASK_DESCRIPTION,
                            elite_code=child.code,
                            elite_concept=child.concept,
                            delta_fitness=max(0.0, reward),
                            diagnosis=child.diagnosis,
                            parent_iid=child.parent_id or "",
                            target_hook="design_array",
                        )
                    except Exception as e:
                        if args.debug:
                            print(f"[skill add failed] {e}", flush=True)
                if child.score > best_score:
                    stagnant = 0
                    best_score = child.score
                    improved += 1
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
        return improved

    while n_evaluated < args.budget:
        with state_lock:
            best = pop.best()
        if best and best.score >= args.target_score:
            break
        if stagnant >= args.stagnation_threshold and len(pop) >= 2:
            print(
                f"[REFLEX] ** STAGNATION {stagnant} steps w/o improvement → "
                f"forced exploration burst (i1 + x_div + m_diag) **",
                flush=True,
            )
            with state_lock:
                stagnant = 0
            _process_batch(["m_inspire", "i1", "x_div", "m_diag"][:max(2, n_workers)] or ["x_div"])
            continue
        batch_n = min(n_workers, max(1, args.budget - n_evaluated))
        _process_batch([None] * batch_n)

    skills.save()
    best = pop.best()
    payload = {
        "regime": args.regime,
        "seed": args.seed,
        "spec": spec,
        "target_score": args.target_score,
        "best": _record(best) if best else None,
        "population": [_record(ind) for ind in pop],
        "history": history,
        "llm_usage": llm.stats(),
        "ucb": ucb.stats(),
        "skills": skills.stats(),
        "n_evaluated": n_evaluated,
        "n_failed": n_failed,
    }
    dump_json(os.path.join(args.log_dir, "history.json"), payload)
    if best:
        Path(args.log_dir, "best_design.py").write_text(best.code + "\n", encoding="utf-8")
    _print_final(
        best=best,
        population=pop,
        skills=skills,
        llm=llm,
        n_evaluated=n_evaluated,
        n_failed=n_failed,
        bandit=bandit,
        log_dir=args.log_dir,
        ucb=ucb,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-host", default=os.environ.get("LLM_HOST", ""))
    p.add_argument("--llm-key", default=os.environ.get("LLM_KEY", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    p.add_argument("--vision-model", default=None)
    p.add_argument("--vision-host", default=None,
                   help="Separate OpenAI-compatible endpoint for the vision model (defaults to --llm-host).")
    p.add_argument("--vision-key", default=os.environ.get("VISION_KEY") or None,
                   help="API key for --vision-host (defaults to --llm-key).")
    p.add_argument("--llm-timeout", type=float, default=300.0)
    p.add_argument("--llm-max-retries", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Generation cap per LLM call (includes thinking tokens).")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--regime", choices=sorted(REGIMES), default="compact")
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--pop-size", type=int, default=16)
    p.add_argument("--target-score", type=float, default=30.0)
    p.add_argument("--ucb-c", type=float, default=1.2,
                   help="UCB1 exploration constant for operator selection.")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--skill-bank", default="runs/antenna/skill_bank.json")
    p.add_argument("--seed", type=seed_argument, default=None,
                   help="Campaign seed, or random (default; the drawn value is logged).")
    p.add_argument("--stagnation-threshold", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=1,
                   help="Concurrent LLM calls per round (MLES-style multi-sampling).")
    p.add_argument("--debug", action="store_true")
    ns = p.parse_args(argv)
    return ns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
