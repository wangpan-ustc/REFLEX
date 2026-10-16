#!/usr/bin/env python
"""Strict 36D REFLEX runner for the CCAA equal-objective comparison.

Every candidate consumes exactly one objective evaluation, so the evaluation
budget can be audited directly. The core Critic/Actor, UCB1, Skill Memory,
crossover, and inspiration operators are active. Generated code is AST-checked
and executed with a restricted builtins/import environment. Each run is seeded
with its run identifier, so a random-vector initialization matches the one
drawn by ``run_ccaa_classical_baselines.py`` for the same run.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import getpass
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.antenna_physics import evaluate_design
from examples.run_antenna import REGIMES, cfg_to_spec, render_antenna_be
from examples.antenna_template import ANTENNA_TASK_DESCRIPTION, ANTENNA_TEMPLATE
from reflex.bandit import UCB1
from reflex.critic_actor import MDCOrchestrator
from reflex.llm_client import LLMClient
from reflex.randomness import prior_run_ids
from reflex.memory import SkillMemory
from reflex.operators import Operators
from reflex.population import Individual, Population
from reflex.utils import short_hash


PROTOCOL_ID = "reflex-ccaa-strict36-random-v2"
CONDITIONS = ("random_vector", "shared_llm_first_prior")
STRICT_TASK_DESCRIPTION = ANTENNA_TASK_DESCRIPTION + """

STRICT FAIR-COMPARISON CONSTRAINTS:
- Return exactly the base 36-dimensional vector: 32 amplitude codes followed by 4 radius codes.
- Do not return rotations, phases, a dict, or a refine/local-search instruction.
- Do not use randomness, time, files, network access, subprocesses, or mutable global state.
- The same function call with the same spec must return the identical vector.
"""

ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "__builtins__", "subprocess", "socket",
    "requests", "urllib", "pathlib", "shutil", "os", "sys",
}
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "float": float, "int": int, "len": len, "list": list, "max": max, "min": min,
    "range": range, "round": round, "sum": sum, "tuple": tuple, "zip": zip,
}


def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)


def validate_ast(code: str) -> ast.Module:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("only numpy and math imports are allowed")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".", 1)[0] not in ALLOWED_IMPORTS:
                raise ValueError("only numpy and math imports are allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"forbidden dunder attribute: {node.attr}")
    return tree


def execute_strict36(code: str, spec: dict[str, Any]) -> np.ndarray:
    tree = validate_ast(code)
    builtins = dict(SAFE_BUILTINS)
    builtins["__import__"] = safe_import
    namespace: dict[str, Any] = {"np": np, "numpy": np, "math": math, "__builtins__": builtins}
    exec(compile(tree, "<reflex-generated>", "exec"), namespace)
    function = namespace.get("design_array")
    if not callable(function):
        raise ValueError("code did not define design_array(spec)")
    first = function(dict(spec))
    second = function(dict(spec))
    if isinstance(first, dict) or isinstance(second, dict):
        raise ValueError("dict/refine output is forbidden")
    x = np.asarray(first, dtype=float).reshape(-1)
    repeated = np.asarray(second, dtype=float).reshape(-1)
    if x.size != 36:
        raise ValueError(f"expected strict 36D vector, got {x.size}D")
    if repeated.size != x.size or not np.array_equal(x, repeated):
        raise ValueError("design_array is nondeterministic")
    if not np.all(np.isfinite(x)):
        raise ValueError("design vector contains non-finite values")
    return np.clip(x, 0.0, 1.0)


def code_from_vector(x: np.ndarray) -> str:
    values = ", ".join(repr(float(value)) for value in np.asarray(x).reshape(-1))
    return (
        "import numpy as np\n\n"
        "def design_array(spec):\n"
        f"    return np.array([{values}], dtype=float)\n"
    )


def sha256_vector(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def initial_candidate(
    condition: str,
    run_id: int,
    prior_root: Path,
    rng: np.random.Generator,
    model: str,
) -> tuple[str, dict]:
    if condition == "random_vector":
        x = rng.random(36)
        return code_from_vector(x), {
            "kind": "seeded_uniform_random_vector",
            "rng_seed": run_id,
            "x_sha256_f64le": sha256_vector(x),
            "score": None,
        }
    path = prior_root / f"run_{run_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    x = np.asarray(data.get("x"), dtype=float)
    if not data.get("valid") or x.size != 36 or data.get("model") != model:
        raise ValueError(f"invalid strict LLM prior: {path}")
    if sha256_vector(x) != data.get("x_sha256_f64le"):
        raise ValueError(f"prior hash mismatch: {path}")
    return data["code"], {
        "kind": "per-run_first_valid_llm_strict36",
        "source": str(path.relative_to(ROOT)),
        "x_sha256_f64le": data["x_sha256_f64le"],
        "code_sha256": data["code_sha256"],
        "score": data["score"],
    }


def record(individual: Individual, evaluation: int, objective_evals: int) -> dict:
    metrics = individual.be_summary or {}
    return {
        "evaluation": evaluation,
        "objective_evaluations": objective_evals,
        "iid": individual.iid,
        "operator": individual.operator,
        "parent_id": individual.parent_id,
        "concept": individual.concept,
        "score": float(individual.score),
        "valid": bool(np.isfinite(individual.score)),
        "error": individual.extra.get("error"),
        "x": individual.extra.get("x"),
        "sll_db": float(metrics["sll_db"]) if "sll_db" in metrics else None,
        "fnbw_deg": float(metrics["fnbw_deg"]) if "fnbw_deg" in metrics else None,
        "directivity_db": float(metrics["directivity_db"]) if "directivity_db" in metrics else None,
        "skill_ids_used": list(individual.skill_ids_used or []),
    }


def run_one(
    condition: str,
    run_id: int,
    budget: int,
    output_root: Path,
    prior_root: Path,
    api_key: str,
    llm_host: str,
    llm_model: str,
) -> dict:
    run_dir = output_root / f"budget_{budget}" / f"run_{run_id}" / condition
    history_path = run_dir / "history.json"
    if history_path.exists():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        if existing.get("objective_evaluations") == budget:
            return {"run": str(run_dir.relative_to(ROOT)), "status": "skip", "best": existing["best_score"]}
        raise RuntimeError(f"refusing to overwrite incomplete run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = REGIMES["compact"]
    spec = cfg_to_spec(cfg)
    py_rng = random.Random(run_id)
    np_rng = np.random.default_rng(run_id)
    llm = LLMClient(
        host=llm_host,
        key=api_key,
        model=llm_model,
        vision_model=llm_model,
        timeout=300.0,
        temperature=0.8,
        max_tokens=4096,
        max_retries=3,
        extra_headers={"X-Request-Id": uuid.uuid4().hex},
    )
    skills = SkillMemory(str(run_dir / "skill_bank.json"), llm=llm, max_skills=256, retrieval_seed=run_id)
    ops = Operators(
        STRICT_TASK_DESCRIPTION,
        ANTENNA_TEMPLATE,
        MDCOrchestrator(llm),
        skills,
        rng=py_rng,
    )
    pop = Population(capacity=16, rng=py_rng)
    ucb = UCB1(["m_diag", "m_text", "x_div", "i1", "m_inspire"], c=1.2)
    history: list[dict] = []
    objective_evals = 0
    invalid = 0
    generation_failures = 0
    skill_extraction_failures = 0
    stagnation = 0
    best_score = -math.inf
    start = time.time()
    events_path = run_dir / "events.jsonl"

    def evaluate_and_commit(individual: Individual) -> None:
        nonlocal objective_evals, invalid, stagnation, best_score
        objective_evals += 1
        individual.iid = individual.iid or f"g{objective_evals:03d}_{short_hash(individual.code, 8)}"
        x = None
        try:
            x = execute_strict36(individual.code, spec)
            metrics = evaluate_design(x, cfg)
            individual.score = float(metrics["score"])
            individual.be_summary = metrics
            individual.extra["x"] = [float(value) for value in x]
        except Exception as exc:
            individual.score = float("-inf")
            individual.extra["error"] = f"{type(exc).__name__}: {exc}"
            invalid += 1
        if np.isfinite(individual.score):
            try:
                individual.image_b64 = render_antenna_be(x, cfg, individual.score)
            except Exception as exc:
                # A rendering failure removes the BE image, not the design.
                individual.extra["render_error"] = f"{type(exc).__name__}: {exc}"
        pop.add(individual)
        if individual.score > best_score:
            best_score = individual.score
            stagnation = 0
        else:
            stagnation += 1
        row = record(individual, len(history) + 1, objective_evals)
        row["best_so_far"] = best_score
        history.append(row)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({
            "run_id": run_id,
            "condition": condition,
            "eval": objective_evals,
            "score": None if not np.isfinite(individual.score) else round(individual.score, 5),
            "best": None if not np.isfinite(best_score) else round(best_score, 5),
            "calls": llm.usage.attempted,
        }), flush=True)

    initial_code, initial_meta = initial_candidate(condition, run_id, prior_root, np_rng, llm_model)
    evaluate_and_commit(Individual(code=initial_code, concept=condition, operator="initial", gen=0))

    generation_attempts = 0
    while objective_evals < budget:
        generation_attempts += 1
        if generation_attempts > budget * 8:
            raise RuntimeError(f"too many generation failures: {generation_failures}")
        best = pop.best()
        ops.best_score = best.score if best else None
        ops.best_concept = best.concept if best else None
        ops.plateau_count = stagnation
        parent = pop.tournament(k=3) or best
        if stagnation >= 12:
            # One forced inspiration proposal per stagnation episode.
            op_name = "m_inspire"
            stagnation = 0
        else:
            op_name = ucb.select()
        if op_name == "x_div" and len(pop) < 2:
            op_name = "i1"
        if op_name == "m_diag" and not (parent and parent.image_b64):
            op_name = "m_text"
        parent_score = parent.score if parent else -math.inf
        try:
            if op_name == "x_div":
                child = ops.crossover_diversity(pop.diverse_sample(3))
            elif op_name == "m_diag":
                child = ops.mutate_diagnostic(parent)
            elif op_name == "m_text":
                child = ops.mutate_text_only(parent)
            elif op_name == "m_inspire":
                child = ops.mutate_inspire(parent)
            else:
                child = ops.init_zero_shot()
            child.operator = op_name
            child.gen = objective_evals
        except Exception:
            generation_failures += 1
            ucb.update(op_name, None)
            continue
        evaluate_and_commit(child)
        reward = child.score - parent_score if np.isfinite(child.score) and np.isfinite(parent_score) else -1.0
        ucb.update(op_name, reward)
        if child.skill_ids_used:
            skills.update_after_eval(child.skill_ids_used, helped=bool(reward > 0), magnitude=min(abs(reward), 1.0))
        # Skill extraction whenever the child beats its parent, as in the control runners.
        if reward > 0:
            try:
                skills.add_from_elite(
                    task_description=STRICT_TASK_DESCRIPTION,
                    elite_code=child.code,
                    elite_concept=child.concept,
                    delta_fitness=reward,
                    diagnosis=child.diagnosis,
                    parent_iid=parent.iid if parent else "",
                    target_hook="design_array",
                )
            except Exception as exc:
                skill_extraction_failures += 1
                print(f"[skill extraction failed] {type(exc).__name__}: {exc}", flush=True)

    skills.save()
    best = pop.best()
    curve = [float(row["best_so_far"]) for row in history]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "endpoint": llm_host,
        "model": llm_model,
        "condition": condition,
        "run_id": run_id,
        "rng_seed": run_id,
        "objective_budget": budget,
        "objective_evaluations": objective_evals,
        "initialization": initial_meta,
        "population_size": 16,
        "ucb_c": 1.2,
        "stagnation_threshold": 12,
        "skill_memory_capacity": 256,
        "sampling_workers": 1,
        "temperature": 0.8,
        "max_tokens": 4096,
        "strict_base36": True,
        "local_polish_enabled": False,
        "repair_revision_evaluations_enabled": False,
        "best_score": float(best.score),
        "best_code": best.code,
        "best_x": best.extra.get("x"),
        "checkpoints": {str(k): curve[k - 1] for k in (50, 100, 240) if k <= budget},
        "evals_to_25_0": next((row["evaluation"] for row in history if row["best_so_far"] >= 25.0), None),
        "evals_to_25_25": next((row["evaluation"] for row in history if row["best_so_far"] >= 25.25), None),
        "invalid_candidates": invalid,
        "generation_failures": generation_failures,
        "skill_extraction_failures": skill_extraction_failures,
        "llm_usage": llm.stats(),
        "llm_call_trace": llm.call_trace(),
        "ucb": ucb.stats(),
        "skills": skills.stats(),
        "elapsed_seconds": time.time() - start,
        "history": history,
        "credential_persisted": False,
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"run": str(run_dir.relative_to(ROOT)), "status": "ok", "best": payload["best_score"],
            "llm_failed": payload["llm_usage"]["failed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-host", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--out-root", default="runs/ccaa_reflex_formal")
    parser.add_argument("--prior-root", default="runs/ccaa_llm_priors_strict36")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--run-workers", type=int, default=2)
    args = parser.parse_args()

    runs_root = (ROOT / "runs").resolve()
    output_root = (ROOT / args.out_root).resolve()
    prior_root = (ROOT / args.prior_root).resolve()
    if output_root.parent != runs_root:
        raise ValueError(f"--out-root must be a direct child of {runs_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_ids = list(range(args.num_runs))
    conditions = [value for value in args.conditions.split(",") if value]
    if not set(conditions).issubset(CONDITIONS):
        raise ValueError(f"unknown condition in {conditions}")
    if "shared_llm_first_prior" in conditions:
        if prior_root.parent != runs_root or not prior_root.is_dir():
            raise ValueError(f"--prior-root must be an existing direct child of {runs_root}")
        run_ids = prior_run_ids(prior_root)
    api_key = os.environ.get("LLM_KEY", "").strip() or getpass.getpass("API key: ").strip()
    if not api_key:
        raise SystemExit("missing API key")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "endpoint": args.llm_host,
        "model": args.llm_model,
        "prior_root": str(prior_root.relative_to(ROOT)),
        "conditions": conditions,
        "run_ids": run_ids,
        "objective_budget": args.budget,
        "n_runs": len(run_ids) * len(conditions),
        "run_workers": args.run_workers,
        "credential_persisted": False,
    }
    (output_root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    jobs = [(condition, run_id) for run_id in run_ids for condition in conditions]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.run_workers) as executor:
        futures = {
            executor.submit(
                run_one,
                condition,
                run_id,
                args.budget,
                output_root,
                prior_root,
                api_key,
                args.llm_host,
                args.llm_model,
            ): (condition, run_id)
            for condition, run_id in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: row["run"])
    (output_root / "campaign_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
