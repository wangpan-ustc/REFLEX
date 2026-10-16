#!/usr/bin/env python
"""Reproducible CPU-only CCAA baselines.

Protocol:
  - compact 5-7-9-11 CCAA, 36-dimensional base design vector;
  - methods: Random Search, CMA-ES, GA, and PSO;
  - initialization: random vector, a fixed analytical Taylor prior, or the
    frozen first-LLM prior of each run;
  - each run is seeded with its run identifier, so the first random vector
    matches the REFLEX random-vector initialization of the same run;
  - exactly 240 objective evaluations per run, including initialization;
  - checkpoints at 50, 100, and 240 evaluations.

No model API is used.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from examples.antenna_physics import CCAAConfig, evaluate_design
from reflex.randomness import prior_run_ids


METHODS = ("random_search", "cma_es", "ga", "pso")
INITIALIZATIONS = ("random_vector", "shared_analytical_prior", "shared_llm_first_prior")
CHECKPOINTS = (50, 100, 240)


@dataclass(frozen=True)
class Protocol:
    protocol_id: str = "reflex-ccaa-classical-random-v2"
    budget: int = 240
    dimension: int = 36
    population_size: int = 16
    target_easy: float = 25.0
    target_hard: float = 25.25
    shared_prior_name: str = "taylor-linear-analytical"


def compact_config() -> CCAAConfig:
    return CCAAConfig(ring_counts=(5, 7, 9, 11), fnbw_target_deg=28.0)


def analytical_prior(cfg: CCAAConfig) -> np.ndarray:
    """Exact vector emitted by examples.antenna_seeds.SEED_TAYLOR_LINEAR."""
    amp_codes: list[float] = []
    for index, count in enumerate(cfg.ring_counts):
        ring_pos = index / max(cfg.num_rings - 1, 1)
        amp_codes.extend([0.95 - 0.55 * ring_pos] * count)
    radius_codes = np.linspace(0.45, 0.55, cfg.num_rings)
    return np.clip(np.concatenate([np.asarray(amp_codes), radius_codes]), 0.0, 1.0)


def vector_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def load_prior(
    initialization: str,
    run_id: int,
    prior_root: Path | None,
    cfg: CCAAConfig,
) -> tuple[np.ndarray, dict]:
    if initialization != "shared_llm_first_prior":
        prior = analytical_prior(cfg)
        return prior, {
            "name": "taylor-linear-analytical",
            "source": "examples.antenna_seeds.SEED_TAYLOR_LINEAR",
            "score": float(evaluate_design(prior, cfg)["score"]),
            "sha256_f64le": vector_sha256(prior),
            "status": "fixed analytical Taylor prior",
        }
    if prior_root is None:
        raise ValueError("--prior-root is required for shared_llm_first_prior")
    path = prior_root / f"run_{run_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    prior = np.asarray(data.get("x"), dtype=float)
    if not data.get("valid") or prior.size != cfg.design_dim:
        raise ValueError(f"invalid strict-36 prior: {path}")
    if vector_sha256(prior) != data.get("x_sha256_f64le"):
        raise ValueError(f"prior vector hash mismatch: {path}")
    score = float(evaluate_design(prior, cfg)["score"])
    if abs(score - float(data["score"])) > 1e-10:
        raise ValueError(f"prior score mismatch: {path}")
    return prior, {
        "name": "llm-first-valid-strict36",
        "source": str(path.relative_to(ROOT)),
        "model": data["model"],
        "score": score,
        "sha256_f64le": data["x_sha256_f64le"],
        "code_sha256": data["code_sha256"],
        "status": "frozen first valid deterministic strict-36 LLM proposal",
    }


class Recorder:
    def __init__(self, cfg: CCAAConfig, budget: int) -> None:
        self.cfg = cfg
        self.budget = budget
        self.rows: list[dict] = []
        self.best = -math.inf

    @property
    def remaining(self) -> int:
        return self.budget - len(self.rows)

    def evaluate(self, x: np.ndarray) -> float:
        if self.remaining <= 0:
            raise RuntimeError("objective budget exhausted")
        x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
        if x.size != self.cfg.design_dim:
            raise ValueError(f"expected {self.cfg.design_dim} parameters, got {x.size}")
        metrics = evaluate_design(x, self.cfg)
        score = float(metrics["score"])
        self.best = max(self.best, score)
        self.rows.append(
            {
                "evaluation": len(self.rows) + 1,
                "score": score,
                "best_so_far": self.best,
                "sll_db": float(metrics["sll_db"]),
                "fnbw_deg": float(metrics["fnbw_deg"]),
                "directivity_db": float(metrics["directivity_db"]),
                "x": [float(v) for v in x],
            }
        )
        return score


def initial_population(
    rng: np.random.Generator,
    size: int,
    dim: int,
    initialization: str,
    prior: np.ndarray,
) -> np.ndarray:
    if initialization == "random_vector":
        return rng.random((size, dim))
    population = np.clip(prior + rng.normal(0.0, 0.15, size=(size, dim)), 0.0, 1.0)
    population[0] = prior
    return population


def run_random_search(
    recorder: Recorder, rng: np.random.Generator, initialization: str, prior: np.ndarray
) -> dict:
    while recorder.remaining:
        if initialization == "random_vector":
            x = rng.random(prior.size)
        elif not recorder.rows:
            x = prior
        else:
            progress = len(recorder.rows) / max(recorder.budget - 1, 1)
            sigma = 0.25 * (0.03 / 0.25) ** progress
            x = prior + rng.normal(0.0, sigma, prior.size)
        recorder.evaluate(x)
    return {"sampler": "uniform" if initialization == "random_vector" else "annealed_gaussian",
            "sigma_start": None if initialization == "random_vector" else 0.25,
            "sigma_end": None if initialization == "random_vector" else 0.03}


def run_cma_es(
    recorder: Recorder, rng: np.random.Generator, initialization: str, prior: np.ndarray
) -> dict:
    n = prior.size
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = 1.0 / np.sum(weights**2)
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

    mean = rng.random(n) if initialization == "random_vector" else prior.copy()
    sigma = 0.25
    cov = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    recorder.evaluate(mean)
    generation = 0
    while recorder.remaining:
        eigvals, basis = np.linalg.eigh((cov + cov.T) * 0.5)
        eigvals = np.maximum(eigvals, 1e-12)
        transform = basis @ np.diag(np.sqrt(eigvals))
        inv_sqrt = basis @ np.diag(1.0 / np.sqrt(eigvals)) @ basis.T
        batch = min(lam, recorder.remaining)
        z = rng.normal(size=(batch, n))
        y = z @ transform.T
        xs = np.clip(mean + sigma * y, 0.0, 1.0)
        scores = np.asarray([recorder.evaluate(x) for x in xs])
        if batch < mu:
            break
        order = np.argsort(scores)[::-1][:mu]
        old_mean = mean.copy()
        mean = np.sum(weights[:, None] * xs[order], axis=0)
        y_w = (mean - old_mean) / max(sigma, 1e-12)
        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt @ y_w)
        generation += 1
        hsig = float(
            np.linalg.norm(ps)
            / math.sqrt(max(1e-12, 1.0 - (1.0 - cs) ** (2.0 * generation)))
            < (1.4 + 2.0 / (n + 1.0)) * chi_n
        )
        pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w
        selected_y = (xs[order] - old_mean) / max(sigma, 1e-12)
        rank_mu = sum(w * np.outer(v, v) for w, v in zip(weights, selected_y))
        cov = (
            (1.0 - c1 - cmu) * cov
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * cov)
            + cmu * rank_mu
        )
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0))
        sigma = float(np.clip(sigma, 1e-4, 0.75))
    return {"lambda": lam, "mu": mu, "sigma_initial": 0.25, "boundary": "clip"}


def run_ga(
    recorder: Recorder, rng: np.random.Generator, initialization: str, prior: np.ndarray
) -> dict:
    pop_size = 16
    initial_size = min(pop_size, recorder.remaining)
    population = initial_population(rng, initial_size, prior.size, initialization, prior)
    scores = np.asarray([recorder.evaluate(x) for x in population])

    def tournament() -> np.ndarray:
        choices = rng.integers(0, len(population), size=3)
        return population[choices[int(np.argmax(scores[choices]))]]

    while recorder.remaining:
        children: list[np.ndarray] = []
        while len(children) < pop_size and len(children) < recorder.remaining:
            p1, p2 = tournament(), tournament()
            alpha = rng.random(prior.size)
            child = alpha * p1 + (1.0 - alpha) * p2
            mask = rng.random(prior.size) < (1.0 / prior.size)
            child[mask] += rng.normal(0.0, 0.12, int(np.sum(mask)))
            children.append(np.clip(child, 0.0, 1.0))
        child_scores = np.asarray([recorder.evaluate(x) for x in children])
        combined = np.vstack([population, np.asarray(children)])
        combined_scores = np.concatenate([scores, child_scores])
        keep = np.argsort(combined_scores)[::-1][:min(pop_size, len(combined_scores))]
        population, scores = combined[keep], combined_scores[keep]
    return {"population_size": pop_size, "selection": "tournament_3", "mutation_sigma": 0.12,
            "mutation_probability_per_gene": 1.0 / prior.size, "survivor": "mu_plus_lambda"}


def run_pso(
    recorder: Recorder, rng: np.random.Generator, initialization: str, prior: np.ndarray
) -> dict:
    swarm_size = min(16, recorder.remaining)
    positions = initial_population(rng, swarm_size, prior.size, initialization, prior)
    velocities = rng.uniform(-0.1, 0.1, size=positions.shape)
    scores = np.asarray([recorder.evaluate(x) for x in positions])
    personal = positions.copy()
    personal_scores = scores.copy()
    global_best = personal[int(np.argmax(personal_scores))].copy()
    while recorder.remaining:
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = 0.7298 * velocities + 1.49618 * r1 * (personal - positions) + 1.49618 * r2 * (global_best - positions)
        velocities = np.clip(velocities, -0.2, 0.2)
        positions = np.clip(positions + velocities, 0.0, 1.0)
        batch = min(swarm_size, recorder.remaining)
        new_scores = np.asarray([recorder.evaluate(x) for x in positions[:batch]])
        improved = new_scores > personal_scores[:batch]
        personal[:batch][improved] = positions[:batch][improved]
        personal_scores[:batch][improved] = new_scores[improved]
        global_best = personal[int(np.argmax(personal_scores))].copy()
    return {"swarm_size": swarm_size, "inertia": 0.7298, "cognitive": 1.49618,
            "social": 1.49618, "velocity_clip": 0.2}


RUNNERS: dict[str, Callable] = {
    "random_search": run_random_search,
    "cma_es": run_cma_es,
    "ga": run_ga,
    "pso": run_pso,
}


def run_one(
    method: str,
    initialization: str,
    run_id: int,
    out_root: Path,
    budget: int,
    prior_root: Path | None,
) -> dict:
    cfg = compact_config()
    prior, prior_meta = load_prior(initialization, run_id, prior_root, cfg)
    protocol = Protocol(budget=budget, shared_prior_name=prior_meta["name"])
    run_dir = out_root / f"budget_{budget}" / f"run_{run_id}" / method / initialization
    history_path = run_dir / "history.json"
    if history_path.exists():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        if int(existing.get("objective_evaluations", 0)) == budget:
            return {"run": str(run_dir.relative_to(ROOT)), "status": "skip", "best": existing["best_score"]}
        raise RuntimeError(f"refusing to overwrite incomplete run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(run_id)
    recorder = Recorder(cfg, budget)
    started = time.time()
    hyperparameters = RUNNERS[method](recorder, rng, initialization, prior)
    elapsed = time.time() - started
    curve = [float(row["best_so_far"]) for row in recorder.rows]
    checkpoints = {str(k): curve[k - 1] for k in CHECKPOINTS if k <= budget}
    best_index = int(np.argmax([row["score"] for row in recorder.rows]))
    first_easy = next((row["evaluation"] for row in recorder.rows if row["best_so_far"] >= protocol.target_easy), None)
    first_hard = next((row["evaluation"] for row in recorder.rows if row["best_so_far"] >= protocol.target_hard), None)
    payload = {
        "protocol": asdict(protocol),
        "method": method,
        "initialization": initialization,
        "run_id": run_id,
        "rng_seed": run_id,
        "objective_evaluations": len(recorder.rows),
        "checkpoints": checkpoints,
        "best_score": curve[-1],
        "best_evaluation": recorder.rows[best_index]["evaluation"],
        "best_x": recorder.rows[best_index]["x"],
        "evals_to_25_0": first_easy,
        "evals_to_25_25": first_hard,
        "shared_prior": {
            **prior_meta,
            "used": initialization != "random_vector",
        },
        "hyperparameters": hyperparameters,
        "elapsed_seconds": elapsed,
        "history": recorder.rows,
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / "best_x.json").write_text(json.dumps(payload["best_x"], indent=2) + "\n", encoding="utf-8")
    return {"run": str(run_dir.relative_to(ROOT)), "status": "ok", "best": payload["best_score"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="runs/ccaa_classical")
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--initializations", default="random_vector,shared_analytical_prior")
    parser.add_argument("--prior-root", default=None,
                        help="direct child of runs/ containing strict run_N.json priors")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_root = (ROOT / args.out_root).resolve()
    runs_root = (ROOT / "runs").resolve()
    if out_root.parent != runs_root:
        raise ValueError(f"--out-root must be a direct child of {runs_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    run_ids = list(range(args.num_runs))
    methods = [value for value in args.methods.split(",") if value]
    initializations = [value for value in args.initializations.split(",") if value]
    if not set(methods).issubset(METHODS):
        raise ValueError(f"unknown method in {methods}")
    if not set(initializations).issubset(INITIALIZATIONS):
        raise ValueError(f"unknown initialization in {initializations}")
    prior_root = None
    if args.prior_root:
        prior_root = (ROOT / args.prior_root).resolve()
        if prior_root.parent != runs_root or not prior_root.is_dir():
            raise ValueError(f"--prior-root must be an existing direct child of {runs_root}")
    if "shared_llm_first_prior" in initializations and prior_root is None:
        raise ValueError("--prior-root is required for shared_llm_first_prior")
    if prior_root is not None:
        run_ids = prior_run_ids(prior_root)

    cfg = compact_config()
    prior = analytical_prior(cfg)
    manifest = {
        "protocol": asdict(Protocol(budget=args.budget)),
        "methods": methods,
        "initializations": initializations,
        "run_ids": run_ids,
        "n_runs": len(methods) * len(initializations) * len(run_ids),
        "objective_budget_per_run": args.budget,
        "model_api_required": False,
        "evaluator": "examples.antenna_physics.evaluate_design",
        "config": {"ring_counts": list(cfg.ring_counts), "fnbw_target_deg": cfg.fnbw_target_deg},
        "shared_prior": {
            "kind": (
                "per-run frozen LLM first valid strict-36 proposal"
                if "shared_llm_first_prior" in initializations
                else "fixed Taylor analytical prior"
            ),
            "prior_root": str(prior_root.relative_to(ROOT)) if prior_root else None,
            "analytical_sha256_f64le": vector_sha256(prior),
            "analytical_score": float(evaluate_design(prior, cfg)["score"]),
        },
    }
    (out_root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    jobs = [(method, initialization, run_id) for run_id in run_ids for method in methods for initialization in initializations]
    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, method, initialization, run_id, out_root, args.budget, prior_root):
            (method, initialization, run_id)
            for method, initialization, run_id in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: row["run"])
    (out_root / "campaign_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
