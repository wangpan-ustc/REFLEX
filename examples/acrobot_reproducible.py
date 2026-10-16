"""Reproducible Acrobot evaluation with per-episode RNG seeding and fresh state."""
from __future__ import annotations

import random
import threading
import time
import traceback
from contextlib import contextmanager

import numpy as np

from examples.acrobot_evaluator import DEFAULT_MAX_STEPS, _be_b64, _rollout_one, _score
from reflex.randomness import environment_seeds

_RNG_LOCK = threading.RLock()


@contextmanager
def policy_rng(seed: int):
    """Restore outer search RNG state after seeding per-episode policy RNGs."""
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            np.random.seed(int(seed) % (2**32))
            random.seed(int(seed))
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def evaluate_acrobot_code_reproducible(
    code: str,
    training_seeds=None,
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    render_be: bool = True,
):
    """Evaluate each environment seed with a freshly compiled policy and seeded RNG."""
    seeds = environment_seeds(training_seeds)
    if not seeds or max_steps < 1:
        raise ValueError("nonempty environment seeds and positive max_steps required")
    try:
        rollouts = []
        for seed in seeds:
            with policy_rng(seed):
                namespace = {}
                exec(code, namespace)
                fn = namespace.get("choose_action")
                if not callable(fn):
                    return float("-inf"), {}, "no choose_action in code", None
                started = time.monotonic()
                rollout = _rollout_one(fn, seed, max_steps=max_steps)
                rollout["wall"] = time.monotonic() - started
                rollouts.append(rollout)
        score, metrics = _score(rollouts)
        metrics["evaluation_protocol"] = "acrobot-policy-rng-v2"
        metrics["episode_steps"] = [int(r["n_steps"]) for r in rollouts]
        metrics["episode_solved"] = [bool(r["solved"]) for r in rollouts]
        return float(score), metrics, "", _be_b64(rollouts, score) if render_be else None
    except Exception:
        return float("-inf"), {}, traceback.format_exc(limit=4), None
