"""Self-contained Acrobot-v1 evaluator with BE trajectory image.

Mirrors the spirit of MLES's AcrobotEvaluation but adds a multi-panel
Behavioral Evidence image (joint-angle traces, phase portrait, action
timeline) so that REFLEX's Critic has a visual signal to diagnose.

Score (higher better, optimum ≈ 0, worst ≈ -2):
  - solved (upper link reaches target):
        score = -(steps_to_goal / max_steps)
  - not solved:
        score = -1 - mean_distance_to_goal_height_violation

Requires `gymnasium` (Acrobot-v1 is included in the base install — no
box2d needed).
"""
from __future__ import annotations

import base64
import io
import time
import traceback
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reflex.randomness import environment_seeds

ENV_NAME = "Acrobot-v1"
DEFAULT_MAX_STEPS = 500
GOAL_HEIGHT = 1.0  # standard Acrobot goal: -cos(theta1) - cos(theta1+theta2) > 1


def _rollout_one(
    fn: Callable, env_seed: int, max_steps: int = DEFAULT_MAX_STEPS
) -> dict[str, Any]:
    import gymnasium as gym

    env = gym.make(ENV_NAME)
    env._max_episode_steps = max_steps
    obs, _ = env.reset(seed=env_seed)
    action = 0
    traj_obs = [list(map(float, obs))]
    actions: list[int] = []
    height: list[float] = []
    n_steps = 0
    solved = False
    for i in range(max_steps + 1):
        action = int(fn(float(obs[0]), float(obs[1]),
                        float(obs[2]), float(obs[3]),
                        float(obs[4]), float(obs[5]), int(action)))
        obs, reward, done, truncated, _ = env.step(action)
        traj_obs.append(list(map(float, obs)))
        actions.append(action)
        ct1, st1, ct2, st2 = obs[0], obs[1], obs[2], obs[3]
        h = -ct1 - (ct1 * ct2 - st1 * st2)
        height.append(float(h))
        n_steps = i + 1
        if done:
            solved = True
            break
        if truncated or i == max_steps:
            break
    env.close()
    traj = np.array(traj_obs)
    return {
        "solved": solved,
        "n_steps": n_steps,
        "max_steps": max_steps,
        "traj": traj,
        "actions": np.array(actions, dtype=int),
        "height": np.array(height, dtype=float),
    }


def _be_b64(rollouts: list[dict], score: float) -> str:
    fig = plt.figure(figsize=(11.0, 6.5))
    worst = min(rollouts, key=lambda r: r["solved"] * 1e6 - r["n_steps"])
    traj = worst["traj"]
    actions = worst["actions"]
    height = worst["height"]
    t = np.arange(traj.shape[0])

    ax_h = fig.add_subplot(2, 2, 1)
    ax_h.plot(t[:len(height)], height, "C0-", lw=1.0, label="tip height")
    ax_h.axhline(GOAL_HEIGHT, color="g", ls="--", lw=0.8, label=f"goal = {GOAL_HEIGHT}")
    ax_h.axhline(0.0, color="0.6", ls=":", lw=0.6)
    ax_h.set_xlabel("step"); ax_h.set_ylabel("tip height")
    ax_h.set_title(f"worst episode  steps={worst['n_steps']}  "
                   f"solved={worst['solved']}  score={score:+.3f}")
    ax_h.legend(loc="lower right", fontsize=8); ax_h.grid(alpha=0.3)

    ax_phase = fig.add_subplot(2, 2, 2)
    th1 = np.arctan2(traj[:, 1], traj[:, 0])
    av1 = traj[:, 4]
    ax_phase.plot(th1, av1, "C2-", lw=0.7, alpha=0.8)
    ax_phase.scatter(th1[:1], av1[:1], c="g", s=30, label="start", zorder=5)
    ax_phase.scatter(th1[-1:], av1[-1:], c="r", s=30, label="end", zorder=5)
    ax_phase.set_xlabel("theta1 (rad)"); ax_phase.set_ylabel("d theta1 / dt")
    ax_phase.set_title("phase portrait (joint 1)")
    ax_phase.legend(loc="upper right", fontsize=8); ax_phase.grid(alpha=0.3)

    ax_act = fig.add_subplot(2, 1, 2)
    if actions.size:
        ax_act.step(np.arange(actions.size), actions - 1, where="post", color="C3", lw=0.8)
    ax_act.set_yticks([-1, 0, 1])
    ax_act.set_yticklabels(["torque -1", "torque 0", "torque +1"])
    ax_act.set_xlabel("step")
    ax_act.set_title("action timeline (worst seed)")
    ax_act.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _score(rollouts: list[dict]) -> tuple[float, dict]:
    n_solved = sum(int(r["solved"]) for r in rollouts)
    n = len(rollouts)
    if n_solved > 0:
        scores = []
        for r in rollouts:
            if r["solved"]:
                scores.append(-(r["n_steps"] / r["max_steps"]))
            else:
                viol = max(0.0, GOAL_HEIGHT - float(np.max(r["height"])) if r["height"].size else 1.0)
                scores.append(-1.0 - viol)
        score = float(np.mean(scores))
    else:
        viol = []
        for r in rollouts:
            v = max(0.0, GOAL_HEIGHT - float(np.max(r["height"])) if r["height"].size else 1.0)
            viol.append(v)
        score = -1.0 - float(np.mean(viol))
    metrics = {
        "score": score,
        "solve_rate": n_solved / n,
        "mean_steps_to_goal": float(np.mean([r["n_steps"] for r in rollouts if r["solved"]]))
            if n_solved else float("nan"),
        "mean_max_height": float(np.mean([float(np.max(r["height"])) if r["height"].size else 0.0 for r in rollouts])),
        "n_seeds": n,
    }
    return score, metrics


def evaluate_acrobot_code(
    code: str, training_seeds: tuple[int, ...] | None = None, max_steps: int = DEFAULT_MAX_STEPS
) -> tuple[float, dict[str, Any], str, str | None]:
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)
        fn = ns.get("choose_action")
        if not callable(fn):
            return float("-inf"), {}, "no `choose_action` in code", None
        rollouts = []
        for s in environment_seeds(training_seeds, count=1):
            t0 = time.time()
            r = _rollout_one(fn, int(s), max_steps=max_steps)
            r["wall"] = time.time() - t0
            rollouts.append(r)
        score, metrics = _score(rollouts)
        img_b64 = _be_b64(rollouts, score)
        return float(score), metrics, "", img_b64
    except Exception:
        return float("-inf"), {}, traceback.format_exc(limit=4), None


__all__ = ["evaluate_acrobot_code"]
