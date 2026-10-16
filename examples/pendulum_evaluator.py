"""Self-contained Pendulum-v1 evaluator with BE trajectory image.

Mirrors MLES's PendulumEvaluation: torque ∈ [-2, 2], 200-step episode,
final-state error scoring. Adds a multi-panel BE image (theta over time,
phase portrait, torque timeline) so the Critic has a visual signal.

Score (higher better, optimum ≈ 0, worst ≈ -3):
  - "stable upright" final state (rare): -(steps/max_steps)
  - otherwise: -(1 + |1 - cos_theta| + |sin_theta|) using mean of last 20 steps
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

ENV_NAME = "Pendulum-v1"
DEFAULT_MAX_STEPS = 200


def _rollout_one(
    fn: Callable, env_seed: int, max_steps: int = DEFAULT_MAX_STEPS
) -> dict[str, Any]:
    import gymnasium as gym

    env = gym.make(ENV_NAME)
    env._max_episode_steps = max_steps
    obs, _ = env.reset(seed=env_seed)
    action = 0.0
    traj_obs = [list(map(float, obs))]
    actions: list[float] = []
    rewards: list[float] = []
    n_steps = 0
    for i in range(max_steps + 1):
        a = fn(float(obs[0]), float(obs[1]), float(obs[2]), float(action))
        action = float(np.clip(a, -2.0, 2.0))
        obs, reward, done, truncated, _ = env.step([action])
        traj_obs.append(list(map(float, obs)))
        actions.append(action)
        rewards.append(float(reward))
        n_steps = i + 1
        if done or truncated or i == max_steps:
            break
    env.close()
    traj = np.array(traj_obs)
    last_n = max(1, min(20, traj.shape[0]))
    cos_t = float(np.mean(traj[-last_n:, 0]))
    sin_t = float(np.abs(np.mean(traj[-last_n:, 1])))
    angle_error = abs(1.0 - cos_t)
    stability_error = sin_t
    error = angle_error + stability_error
    return {
        "traj": traj,
        "actions": np.array(actions, dtype=float),
        "rewards": np.array(rewards, dtype=float),
        "n_steps": n_steps,
        "max_steps": max_steps,
        "final_cos": cos_t,
        "final_sin": sin_t,
        "error": error,
        "stable_upright": error < 0.05,
        "total_reward": float(np.sum(rewards)),
    }


def _be_b64(rollouts: list[dict], score: float) -> str:
    fig = plt.figure(figsize=(11.0, 6.5))
    worst = min(rollouts, key=lambda r: -r["error"])
    traj = worst["traj"]
    actions = worst["actions"]
    rewards = worst["rewards"]
    theta = np.arctan2(traj[:, 1], traj[:, 0])
    av = traj[:, 2]
    t = np.arange(traj.shape[0])

    ax_th = fig.add_subplot(2, 2, 1)
    ax_th.plot(t, theta, "C0-", lw=1.0, label="theta")
    ax_th.axhline(0.0, color="g", ls="--", lw=0.8, label="upright (theta=0)")
    ax_th.axhline(np.pi, color="r", ls=":", lw=0.6, label="downward (±π)")
    ax_th.axhline(-np.pi, color="r", ls=":", lw=0.6)
    ax_th.set_xlabel("step"); ax_th.set_ylabel("theta (rad)")
    ax_th.set_title(f"worst episode  err={worst['error']:.3f}  "
                    f"score={score:+.3f}  total_R={worst['total_reward']:+.0f}")
    ax_th.legend(loc="lower right", fontsize=8); ax_th.grid(alpha=0.3)

    ax_phase = fig.add_subplot(2, 2, 2)
    ax_phase.plot(theta, av, "C2-", lw=0.7, alpha=0.8)
    ax_phase.scatter(theta[:1], av[:1], c="g", s=30, label="start", zorder=5)
    ax_phase.scatter(theta[-1:], av[-1:], c="r", s=30, label="end", zorder=5)
    ax_phase.set_xlabel("theta (rad)"); ax_phase.set_ylabel("d theta / dt")
    ax_phase.set_title("phase portrait")
    ax_phase.legend(loc="upper right", fontsize=8); ax_phase.grid(alpha=0.3)

    ax_act = fig.add_subplot(2, 1, 2)
    if actions.size:
        ax_act.plot(np.arange(actions.size), actions, color="C3", lw=0.7)
        ax_act.fill_between(np.arange(actions.size), 0, actions, color="C3", alpha=0.2)
    ax_act.axhline(0.0, color="0.5", ls="--", lw=0.5)
    ax_act.set_ylim(-2.2, 2.2)
    ax_act.set_xlabel("step"); ax_act.set_ylabel("torque")
    ax_act.set_title("action (torque) timeline")
    ax_act.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _score(rollouts: list[dict]) -> tuple[float, dict]:
    scores = []
    for r in rollouts:
        if r["stable_upright"]:
            scores.append(-(r["n_steps"] / r["max_steps"]))
        else:
            scores.append(-(1.0 + r["error"]))
    score = float(np.mean(scores))
    n_stable = sum(int(r["stable_upright"]) for r in rollouts)
    metrics = {
        "score": score,
        "stable_upright_rate": n_stable / len(rollouts),
        "mean_final_error": float(np.mean([r["error"] for r in rollouts])),
        "mean_final_cos": float(np.mean([r["final_cos"] for r in rollouts])),
        "mean_total_reward": float(np.mean([r["total_reward"] for r in rollouts])),
        "n_seeds": len(rollouts),
    }
    return score, metrics


def evaluate_pendulum_code(
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


__all__ = ["evaluate_pendulum_code"]
