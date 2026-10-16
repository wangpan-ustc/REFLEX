"""Self-contained Lunar Lander evaluator (mirrors MLES MoonLanderEvaluation).

Reproduces MLES's exact rollout + trajectory-canvas + base64-PNG pipeline so
results are directly comparable to the MLES ICLR-2026 paper, but without
importing `llm4ad` (which transitively requires codebleu / datasets / torch).

Requires only:
  - gymnasium[box2d]   (for LunarLander-v3)
  - matplotlib         (already used by run_antenna)
  - numpy

Returns (score, metrics, error_str, image_b64) — the same tuple shape as
`run_antenna.evaluate_code`.

Score is the MLES NWS:
  NWS = 0.6 * mean_reward/200
      + 0.2 * (1 - min(mean_fuel/100, 1))
      + 0.2 * success_rate          (success := episode_reward >= 200)
"""
from __future__ import annotations

import base64
import copy
import io
import time
import traceback
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reflex.randomness import environment_seeds

ENV_NAME = "LunarLander-v3"
MAX_STEPS = 200


def _rollout_one(
    fn: Callable, env_seed: int, max_steps: int = MAX_STEPS
) -> tuple[dict[str, Any], np.ndarray]:
    import gymnasium as gym

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, _ = env.reset(seed=env_seed)
    action = 0
    episode_reward = 0.0
    episode_fuel = 0
    canvas = np.zeros((400, 600, 3), dtype=np.float32)
    observations = []

    pre_obs = copy.deepcopy(obs)
    obs, reward, done, truncated, info = env.step(action)

    flash = 0
    for i in range(max_steps + 1):
        action = int(fn(obs, action, pre_obs))
        pre_obs = copy.deepcopy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += float(reward)
        if action in (1, 2, 3):
            episode_fuel += 1
        if flash >= 10:
            img = env.render()
            mask = np.any(img != [0, 0, 0], axis=-1)
            alpha = min(i / max_steps, 1.0)
            canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
            observations.append([float(x) for x in obs])
            flash = 0
        flash += 1
        if done or truncated or i == max_steps:
            img = env.render()
            mask = np.any(img != [0, 0, 0], axis=-1)
            alpha = min(i / max_steps, 1.0)
            canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
            env.close()
            return (
                {
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "episode_fuel": episode_fuel,
                    "episode_reward": episode_reward,
                    "n_steps": i + 1,
                },
                canvas,
            )
    env.close()
    return ({"done": False, "truncated": True, "episode_fuel": episode_fuel,
             "episode_reward": episode_reward, "n_steps": max_steps}, canvas)


def _canvas_to_b64(canvas: np.ndarray, score: float, info: dict) -> str:
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.imshow(np.clip(canvas, 0, 255).astype(np.uint8))
    final_state = (
        "Landed safely" if info.get("done") else
        "Crashed" if info.get("truncated") else
        "Landing failed"
    )
    plt.title(f"Lander Trajectory ({info.get('n_steps', '?')} steps)\n"
              f"NWS: {score:.3f} | reward: {info.get('episode_reward', float('nan')):.1f} | {final_state}")
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def evaluate_lander_code(
    code: str, training_seeds: tuple[int, ...] | None = None, max_steps: int = MAX_STEPS
) -> tuple[float, dict[str, Any], str, str | None]:
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)
        fn = ns.get("choose_action")
        if not callable(fn):
            return float("-inf"), {}, "no `choose_action` in code", None
        rewards: list[float] = []
        fuels: list[int] = []
        successes = 0
        per_seed_info: list[dict] = []
        canvases = []
        for s in environment_seeds(training_seeds, count=1):
            t0 = time.time()
            info, canvas = _rollout_one(fn, int(s), max_steps=max_steps)
            info["wall"] = time.time() - t0
            rewards.append(info["episode_reward"])
            fuels.append(info["episode_fuel"])
            if info["episode_reward"] >= 200:
                successes += 1
            per_seed_info.append(info)
            canvases.append(canvas)
        n = len(rewards)
        mean_reward = float(np.mean(rewards))
        mean_fuel = float(np.mean(fuels))
        success_rate = successes / n
        nws = (mean_reward / 200) * 0.6 + (1 - min(mean_fuel / 100, 1)) * 0.2 + success_rate * 0.2
        worst_idx = int(np.argmin(rewards))
        img_b64 = _canvas_to_b64(canvases[worst_idx], nws, per_seed_info[worst_idx])
        metrics = {
            "score": nws,
            "mean_reward": mean_reward,
            "mean_fuel": mean_fuel,
            "success_rate": success_rate,
            "per_seed_reward": rewards,
            "per_seed_done": [p["done"] for p in per_seed_info],
            "n_seeds": n,
        }
        return float(nws), metrics, "", img_b64
    except Exception:
        return float("-inf"), {}, traceback.format_exc(limit=4), None


__all__ = ["evaluate_lander_code"]
