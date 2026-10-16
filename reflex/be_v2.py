"""Behavioral Evidence v2.

A train-free upgrade over MLES single-frame BE. Three improvements:

1. **Keyframe selection by reward-delta events**:
   Pick K frames where |Δreward| is locally maximal. This captures the
   moments where the policy's behaviour qualitatively shifts.

2. **Multi-panel overlay**:
   - top-left  : trajectory or rendered keyframe collage
   - top-right : score / reward curve with keyframes marked
   - bot-left  : action timeline (per-dim plot)
   - bot-right : code-line attribution histogram

3. **Code-line attribution** via lightweight `sys.settrace`:
   counts how many times each LINE of the policy function executed
   during the rollout. The histogram is overlaid as a small panel so
   the Critic can localise problems to specific code lines.

The output is a base64-encoded PNG plus a textual `summary` dict (used by
the Referee to compare two evidences without re-uploading images).

This module is designed to be reused for any task: gym envs, antenna
design, or any custom evaluator. Tasks supply per-step records; BE v2
turns them into a single image + a summary dict.
"""
from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import b64_image_from_pil  # noqa: E402


@dataclass
class EvidenceFrame:
    """One time-step record fed into the BE builder."""

    t: int
    reward: float = 0.0
    action: np.ndarray | list[float] | None = None
    image: np.ndarray | None = None  # HxWx3 uint8, optional
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Code-line attribution tracer
# ---------------------------------------------------------------------------
class LineTracer:
    """Counts line-hits within a target file/function during execution.

    Train-free: pure `sys.settrace`. Adds <5% overhead in our experiments.
    """

    def __init__(self, target_filename: str | None = None,
                 target_funcname: str | None = None):
        self.target_filename = target_filename
        self.target_funcname = target_funcname
        self.counts: dict[int, int] = {}
        self._prev_trace = None

    def _tracer(self, frame, event, arg):
        if event != "line":
            return self._tracer
        co = frame.f_code
        if self.target_filename and self.target_filename not in co.co_filename:
            return self._tracer
        if self.target_funcname and co.co_name != self.target_funcname:
            return self._tracer
        ln = frame.f_lineno
        self.counts[ln] = self.counts.get(ln, 0) + 1
        return self._tracer

    @contextmanager
    def __call__(self):
        self._prev_trace = sys.gettrace()
        sys.settrace(self._tracer)
        try:
            yield self
        finally:
            sys.settrace(self._prev_trace)


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------
def select_keyframes(
    rewards: list[float], k: int = 4, min_gap: int = 5
) -> list[int]:
    """Select indices of frames with locally maximal |Δreward|.

    Returns at most k indices (sorted ascending). Train-free.
    """
    if len(rewards) <= 1:
        return list(range(len(rewards)))
    deltas = np.abs(np.diff(np.asarray(rewards, dtype=float)))
    if deltas.size == 0:
        return [0]
    order = np.argsort(deltas)[::-1]
    chosen: list[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= k:
            break
    if not chosen:
        chosen = [int(np.argmax(deltas))]
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class BehavioralEvidenceV2:
    """Convert per-step records into a single BE image + summary dict."""

    def __init__(
        self,
        n_keyframes: int = 4,
        action_labels: tuple[str, ...] = ("steering", "gas", "brake"),
        figsize: tuple[float, float] = (12.0, 8.0),
    ):
        self.n_keyframes = n_keyframes
        self.action_labels = action_labels
        self.figsize = figsize

    def build(
        self,
        frames: list[EvidenceFrame],
        line_counts: dict[int, int] | None = None,
        title: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Return (base64-PNG, summary-dict)."""
        if not frames:
            return self._empty(), {"empty": True}

        rewards = [f.reward for f in frames]
        keyframe_idx = select_keyframes(rewards, k=self.n_keyframes)
        actions = np.array(
            [
                np.asarray(f.action, dtype=float)
                if f.action is not None
                else np.zeros(len(self.action_labels))
                for f in frames
            ]
        )
        if actions.ndim == 1:
            actions = actions[:, None]

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        if title:
            fig.suptitle(title, fontsize=11)

        # ---------- top-left: keyframe collage or trajectory ----------
        ax_tl = axes[0, 0]
        keyframes_with_image = [i for i in keyframe_idx if frames[i].image is not None]
        if keyframes_with_image:
            n = len(keyframes_with_image)
            sample = frames[keyframes_with_image[0]].image
            h, w = sample.shape[:2]
            collage = np.zeros((h, w * n, 3), dtype=np.uint8)
            for j, idx in enumerate(keyframes_with_image):
                im = frames[idx].image
                if im.shape[:2] != (h, w):
                    continue
                collage[:, j * w : (j + 1) * w] = im
                ax_tl.text(
                    j * w + 5,
                    12,
                    f"t={frames[idx].t}",
                    color="yellow",
                    fontsize=8,
                )
            ax_tl.imshow(collage)
            ax_tl.set_title(f"Keyframes ({n})")
        elif "xy" in frames[0].extra:
            xs = [f.extra.get("xy", (0, 0))[0] for f in frames]
            ys = [f.extra.get("xy", (0, 0))[1] for f in frames]
            ax_tl.plot(xs, ys, "-", color="steelblue", lw=1)
            ax_tl.scatter(
                [xs[i] for i in keyframe_idx],
                [ys[i] for i in keyframe_idx],
                c="red",
                s=40,
                zorder=5,
            )
            ax_tl.set_title("Trajectory")
            ax_tl.set_aspect("equal", adjustable="datalim")
        else:
            ax_tl.text(0.5, 0.5, "(no visual frames)", ha="center", va="center")
            ax_tl.set_title("Trajectory / Keyframes")
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])

        # ---------- top-right: reward / score curve ----------
        ax_tr = axes[0, 1]
        ax_tr.plot(rewards, color="seagreen", lw=1.4)
        ax_tr.set_title("Reward / score over time")
        ax_tr.set_xlabel("step")
        ax_tr.set_ylabel("reward")
        for k in keyframe_idx:
            ax_tr.axvline(k, color="red", alpha=0.3, lw=1)

        # ---------- bottom-left: action timeline ----------
        ax_bl = axes[1, 0]
        n_act = actions.shape[1]
        for d in range(n_act):
            label = (
                self.action_labels[d]
                if d < len(self.action_labels)
                else f"a[{d}]"
            )
            ax_bl.plot(actions[:, d], lw=1, label=label)
        ax_bl.legend(fontsize=8, loc="best")
        ax_bl.set_title("Action timeline")
        ax_bl.set_xlabel("step")

        # ---------- bottom-right: code-line attribution ----------
        ax_br = axes[1, 1]
        if line_counts:
            lines = sorted(line_counts.keys())
            counts = [line_counts[l] for l in lines]
            ax_br.barh([str(l) for l in lines], counts, color="slategray")
            ax_br.set_title("Code-line hit counts")
            ax_br.set_xlabel("hits")
            ax_br.invert_yaxis()
        else:
            ax_br.text(0.5, 0.5, "(no line trace)", ha="center", va="center")
            ax_br.set_title("Code-line attribution")
            ax_br.set_xticks([])
            ax_br.set_yticks([])

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image

        img = Image.open(buf)
        b64 = b64_image_from_pil(img)

        summary = self._summarize(frames, actions, rewards, keyframe_idx, line_counts)
        return b64, summary

    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(
        frames: list[EvidenceFrame],
        actions: np.ndarray,
        rewards: list[float],
        keyframe_idx: list[int],
        line_counts: dict[int, int] | None,
    ) -> dict[str, Any]:
        rew = np.asarray(rewards, dtype=float)
        return {
            "horizon": len(frames),
            "total_reward": float(rew.sum()),
            "min_reward": float(rew.min()) if rew.size else 0.0,
            "max_reward": float(rew.max()) if rew.size else 0.0,
            "reward_volatility": float(np.std(np.diff(rew))) if rew.size > 1 else 0.0,
            "action_dim": int(actions.shape[1]),
            "action_mean": [float(x) for x in actions.mean(axis=0)],
            "action_std": [float(x) for x in actions.std(axis=0)],
            "keyframes": [
                {"t": int(frames[i].t), "reward": float(frames[i].reward)}
                for i in keyframe_idx
            ],
            "n_distinct_lines_hit": len(line_counts or {}),
            "hottest_line": (
                max(line_counts.items(), key=lambda kv: kv[1])[0] if line_counts else None
            ),
        }

    @staticmethod
    def _empty() -> str:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no behavioral evidence", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image

        return b64_image_from_pil(Image.open(buf))


# ---------------------------------------------------------------------------
# Counterfactual rollout helper (optional, plugged into evaluators)
# ---------------------------------------------------------------------------
def counterfactual_rollouts(
    rollout_fn: Callable[..., list[EvidenceFrame]],
    parent_frames: list[EvidenceFrame],
    perturb_fn: Callable[[np.ndarray], np.ndarray],
    n_branches: int = 2,
    branch_horizon: int = 10,
) -> dict[int, list[list[EvidenceFrame]]]:
    """For each keyframe, run `n_branches` perturbed rollouts of length
    `branch_horizon`, returning {keyframe_t: [branch_frames, ...]}.

    `rollout_fn(start_state, action_seq) -> frames` is supplied by the
    task-side evaluator. This helper itself is task-agnostic.
    """
    rewards = [f.reward for f in parent_frames]
    keyframes = select_keyframes(rewards, k=4)
    branches: dict[int, list[list[EvidenceFrame]]] = {}
    for kf in keyframes:
        if "state" not in parent_frames[kf].extra:
            continue
        start_state = parent_frames[kf].extra["state"]
        kf_branches: list[list[EvidenceFrame]] = []
        for _ in range(n_branches):
            action_seq = []
            for j in range(branch_horizon):
                idx = min(kf + j, len(parent_frames) - 1)
                a = parent_frames[idx].action
                action_seq.append(perturb_fn(np.asarray(a, dtype=float)))
            try:
                kf_branches.append(rollout_fn(start_state, action_seq))
            except Exception:
                continue
        if kf_branches:
            branches[kf] = kf_branches
    return branches
