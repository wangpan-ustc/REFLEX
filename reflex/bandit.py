"""UCB1 operator-selection bandit.

Implements the bounded-reward UCB1 described in the REFLEX paper
(Algorithm 1 line 6 / Eq. 2): at step t select the operator o maximizing

    U_t(o) = mu_o + C * sqrt(ln t / N_o)

Raw fitness deltas are clipped to [0, delta_max] and normalized to [0, 1]
before updating mu_o, satisfying UCB1's bounded-reward assumption.
delta_max adapts online (running max of observed positive deltas), so the
same bandit works across tasks with very different score scales
(NWS ~ 1.0, pendulum ~ -1.0, antenna ~ 25).

Thread-safe: runners issue concurrent proposals from a thread pool.
"""
from __future__ import annotations

import math
import threading
from typing import Any


class UCB1:
    def __init__(self, arms: list[str], c: float = 1.2, delta_max: float | None = None):
        if not arms:
            raise ValueError("UCB1 needs at least one arm")
        self.arms = list(arms)
        self.c = c
        self.counts: dict[str, int] = {a: 0 for a in self.arms}
        self.means: dict[str, float] = {a: 0.0 for a in self.arms}
        # Selections handed out but not yet updated (concurrent proposals);
        # counted in the exploration term so parallel workers spread out.
        self._pending: dict[str, int] = {a: 0 for a in self.arms}
        self.t = 0
        # None -> adapt from data; start tiny so first improvement saturates.
        self._delta_max = delta_max
        self._adaptive = delta_max is None
        self._delta_seen = 1e-9
        self._lock = threading.Lock()

    def select(self) -> str:
        with self._lock:
            self.t += 1
            for a in self.arms:  # play each arm once first
                if self.counts[a] + self._pending[a] == 0:
                    self._pending[a] += 1
                    return a
            log_t = math.log(max(self.t, 2))
            pick = max(
                self.arms,
                key=lambda a: self.means[a]
                + self.c * math.sqrt(log_t / (self.counts[a] + self._pending[a])),
            )
            self._pending[pick] += 1
            return pick

    def update(self, arm: str, raw_delta: float | None) -> None:
        """Record the outcome of one pull.

        raw_delta: child_fitness - parent_fitness. None / non-finite
        (LLM failure, crashed evaluation) counts as reward 0.
        """
        base = arm.removesuffix("_forced")
        with self._lock:
            if base not in self.counts:
                self.arms.append(base)
                self.counts[base] = 0
                self.means[base] = 0.0
                self._pending[base] = 0
            self._pending[base] = max(0, self._pending[base] - 1)
            if raw_delta is not None and math.isfinite(raw_delta) and raw_delta > 0:
                self._delta_seen = max(self._delta_seen, raw_delta)
            dmax = self._delta_seen if self._adaptive else (self._delta_max or 1.0)
            if raw_delta is None or not math.isfinite(raw_delta):
                reward = 0.0
            else:
                reward = min(max(raw_delta, 0.0), dmax) / dmax
            self.counts[base] += 1
            self.means[base] += (reward - self.means[base]) / self.counts[base]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                a: {"n": self.counts[a], "mean_reward": round(self.means[a], 4)}
                for a in self.arms
            }
