"""Unit tests for the UCB1 operator-selection bandit.

Run with:  python -m pytest tests/test_bandit.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from reflex.bandit import UCB1  # noqa: E402


def test_plays_every_arm_once_first():
    ucb = UCB1(["a", "b", "c"])
    first = [ucb.select() for _ in range(3)]
    for arm in first:
        ucb.update(arm, 0.0)
    assert set(first) == {"a", "b", "c"}


def test_converges_to_best_arm():
    ucb = UCB1(["good", "bad"], c=0.5)
    for _ in range(60):
        arm = ucb.select()
        ucb.update(arm, 1.0 if arm == "good" else 0.0)
    picks = [ucb.select() for _ in range(10)]
    # selection without update does not change means; counts/t drift is fine
    for a in picks:
        ucb.update(a, 1.0 if a == "good" else 0.0)
    assert picks.count("good") >= 8
    assert ucb.stats()["good"]["mean_reward"] > ucb.stats()["bad"]["mean_reward"]


def test_reward_normalization_across_scales():
    # Antenna-scale deltas (~1.0) and NWS-scale deltas (~0.01) both map into [0,1].
    ucb = UCB1(["x", "y"])
    ucb.update("x", 1.0)     # sets adaptive delta_max to 1.0
    ucb.update("y", 0.01)
    s = ucb.stats()
    assert 0.0 <= s["x"]["mean_reward"] <= 1.0
    assert 0.0 <= s["y"]["mean_reward"] <= 1.0
    assert s["x"]["mean_reward"] > s["y"]["mean_reward"]


def test_forced_and_failed_updates():
    ucb = UCB1(["m_diag", "m_text"])
    ucb.update("m_diag_forced", 0.5)   # forced suffix folds into base arm
    assert ucb.stats()["m_diag"]["n"] == 1
    ucb.update("m_text", None)         # failure counts as reward 0
    ucb.update("m_text", float("nan"))
    s = ucb.stats()
    assert s["m_text"]["n"] == 2
    assert s["m_text"]["mean_reward"] == 0.0


def test_negative_delta_is_zero_reward():
    ucb = UCB1(["a"])
    ucb.update("a", -5.0)
    assert ucb.stats()["a"]["mean_reward"] == 0.0


if __name__ == "__main__":
    test_plays_every_arm_once_first()
    test_converges_to_best_arm()
    test_reward_normalization_across_scales()
    test_forced_and_failed_updates()
    test_negative_delta_is_zero_reward()
    print("All bandit tests passed.")
