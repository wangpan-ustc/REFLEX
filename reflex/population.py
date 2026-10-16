"""Lightweight individual / population container.

Kept minimal so REFLEX can wrap any existing evaluator (e.g. MLES tasks)
without inheriting their object hierarchy.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    code: str
    score: float = float("-inf")
    concept: str = ""
    image_b64: str | None = None
    be_summary: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] | None = None
    parent_id: str | None = None
    skill_ids_used: list[str] = field(default_factory=list)
    operator: str = ""
    gen: int = 0
    iid: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.score > float("-inf") and bool(self.code)


class Population:
    """Bounded elitist population.

    Behaviour mirrors a (mu+lambda) ES: insertion always succeeds; if
    the size exceeds `capacity`, the worst individual is dropped.
    """

    def __init__(self, capacity: int = 16, rng: random.Random | None = None):
        self.capacity = capacity
        self._items: list[Individual] = []
        self._rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def add(self, ind: Individual) -> bool:
        if not ind.is_valid():
            return False
        # Reject exact-duplicate code: re-adding an identical policy wastes a
        # slot and degrades diversity-aware parent sampling.
        if any(x.code == ind.code for x in self._items):
            return False
        self._items.append(ind)
        self._items.sort(key=lambda x: x.score, reverse=True)
        if len(self._items) > self.capacity:
            self._items = self._items[: self.capacity]
        return True

    def best(self) -> Individual | None:
        return self._items[0] if self._items else None

    def topk(self, k: int) -> list[Individual]:
        return list(self._items[:k])

    def worst(self) -> Individual | None:
        return self._items[-1] if self._items else None

    def sample(self, n: int) -> list[Individual]:
        if not self._items:
            return []
        return self._rng.sample(self._items, k=min(n, len(self._items)))

    def diverse_sample(self, n: int) -> list[Individual]:
        """Greedy max-diversity subset by concept token overlap.

        Crossover works best when the parents carry genuinely different
        mechanisms; random sampling often returns siblings from the same
        conceptual family. Start from the best individual, then repeatedly
        add the member whose concept is least similar (Jaccard over word
        tokens) to those already picked.
        """
        if len(self._items) <= n:
            return list(self._items)

        def toks(ind: Individual) -> set[str]:
            return {w.lower() for w in (ind.concept or ind.code[:200]).split() if len(w) > 3}

        picked = [self._items[0]]
        rest = list(self._items[1:])
        while len(picked) < n and rest:
            picked_toks = [toks(p) for p in picked]

            def dissim(ind: Individual) -> float:
                t = toks(ind)
                if not t:
                    return 0.0
                worst = 0.0
                for pt in picked_toks:
                    inter = len(t & pt)
                    union = len(t | pt) or 1
                    worst = max(worst, inter / union)
                return -worst  # less similar -> larger value

            rest.sort(key=dissim, reverse=True)
            picked.append(rest.pop(0))
        return picked

    def tournament(self, k: int = 3) -> Individual | None:
        if not self._items:
            return None
        cand = self._rng.sample(self._items, k=min(k, len(self._items)))
        return max(cand, key=lambda x: x.score)

    def to_records(self) -> list[dict[str, Any]]:
        out = []
        for ind in self._items:
            out.append(
                {
                    "iid": ind.iid,
                    "score": ind.score,
                    "concept": ind.concept,
                    "operator": ind.operator,
                    "gen": ind.gen,
                    "parent_id": ind.parent_id,
                    "skill_ids_used": list(ind.skill_ids_used),
                }
            )
        return out
