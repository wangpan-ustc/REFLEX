"""Self-evolving SkillMemory.

A persistent, BM25-indexed bank of reusable code snippets. The bank
manages itself through five rules, all train-free:

    add      :  Δfitness above τ_add  →  abstract elite into a Skill
    promote  :  retrieved skill helped child beat parent  →  utility ↑
    demote   :  retrieved skill produced worse child       →  utility ↓
    merge    :  cosine(BM25-bow, BM25-bow) > τ_merge AND
                same target hook  →  LLM-rewrite into a single skill
    evict    :  utility < τ_evict for ≥ N retrievals       →  archive

State is persisted as JSON on disk so memory survives across runs and
across tasks (cross-task knowledge reuse comes for free).

The primary method deliberately uses BM25 over tokenized text triggers and
tags, which keeps the framework train-free.  Ablation runs may additionally
select a deterministic local LSA retriever.  It uses only the current bank,
does not call an external embedding service, and never silently falls back to
BM25.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # tiny fallback so REFLEX runs without rank_bm25
    class BM25Okapi:  # noqa: N801 - mimic upstream class name
        """Minimal BM25 fallback (Okapi BM25, k1=1.5, b=0.75)."""

        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.k1 = k1
            self.b = b
            self.corpus = corpus
            self.N = len(corpus)
            self.avgdl = (sum(len(d) for d in corpus) / self.N) if self.N else 0.0
            self.doc_freqs: list[dict[str, int]] = []
            self.df: dict[str, int] = {}
            for doc in corpus:
                tf: dict[str, int] = {}
                for tok in doc:
                    tf[tok] = tf.get(tok, 0) + 1
                self.doc_freqs.append(tf)
                for tok in tf:
                    self.df[tok] = self.df.get(tok, 0) + 1
            self.idf: dict[str, float] = {}
            import math as _m
            for tok, df in self.df.items():
                self.idf[tok] = _m.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        def get_scores(self, query: list[str]) -> list[float]:
            scores: list[float] = []
            for i, tf in enumerate(self.doc_freqs):
                dl = sum(tf.values()) or 1
                norm = 1 - self.b + self.b * dl / (self.avgdl or 1)
                s = 0.0
                for tok in query:
                    f = tf.get(tok, 0)
                    if f == 0:
                        continue
                    idf = self.idf.get(tok, 0.0)
                    s += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
                scores.append(s)
            return scores

from . import prompts
from .llm_client import LLMClient
from .utils import dump_json, ensure_dir, load_json, safe_json_loads, short_hash, utc_ts


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _cosine_sim(a, b) -> float:
    """Cosine similarity for two vectors (lists or np arrays)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class Skill:
    id: str
    trigger: str
    snippet: str
    tags: list[str] = field(default_factory=list)
    target_hook: str = ""  # e.g. function name it patches into
    utility: float = 1.0
    n_retrieved: int = 0
    n_helped: int = 0
    n_hurt: int = 0
    born_at: str = ""
    last_used: str = ""
    parent_iid: str = ""
    archived: bool = False
    artifact_type: str = "executable_fragment"
    retrieval_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(**d)

    def doc_text(self) -> str:
        """The text fed to BM25 for retrieval."""
        if self.retrieval_text:
            return self.retrieval_text
        return " ".join(
            [
                self.trigger,
                " ".join(self.tags),
                self.target_hook,
                # include first non-empty code line as a weak hint
                next(
                    (
                        line.strip()
                        for line in self.snippet.splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ),
                    "",
                ),
            ]
        )


class SkillMemory:
    """Self-evolving skill bank.

    Thread-safe: a single mutex guards add/update/merge/evict.
    """

    def __init__(
        self,
        store_path: str,
        llm: LLMClient | None = None,
        tau_add: float = 0.0,
        tau_merge: float = 0.85,
        tau_evict_utility: float = -1.0,
        evict_min_retrievals: int = 3,
        max_skills: int = 256,
        retrieval_mode: str = "bm25_utility",
        retrieval_seed: int | None = None,
        frozen: bool = False,
        debug: bool = False,
    ):
        self.store_path = store_path
        self.llm = llm
        self.tau_add = tau_add
        self.tau_merge = tau_merge
        self.tau_evict_utility = tau_evict_utility
        self.evict_min_retrievals = evict_min_retrievals
        self.max_skills = max_skills
        if retrieval_mode not in {
            "bm25_utility",
            "bm25_constant",
            "random",
            "none",
            "dense_embedding",
            "dense_lsa",
        }:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode
        self._retrieval_rng = random.Random(retrieval_seed)
        self.frozen = frozen
        self.debug = debug

        self._skills: dict[str, Skill] = {}
        self._archive: dict[str, Skill] = {}
        self._bm25: BM25Okapi | None = None
        self._index_ids: list[str] = []
        self._retrieval_trace: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        # Dense embedding index (populated lazily when mode == "dense_embedding")
        self._embeddings: np.ndarray | None = None  # (n_skills, dim)
        self._embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
        self._lsa_vocabulary: dict[str, int] = {}
        self._lsa_idf: np.ndarray | None = None
        self._lsa_components: np.ndarray | None = None

        ensure_dir(os.path.dirname(self.store_path) or ".")
        self._load()

    # ----------------------------- IO ------------------------------
    def _load(self) -> None:
        data = load_json(self.store_path, default=None)
        if not data:
            return
        for d in data.get("skills", []):
            sk = Skill.from_dict(d)
            (self._archive if sk.archived else self._skills)[sk.id] = sk
        self._rebuild_index()
        if self.debug:
            print(
                f"[SkillMemory] loaded {len(self._skills)} active, "
                f"{len(self._archive)} archived"
            )

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "saved_at": utc_ts(),
                "skills": [sk.to_dict() for sk in self._skills.values()]
                + [sk.to_dict() for sk in self._archive.values()],
            }
            dump_json(self.store_path, payload)

    # ---------------------------- Index ----------------------------
    def _rebuild_index(self) -> None:
        self._index_ids = list(self._skills.keys())
        if not self._index_ids:
            self._bm25 = None
            self._embeddings = None
            self._lsa_vocabulary = {}
            self._lsa_idf = None
            self._lsa_components = None
            return
        corpus = [_tokenize(self._skills[i].doc_text()) for i in self._index_ids]
        self._bm25 = BM25Okapi(corpus)
        if self.retrieval_mode == "dense_embedding":
            self._rebuild_embedding_index()
        elif self.retrieval_mode == "dense_lsa":
            self._rebuild_lsa_index(corpus)

    # --------------------- Dense embedding -------------------------
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI-compatible embedding endpoint for a batch of texts."""
        if not self.llm:
            return []
        try:
            client = self.llm._client(vision=False)
            resp = client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            if self.debug:
                print(f"[SkillMemory] embedding call failed: {e}")
            return []

    def _rebuild_embedding_index(self) -> None:
        """Build a dense embedding matrix for all active skills."""
        if not self._index_ids:
            self._embeddings = None
            return
        texts = [self._skills[i].doc_text() for i in self._index_ids]
        embeddings = self._embed_texts(texts)
        if len(embeddings) == len(self._index_ids):
            self._embeddings = np.array(embeddings, dtype=np.float32)
        else:
            if self.debug:
                print(
                    f"[SkillMemory] embedding mismatch: "
                    f"{len(embeddings)}/{len(self._index_ids)}"
                )
            self._embeddings = None

    def _rebuild_lsa_index(self, corpus: list[list[str]]) -> None:
        """Build a deterministic dense latent-semantic index.

        The representation is TF-IDF projected into the right-singular-vector
        basis of the current skill bank.  This is intentionally local and
        dependency-free so the dense-retrieval ablation cannot change behavior
        because an embedding endpoint is unavailable.
        """
        terms = sorted({token for document in corpus for token in document})
        self._lsa_vocabulary = {term: index for index, term in enumerate(terms)}
        if not terms:
            self._lsa_idf = None
            self._lsa_components = None
            self._embeddings = None
            return
        matrix = np.zeros((len(corpus), len(terms)), dtype=np.float64)
        document_frequency = np.zeros(len(terms), dtype=np.float64)
        for row_index, document in enumerate(corpus):
            counts: dict[int, int] = {}
            for token in document:
                column = self._lsa_vocabulary[token]
                counts[column] = counts.get(column, 0) + 1
            for column, count in counts.items():
                matrix[row_index, column] = 1.0 + math.log(float(count))
                document_frequency[column] += 1.0
        idf = np.log((1.0 + len(corpus)) / (1.0 + document_frequency)) + 1.0
        matrix *= idf
        # At most 64 dimensions; keeping all available singular directions for
        # tiny banks avoids an arbitrary variance threshold.
        dimensions = min(64, matrix.shape[0], matrix.shape[1])
        if dimensions < 1 or not np.any(matrix):
            self._lsa_idf = idf
            self._lsa_components = None
            self._embeddings = None
            return
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        components = vt[:dimensions].T
        embeddings = matrix @ components
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._lsa_idf = idf
        self._lsa_components = components
        self._embeddings = (embeddings / norms).astype(np.float32)

    def _lsa_query(self, tokens: list[str]) -> np.ndarray | None:
        if (
            self._lsa_idf is None
            or self._lsa_components is None
            or self._embeddings is None
        ):
            return None
        counts: dict[int, int] = {}
        for token in tokens:
            column = self._lsa_vocabulary.get(token)
            if column is not None:
                counts[column] = counts.get(column, 0) + 1
        if not counts:
            return None
        vector = np.zeros(len(self._lsa_vocabulary), dtype=np.float64)
        for column, count in counts.items():
            vector[column] = (1.0 + math.log(float(count))) * self._lsa_idf[column]
        projected = vector @ self._lsa_components
        norm = float(np.linalg.norm(projected))
        if norm == 0.0:
            return None
        return (projected / norm).astype(np.float32)

    # ----------------------------- API -----------------------------
    def __len__(self) -> int:
        return len(self._skills)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def top_by_utility(self, k: int = 3) -> list[Skill]:
        """Highest-utility active skills (no query needed); for informed init."""
        with self._lock:
            if self.retrieval_mode == "none":
                return []
            if self.retrieval_mode == "random":
                candidates = list(self._skills.values())
                return self._retrieval_rng.sample(candidates, min(k, len(candidates)))
            if self.retrieval_mode == "bm25_constant":
                return list(self._skills.values())[:k]
            return sorted(self._skills.values(), key=lambda s: -s.utility)[:k]

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        with self._lock:
            if self.retrieval_mode == "none":
                self._retrieval_trace.append(
                    {"query": query, "mode": self.retrieval_mode, "picked": []}
                )
                return []
            if not self._bm25 or not self._index_ids:
                self._retrieval_trace.append(
                    {"query": query, "mode": self.retrieval_mode, "picked": []}
                )
                return []
            tokens = _tokenize(query)
            if not tokens:
                self._retrieval_trace.append(
                    {"query": query, "mode": self.retrieval_mode, "picked": []}
                )
                return []
            picked_scores: list[tuple[Skill, float | None]] = []
            if self.retrieval_mode == "random":
                ids = self._retrieval_rng.sample(
                    self._index_ids, min(k, len(self._index_ids))
                )
                picked_scores = [(self._skills[i], None) for i in ids]
            elif self.retrieval_mode == "dense_lsa":
                query_vector = self._lsa_query(tokens)
                if query_vector is not None and self._embeddings is not None:
                    similarities = self._embeddings @ query_vector
                    ranking = np.argsort(-similarities, kind="stable")[:k]
                    picked_scores = [
                        (
                            self._skills[self._index_ids[int(index)]],
                            float(similarities[int(index)]),
                        )
                        for index in ranking
                    ]
            elif self.retrieval_mode == "dense_embedding":
                if self._embeddings is not None:
                    q_emb = self._embed_texts([query])
                    if q_emb:
                        q_vec = np.array(q_emb[0], dtype=np.float32)
                        sims = [
                            (_cosine_sim(q_vec, row), sid)
                            for row, sid in zip(self._embeddings, self._index_ids)
                        ]
                        sims.sort(reverse=True)
                        for sim, sid in sims[:k]:
                            picked_scores.append((self._skills[sid], float(sim)))
                    else:
                        scores = self._bm25.get_scores(tokens)
                        ranked = sorted(
                            zip(self._index_ids, scores), key=lambda x: x[1], reverse=True
                        )
                        for i, score in ranked[:k]:
                            picked_scores.append((self._skills[i], float(score)))
                else:
                    scores = self._bm25.get_scores(tokens)
                    ranked = sorted(
                        zip(self._index_ids, scores), key=lambda x: x[1], reverse=True
                    )
                    for i, score in ranked[:k]:
                        picked_scores.append((self._skills[i], float(score)))
            else:
                scores = self._bm25.get_scores(tokens)
                ranked = sorted(
                    zip(self._index_ids, scores), key=lambda x: x[1], reverse=True
                )
                token_set = set(tokens)
                for i, score in ranked:
                    sk = self._skills[i]
                    if score > 0 or token_set & set(_tokenize(sk.doc_text())):
                        picked_scores.append((sk, float(score)))
                    if len(picked_scores) >= k:
                        break
            picked = [sk for sk, _ in picked_scores]
            now = utc_ts()
            for sk in picked:
                sk.n_retrieved += 1
                sk.last_used = now
            self._retrieval_trace.append(
                {
                    "query": query,
                    "mode": self.retrieval_mode,
                    "picked": [
                        {"id": sk.id, "score": score, "utility": sk.utility}
                        for sk, score in picked_scores
                    ],
                }
            )
            return picked

    def retrieval_trace(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._retrieval_trace)

    # ---- Self-evolving operators ---------------------------------
    def add_from_elite(
        self,
        task_description: str,
        elite_code: str,
        elite_concept: str,
        delta_fitness: float,
        diagnosis: dict | None = None,
        parent_iid: str = "",
        target_hook: str = "",
    ) -> Skill | None:
        """Maybe abstract this elite into a new skill via LLM.

        Returns the new Skill if accepted, else None.
        """
        if self.frozen:
            return None
        if delta_fitness < self.tau_add or self.llm is None:
            return None
        prompt = prompts.abstract_skill_prompt(
            task_description=task_description,
            code=elite_code,
            concept=elite_concept,
            diagnosis=diagnosis,
        )
        resp = self.llm.chat(
            user=prompt,
            system=prompts.ABSTRACT_SKILL_SYSTEM,
            temperature=0.3,
            extra_body={"enable_thinking": False},
            usage_role="skill_abstract",
        )
        obj = safe_json_loads(resp) or {}
        snippet = (obj.get("snippet") or "").strip()
        trigger = (obj.get("trigger") or "").strip()
        if not snippet or not trigger:
            return None

        sk = Skill(
            id=self._fresh_id(obj.get("id_hint", "skill")),
            trigger=trigger,
            snippet=snippet,
            tags=list(obj.get("tags", []) or []),
            target_hook=target_hook,
            utility=1.0,
            born_at=utc_ts(),
            parent_iid=parent_iid,
        )
        with self._lock:
            self._skills[sk.id] = sk
            self._maybe_merge_into(sk)
            self._enforce_capacity()
            self._rebuild_index()
        if self.debug:
            print(f"[SkillMemory] +add {sk.id}: {sk.trigger[:60]!r}")
        return sk

    def update_after_eval(
        self,
        skill_ids_used: Iterable[str],
        helped: bool,
        magnitude: float = 1.0,
    ) -> None:
        """Promote / demote skills retrieved during an Actor call."""
        if self.retrieval_mode in {"bm25_constant", "none"}:
            return
        with self._lock:
            for sid in skill_ids_used:
                sk = self._skills.get(sid) or self._archive.get(sid)
                if not sk:
                    continue
                if helped:
                    sk.utility += magnitude
                    sk.n_helped += 1
                else:
                    sk.utility -= 0.5 * magnitude
                    sk.n_hurt += 1
            self._garbage_collect()

    # ---- Internal helpers ----------------------------------------
    def _fresh_id(self, hint: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", hint.lower()).strip("-")[:32] or "skill"
        return f"{slug}-{short_hash(slug + uuid.uuid4().hex, 6)}"

    def _maybe_merge_into(self, new_sk: Skill) -> None:
        """If `new_sk` overlaps strongly with an existing skill, merge."""
        if self.llm is None or len(self._skills) < 2:
            return
        new_doc = _tokenize(new_sk.doc_text())
        if not new_doc:
            return
        new_set = set(new_doc)

        best_id, best_sim = "", 0.0
        for sid, sk in self._skills.items():
            if sid == new_sk.id:
                continue
            other_set = set(_tokenize(sk.doc_text()))
            if not other_set:
                continue
            inter = len(new_set & other_set)
            union = len(new_set | other_set) or 1
            sim = inter / union  # Jaccard, cheap proxy for cosine
            if sim > best_sim:
                best_sim, best_id = sim, sid

        if best_sim < self.tau_merge or not best_id:
            return
        if new_sk.target_hook and self._skills[best_id].target_hook:
            if new_sk.target_hook != self._skills[best_id].target_hook:
                return  # different hooks → keep both

        other = self._skills[best_id]
        try:
            resp = self.llm.chat(
                user=prompts.merge_skill_prompt(
                    skill_a=other.to_dict(), skill_b=new_sk.to_dict()
                ),
                system=prompts.MERGE_SKILL_SYSTEM,
                temperature=0.2,
                extra_body={"enable_thinking": False},
                usage_role="skill_merge",
            )
            obj = safe_json_loads(resp) or {}
            snippet = (obj.get("snippet") or "").strip()
            trigger = (obj.get("trigger") or "").strip()
            if not snippet or not trigger:
                return
            merged = Skill(
                id=self._fresh_id(obj.get("id_hint", "merged")),
                trigger=trigger,
                snippet=snippet,
                tags=sorted(set((obj.get("tags") or [])) | set(other.tags) | set(new_sk.tags)),
                target_hook=other.target_hook or new_sk.target_hook,
                utility=other.utility + new_sk.utility,
                n_retrieved=other.n_retrieved + new_sk.n_retrieved,
                n_helped=other.n_helped + new_sk.n_helped,
                n_hurt=other.n_hurt + new_sk.n_hurt,
                born_at=utc_ts(),
                parent_iid=f"merge({other.id},{new_sk.id})",
            )
            other.archived = True
            new_sk.archived = True
            self._archive[other.id] = self._skills.pop(other.id)
            self._archive[new_sk.id] = self._skills.pop(new_sk.id)
            self._skills[merged.id] = merged
            if self.debug:
                print(
                    f"[SkillMemory] merged {other.id}+{new_sk.id} "
                    f"-> {merged.id} (sim={best_sim:.2f})"
                )
        except Exception as e:
            if self.debug:
                print(f"[SkillMemory] merge failed: {e}")

    def _garbage_collect(self) -> None:
        to_archive: list[str] = []
        for sid, sk in self._skills.items():
            if (
                sk.n_retrieved >= self.evict_min_retrievals
                and sk.utility < self.tau_evict_utility
            ):
                to_archive.append(sid)
        for sid in to_archive:
            sk = self._skills.pop(sid)
            sk.archived = True
            self._archive[sid] = sk
            if self.debug:
                print(f"[SkillMemory] -evict {sid}")
        if to_archive:
            self._rebuild_index()

    def _enforce_capacity(self) -> None:
        if len(self._skills) <= self.max_skills:
            return
        # Drop lowest-utility skills first.
        sorted_items = sorted(self._skills.items(), key=lambda kv: kv[1].utility)
        n_drop = len(self._skills) - self.max_skills
        for sid, sk in sorted_items[:n_drop]:
            sk.archived = True
            self._archive[sid] = self._skills.pop(sid)
        if self.debug:
            print(f"[SkillMemory] capacity enforced, dropped {n_drop}")

    # ---- Diagnostics ---------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            utilities = [sk.utility for sk in self._skills.values()]
            return {
                "retrieval_mode": self.retrieval_mode,
                "frozen": self.frozen,
                "n_active": len(self._skills),
                "n_archived": len(self._archive),
                "mean_utility": float(sum(utilities) / len(utilities))
                if utilities
                else 0.0,
                "max_utility": float(max(utilities)) if utilities else 0.0,
                "min_utility": float(min(utilities)) if utilities else 0.0,
            }
