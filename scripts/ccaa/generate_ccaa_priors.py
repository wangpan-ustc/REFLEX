#!/usr/bin/env python
"""Generate and freeze one first valid multimodal-LLM CCAA proposal per run_id.

The endpoint and model are supplied by the caller. The API key comes from
LLM_KEY or a hidden terminal prompt and is never persisted or printed.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.antenna_physics import evaluate_design
from examples.run_antenna import REGIMES, cfg_to_spec, _safe_exec_design
from examples.antenna_template import ANTENNA_TASK_DESCRIPTION, ANTENNA_TEMPLATE
from reflex.critic_actor import MDCOrchestrator
from reflex.llm_client import LLMClient
from reflex.memory import SkillMemory
from reflex.operators import Operators


PROTOCOL_ID = "reflex-ccaa-first-prior-strict36-random-v2"
STRICT_TASK_DESCRIPTION = ANTENNA_TASK_DESCRIPTION + """

STRICT FAIR-COMPARISON CONSTRAINTS:
- Return exactly the base 36-dimensional vector: 32 amplitude codes followed by 4 radius codes.
- Do not return rotations, phases, a dict, or a refine/local-search instruction.
- Do not use randomness, time, files, network access, or mutable global state.
- The same function call with the same spec must return the identical vector.
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_vector(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-host", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--out-root", default="runs/ccaa_llm_priors_strict36")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--max-validity-attempts", type=int, default=5)
    args = parser.parse_args()

    out_root = (ROOT / args.out_root).resolve()
    runs_root = (ROOT / "runs").resolve()
    if out_root.parent != runs_root:
        raise ValueError(f"--out-root must be a direct child of {runs_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    run_ids = list(range(args.num_runs))
    key = os.environ.get("LLM_KEY", "").strip() or getpass.getpass("API key: ").strip()
    if not key:
        raise SystemExit("missing API key")

    cfg = REGIMES["compact"]
    spec = cfg_to_spec(cfg)
    rows = []
    for run_id in run_ids:
        output = out_root / f"run_{run_id}.json"
        if output.exists():
            row = json.loads(output.read_text(encoding="utf-8"))
            if row.get("valid") and row.get("protocol_id") == PROTOCOL_ID:
                rows.append({"run_id": run_id, "status": "skip", "score": row["score"]})
                print(json.dumps(rows[-1]), flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite incompatible prior: {output}")

        llm = LLMClient(
            host=args.llm_host,
            key=key,
            model=args.llm_model,
            vision_model=args.llm_model,
            timeout=300.0,
            temperature=0.8,
            max_tokens=4096,
            max_retries=3,
            extra_headers={"X-Request-Id": uuid.uuid4().hex},
        )
        skills = SkillMemory(str(out_root / f"run_{run_id}_unused_skill_bank.json"), llm=llm)
        ops = Operators(
            STRICT_TASK_DESCRIPTION,
            ANTENNA_TEMPLATE,
            MDCOrchestrator(llm),
            skills,
            rng=random.Random(run_id),
        )
        attempts = []
        accepted = None
        for validity_attempt in range(1, args.max_validity_attempts + 1):
            proposal = ops.init_zero_shot()
            error = ""
            score = None
            metrics = None
            x = None
            hint = None
            try:
                x, hint = _safe_exec_design(proposal.code, spec)
                x_repeat, hint_repeat = _safe_exec_design(proposal.code, spec)
                if hint or hint_repeat:
                    raise ValueError("refine/local-search hint is forbidden in the frozen prior")
                if x.size != cfg.design_dim:
                    raise ValueError(f"expected strict 36D base vector, got {x.size}D")
                if x_repeat.size != x.size or not np.array_equal(x, x_repeat):
                    raise ValueError("proposal is nondeterministic across repeated execution")
                x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
                metrics = evaluate_design(x, cfg)
                score = float(metrics["score"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            valid = bool(x is not None and x.size == cfg.design_dim and score is not None and np.isfinite(score) and not error)
            attempts.append(
                {
                    "validity_attempt": validity_attempt,
                    "valid": valid,
                    "concept": proposal.concept,
                    "code": proposal.code,
                    "code_sha256": sha256_text(proposal.code),
                    "vector_dim": int(x.size) if x is not None else None,
                    "had_refine_hint": bool(hint),
                    "score": float(score) if score is not None and np.isfinite(score) else None,
                    "error": error,
                }
            )
            if valid:
                accepted = (proposal, float(score), metrics, np.asarray(x, dtype=float), validity_attempt)
                break
        if accepted is None:
            raise RuntimeError(f"run_id {run_id}: no valid proposal in {args.max_validity_attempts} attempts")

        proposal, score, metrics, x, validity_attempt = accepted
        payload = {
            "protocol_id": PROTOCOL_ID,
            "endpoint": args.llm_host,
            "model": args.llm_model,
            "run_id": run_id,
            "temperature": 0.8,
            "max_tokens": 4096,
            "valid": True,
            "accepted_validity_attempt": validity_attempt,
            "concept": proposal.concept,
            "code": proposal.code,
            "code_sha256": sha256_text(proposal.code),
            "x": [float(value) for value in x],
            "x_sha256_f64le": sha256_vector(x),
            "score": score,
            "metrics": {
                "sll_db": float(metrics["sll_db"]),
                "fnbw_deg": float(metrics["fnbw_deg"]),
                "directivity_db": float(metrics["directivity_db"]),
            },
            "attempts": attempts,
            "llm_usage": llm.stats(),
            "llm_call_trace": llm.call_trace(),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append({"run_id": run_id, "status": "ok", "score": score, "attempts": validity_attempt})
        print(json.dumps(rows[-1]), flush=True)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "endpoint": args.llm_host,
        "model": args.llm_model,
        "run_ids": run_ids,
        "temperature": 0.8,
        "max_tokens": 4096,
        "n_valid": len(rows),
        "results": rows,
        "credential_persisted": False,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
