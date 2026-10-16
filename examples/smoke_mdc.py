"""One-shot smoke test of the Critic→Actor (m_diag) pipeline.

Runs a SINGLE diagnose+patch cycle on the taylor-linear seed, prints
prompts and responses, and writes the BE image to disk. ~5-15 seconds and
2 LLM calls — useful for validating the pipeline without burning a full run.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.antenna_physics import CCAAConfig
from examples.antenna_seeds import CLASSIC_SEEDS
from examples.antenna_template import ANTENNA_TASK_DESCRIPTION, ANTENNA_TEMPLATE
from examples.run_antenna import (
    REGIMES,
    cfg_to_spec,
    evaluate_code,
    render_antenna_be,
)
from reflex.critic_actor import MDCOrchestrator
from reflex.llm_client import LLMClient
from reflex.memory import SkillMemory
from reflex.operators import Operators
from reflex.population import Individual


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-host", default=os.environ.get("LLM_HOST", ""))
    p.add_argument("--llm-key", default=os.environ.get("LLM_KEY", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    p.add_argument("--vision-model", default="qwen-vl-max")
    p.add_argument("--vision-host", default=None)
    p.add_argument("--vision-key", default=os.environ.get("VISION_KEY") or None)
    p.add_argument("--seed-name", default="taylor-linear")
    p.add_argument("--out", default="runs/smoke_mdc")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = REGIMES["compact"]
    spec = cfg_to_spec(cfg)

    seed = next(s for s in CLASSIC_SEEDS if s["name"] == args.seed_name)
    score, metrics, err, x = evaluate_code(seed["code"], cfg, spec)
    assert not err, err
    print(f"[smoke] seed={seed['name']}  score={score:+.4f}  metrics={metrics['sll_db']:.2f}/{metrics['fnbw_deg']:.2f}/{metrics['directivity_db']:.2f}")

    image_b64 = render_antenna_be(x, cfg, score)
    Path(args.out, "be.png").write_bytes(base64.b64decode(image_b64))
    print(f"[smoke] BE image written: {args.out}/be.png  ({len(image_b64)*3//4} bytes)")

    llm = LLMClient(
        host=args.llm_host, key=args.llm_key, model=args.llm_model,
        vision_model=args.vision_model,
        vision_host=args.vision_host, vision_key=args.vision_key,
        extra_body={"enable_thinking": True}, debug=True,
    )
    skills = SkillMemory(os.path.join(args.out, "bank.json"), llm=llm, debug=True)
    mdc = MDCOrchestrator(llm, debug=True)

    parent = Individual(
        code=seed["code"], score=score, concept=seed["concept"],
        operator="seed", gen=0, iid="parent",
        be_summary=metrics, image_b64=image_b64,
    )

    print("\n[smoke] === Critic call ===")
    diag = mdc.diagnose(
        task_description=ANTENNA_TASK_DESCRIPTION,
        be_image_b64=image_b64,
        parent_score=score,
    )
    print(f"  failure_modes : {diag.failure_modes}")
    print(f"  root_cause    : {diag.root_cause}")
    print(f"  fix_type      : {diag.suggested_fix_type}")
    print(f"  confidence    : {diag.confidence}")

    ops = Operators(ANTENNA_TASK_DESCRIPTION, ANTENNA_TEMPLATE, mdc, skills)
    ops.best_score = score
    ops.best_concept = seed["concept"]

    print("\n[smoke] === Actor call ===")
    child = ops.mutate_diagnostic(parent)
    print(f"  concept: {child.concept}")
    print(f"  code (first 600 chars):\n    {child.code[:600]!r}")

    cscore, cmetrics, cerr, cx = evaluate_code(child.code, cfg, spec)
    print(f"\n[smoke] child score={cscore:+.4f}  err={'YES' if cerr else 'no'}")
    if cerr:
        print(f"  err: {cerr[:400]}")
    print(f"\n[smoke] LLM usage: {llm.stats()}")


if __name__ == "__main__":
    main()
