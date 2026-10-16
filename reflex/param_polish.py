"""Structure/parameter separation: local hill-climbing over float constants.

The LLM proposes the *structure* of a policy; this module tunes the
*numeric constants* inside `choose_action` with a cheap multiplicative
hill climb, without any further LLM calls.

Only float literals are touched (skipping 0.0). Integer literals are
left alone — in the Acrobot task the discrete actions 0/1/2 are ints and
must never be perturbed.
"""
from __future__ import annotations

import ast
from typing import Callable

import numpy as np


def _find_choose_action(tree: ast.Module) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "choose_action":
            return node
    return None


def _collect_float_constants(fn_node: ast.AST) -> list[ast.Constant]:
    """Float literals in the function *body* (bool is int, so it's excluded)."""
    nodes: list[ast.Constant] = []
    for stmt in fn_node.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, float)
                and node.value != 0.0
            ):
                nodes.append(node)
    return nodes


def polish_constants(
    code: str,
    eval_fn: Callable[[str], float],
    iters: int = 20,
    rel_scale: float = 0.25,
    seed: int | None = None,
) -> tuple[str, float]:
    """Hill-climb the float constants of `choose_action` in `code`.

    Each round perturbs the current base values multiplicatively:
        new = v * (1 + N(0, rel_scale * 0.97**i))
    (if more than 12 floats exist, only 6 randomly chosen ones move per
    round), rebuilds the source via ast.unparse (comments are lost — the
    rebuilt code stays executable with the same signature), and adopts
    the candidate as new base iff `eval_fn` scores it strictly higher.

    Returns (best_code, best_score). `eval_fn` must return -inf on failure.
    """
    rng = np.random.default_rng(seed)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, float(eval_fn(code))

    fn_node = _find_choose_action(tree)
    const_nodes = _collect_float_constants(fn_node) if fn_node is not None else []
    base_score = float(eval_fn(code))
    if not const_nodes:
        return code, base_score

    base_vals = np.array([n.value for n in const_nodes], dtype=float)
    best_code, best_score = code, base_score
    n = len(const_nodes)

    for i in range(iters):
        sigma = rel_scale * (0.97 ** i)
        idx = rng.choice(n, size=6, replace=False) if n > 12 else np.arange(n)
        cand = base_vals.copy()
        cand[idx] = cand[idx] * (1.0 + rng.normal(0.0, sigma, size=len(idx)))
        for node, v in zip(const_nodes, cand):
            node.value = float(v)
        try:
            new_code = ast.unparse(ast.fix_missing_locations(tree))
        except Exception:
            continue
        try:
            s = float(eval_fn(new_code))
        except Exception:
            s = float("-inf")
        if np.isfinite(s) and s > best_score:
            best_code, best_score = new_code, s
            base_vals = cand

    return best_code, best_score


def polish_constants_cem(
    code: str,
    eval_fn: Callable[[str], float],
    n_per_gen: int = 7,
    n_gens: int = 3,
    elite_frac: float = 0.45,
    init_rel_sigma: float = 0.25,
    smooth: float = 0.7,
    seed: int | None = None,
    constant_selector: Callable[[ast.AST], list[ast.Constant]] | None = None,
) -> tuple[str, float]:
    """CEM (cross-entropy method) polish of the float constants in `choose_action`.

    Instead of the point-wise accept/reject of `polish_constants`, a full
    sampling distribution N(mu, sigma) is fit to the elite fraction of each
    generation, which is markedly less sensitive to evaluation noise. The
    best-so-far vector is kept in every generation's pool (elitist anchor),
    so the elite refit never drifts away from an already-found optimum.

    Budget: n_per_gen * n_gens + 1 evaluations (defaults: 22), on par with
    the 21 used by the hill-climb version. Returns (best_code, best_score);
    `eval_fn` must return -inf on failure.
    """
    rng = np.random.default_rng(seed)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, float(eval_fn(code))

    fn_node = _find_choose_action(tree)
    selector = constant_selector or _collect_float_constants
    const_nodes = selector(fn_node) if fn_node is not None else []
    base_score = float(eval_fn(code))
    if not const_nodes:
        return code, base_score

    def _rebuild(vals: np.ndarray) -> str | None:
        for node, v in zip(const_nodes, vals):
            node.value = float(v)
        try:
            return ast.unparse(ast.fix_missing_locations(tree))
        except Exception:
            return None

    mu = np.array([n.value for n in const_nodes], dtype=float)
    sigma = np.maximum(init_rel_sigma * np.abs(mu), 1e-3)
    best_code, best_score = code, base_score
    best_vec = mu.copy()

    for _ in range(n_gens):
        pool: list[tuple[float, np.ndarray]] = []
        for _ in range(n_per_gen):
            vec = rng.normal(mu, sigma)
            new_code = _rebuild(vec)
            if new_code is None:
                pool.append((float("-inf"), vec))
                continue
            try:
                s = float(eval_fn(new_code))
            except Exception:
                s = float("-inf")
            pool.append((s, vec))
            if np.isfinite(s) and s > best_score:
                best_code, best_score = new_code, s
                best_vec = vec.copy()
        # Elitist anchor joins the sort with its known score (generation 1:
        # the original mu vector with base_score). Refit mu is never
        # re-evaluated -- the budget is n_per_gen * n_gens + 1 -- so the
        # best-so-far vector plays the role of "current mu" thereafter.
        pool.append((best_score, best_vec.copy()))
        pool.sort(key=lambda t: t[0], reverse=True)
        n_elite = max(1, int(np.ceil(elite_frac * len(pool) - 1e-9)))
        elites = [vec for s, vec in pool[:n_elite] if np.isfinite(s)]
        if not elites:
            continue  # every candidate failed: keep the current distribution
        elite_arr = np.stack(elites)
        mu = smooth * elite_arr.mean(axis=0) + (1.0 - smooth) * mu
        sigma = smooth * np.maximum(elite_arr.std(axis=0), 1e-3) + (1.0 - smooth) * sigma
        sigma = np.maximum(sigma, 1e-3)

    return best_code, best_score


__all__ = ["polish_constants", "polish_constants_cem"]
