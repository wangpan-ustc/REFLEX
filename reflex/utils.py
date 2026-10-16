"""Generic utilities for REFLEX."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import time
from typing import Any


def utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def safe_json_loads(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response.

    Strategy:
        1. Try the entire string.
        2. Try the largest ```json ...``` fenced block.
        3. Try the first balanced { ... } region.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = -1
    return None


def extract_python_block(text: str) -> str | None:
    """Pull out the first ```python ...``` fenced code block."""
    if not text:
        return None
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models finish a complete function at the generation boundary but
    # omit only the closing Markdown fence. Preserve that code and let the
    # downstream AST/import checks reject genuinely truncated Python.
    m = re.search(r"```(?:python)?\s*(.*)\Z", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def get_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def code_complexity(code: str) -> dict[str, int]:
    """Cheap structural metrics of a code string."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"lines": code.count("\n") + 1, "branches": 0, "calls": 0}
    branches = sum(
        isinstance(n, (ast.If, ast.For, ast.While, ast.Try)) for n in ast.walk(tree)
    )
    calls = sum(isinstance(n, ast.Call) for n in ast.walk(tree))
    return {"lines": code.count("\n") + 1, "branches": branches, "calls": calls}


def b64_image_from_bytes(buf: bytes) -> str:
    return base64.b64encode(buf).decode("ascii")


def b64_image_from_pil(img) -> str:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return b64_image_from_bytes(out.getvalue())


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
