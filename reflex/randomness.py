"""Seed handling for examples and scripts.

Seeds default to fresh system randomness, and the values actually used are
returned so runners can record them in their output metadata.
"""
from __future__ import annotations

import secrets
import json
from pathlib import Path


def seed_argument(value):
    """Parse ``--seed``: ``random`` (or None) defers to :func:`resolve_seed`."""
    return None if value is None or str(value).lower() == 'random' else int(value)


def resolve_seed(value=None):
    """Return an explicit seed, or draw a fresh 32-bit seed to be logged."""
    return secrets.randbits(32) if value is None else int(value)


def environment_seeds(value=None, count=3):
    """Parse explicit environment seeds, or draw ``count`` fresh ones to be logged."""
    if value is None or (isinstance(value, str) and value.lower() == 'random'):
        return tuple(secrets.randbits(32) for _ in range(count))
    values = value.split(',') if isinstance(value, str) else value
    return tuple(int(item) for item in values)


def prior_run_ids(directory):
    """Return the run identifiers of a frozen-prior directory."""
    root = Path(directory)
    manifest = root / 'manifest.json'
    if manifest.is_file():
        identifiers = json.loads(manifest.read_text()).get('run_ids')
        if identifiers:
            return [int(item) for item in identifiers]
    identifiers = sorted(int(p.stem[4:]) for p in root.glob('run_*.json')
                         if p.stem[4:].isdigit())
    if not identifiers:
        raise ValueError('No frozen priors found in ' + str(root))
    return identifiers
