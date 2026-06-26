"""Stable hashing of a :class:`~localforge.core.types.RunSpec`.

The hash is a content address: equal specs hash equally, any field change
changes the hash. It keys results on disk (``results/runs/<spec_hash>.json``)
and deduplicates repeated work. See docs/IMPLEMENTATION.md §4.
"""

from __future__ import annotations

import hashlib
import json

from localforge.core.types import RunSpec


def spec_hash(spec: RunSpec) -> str:
    """Return a short, stable hex digest for ``spec``.

    Uses canonical JSON (sorted keys, enum values coerced to their string form)
    so the digest is independent of field declaration order and stable across
    processes and Python versions.
    """
    payload = spec.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
