"""Filesystem result store (docs/IMPLEMENTATION.md §2.4, §4).

The filesystem under ``results/`` is the system of record. RunResults are
immutable JSON keyed by spec hash; a suite run records which results belong to
it plus reproducibility metadata (git SHA, host, timestamp).
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from localforge.core.types import RunResult


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class SuiteRun:
    suite_id: str
    git_sha: str
    host: str
    created_utc: str
    spec_hashes: list[str] = field(default_factory=list)


class ResultStore:
    def __init__(self, results_dir: Path) -> None:
        self._root = results_dir
        self._runs = results_dir / "runs"
        self._suites = results_dir / "suites"

    def save_result(self, result: RunResult) -> Path:
        self._runs.mkdir(parents=True, exist_ok=True)
        path = self._runs / f"{result.spec_hash}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_result(self, spec_hash: str) -> RunResult | None:
        path = self._runs / f"{spec_hash}.json"
        if not path.exists():
            return None
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def save_suite(self, suite_id: str, results: list[RunResult]) -> Path:
        for result in results:
            self.save_result(result)
        run = SuiteRun(
            suite_id=suite_id,
            git_sha=_git_sha(),
            host=platform.node() or "localhost",
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            spec_hashes=[r.spec_hash for r in results],
        )
        self._suites.mkdir(parents=True, exist_ok=True)
        path = self._suites / f"{suite_id}.json"
        path.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        return path

    def load_suite(self, suite_id: str) -> SuiteRun | None:
        path = self._suites / f"{suite_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SuiteRun(**data)

    def load_suite_results(self, suite_id: str) -> list[RunResult]:
        run = self.load_suite(suite_id)
        if run is None:
            return []
        results = [self.load_result(h) for h in run.spec_hashes]
        return [r for r in results if r is not None]
