# localforge — Implementation Plan

> Technical blueprint derived from SPECIFICATION.md. Cross-references use SPEC §X.

## 1. Tech Stack

### 1.1 Stack Summary

| Layer | Technology | Version (uv-resolved min) | Rationale |
|---|---|---|---|
| Language | Python | 3.11+ | Pattern-matching, `tomllib`, mature typing; required by modern `transformers`/`peft`. |
| Env & deps | uv | latest | Lecture mandates `uv` (SPEC §11.1); fast, lockfile-reproducible for graders. |
| CLI | Typer | ≥0.12 | Type-hint-driven commands, auto `--help`, minimal boilerplate; pairs with Rich. |
| Console/TUI | Rich + Textual | ≥13 / ≥0.60 | Rich for tables/plain output; Textual for the live Paging Visualizer (SPEC §3.6). |
| Core ML | PyTorch | ≥2.2 | Backbone for `transformers`, AirLLM, PEFT; CPU + CUDA builds. |
| Models | transformers | ≥4.44 | HF backend, tokenizers, generation API (SPEC §3.2). |
| HF Hub | huggingface_hub | ≥0.24 | `snapshot_download`, auth via token (SPEC §3.1). |
| PEFT | peft | ≥0.12 | LoRA/QLoRA/OLoRA adapters (SPEC §3.5). |
| Quantization | bitsandbytes | ≥0.43 | NF4 4-bit (CUDA-only path, SPEC §3.2.2). |
| Accel | accelerate | ≥0.33 | Device placement, dtype, offload helpers. |
| Layer streaming | airllm | ≥2.11 | Disk-as-virtual-memory inference (SPEC §3.6); optional/soft dependency. |
| Ollama client | ollama (python) | ≥0.3 | OpenAI-compatible local API at `:11434` (SPEC §3.2.1). |
| System probe | psutil | ≥6.0 | Process RSS sampling for profiler (SPEC §3.3). |
| Config | pydantic + pydantic-settings | ≥2.7 | Typed `RunSpec`/settings, `.env` loading, validation. |
| Data/plots | pandas + matplotlib | ≥2.2 / ≥3.9 | Comparison Matrix assembly + static PNG charts (SPEC §3.4). |
| Templating | Jinja2 | ≥3.1 | `report.html` rendering. |
| Testing | pytest + pytest-cov | ≥8.0 | Unit/integration; coverage gate. |
| Lint/format | ruff | ≥0.6 | Single fast tool for lint + format (matches prior HW toolchain). |
| Types | mypy | ≥1.11 | Static typing on `core`/`backends`/`profiling`. |
| CI | GitHub Actions | — | Lint → type → test → CPU smoke run; matches submission (shared GitHub repo). |

### 1.2 Key Technical Decisions

#### Decision: Backend abstraction via Protocol + Factory
- **Context**: SPEC §3.2 requires three interchangeable engines selectable by name.
- **Options**: (1) `if/elif` dispatch — simple but unextensible; (2) ABC base class — works but heavier; (3) `typing.Protocol` + registry factory — structural typing, zero inheritance coupling, trivial to add a 4th backend (SPEC §12).
- **Choice**: Protocol + registry factory.
- **Rationale**: Backends live in separate optional dependency islands (Ollama daemon, AirLLM); Protocol lets each be imported lazily and registered without a shared base import, so a missing optional dep can't break the others.
- **Consequences**: Each backend self-registers; the factory must handle `ImportError`/availability and yield an "unavailable" marker (SPEC §3.2.1 edge case).

#### Decision: Soft/optional dependencies for ollama, airllm, bitsandbytes
- **Context**: SPEC §11.1/§11.3 — must run on a CPU-only box with neither Ollama nor AirLLM installed.
- **Options**: (1) hard deps — simplest, but breaks graders without GPU/daemon; (2) optional extras (`[gpu]`, `[airllm]`, `[ollama]`) + runtime capability probes.
- **Choice**: optional extras + `capabilities.py` probe.
- **Rationale**: Core install stays light and always works; heavy/finicky deps are opt-in; suite reports "skipped (reason)" rows instead of crashing (SPEC §3.4.1).
- **Consequences**: Need a central capability detector and disciplined lazy imports inside backend modules.

#### Decision: Profiling as a context manager wrapping every backend call
- **Context**: SPEC §3.3 — identical measurement for fair comparison.
- **Options**: (1) decorators per backend — duplicated; (2) one `Profiler` context manager owned by the runner.
- **Choice**: single `Profiler` context manager in the runner.
- **Rationale**: Measurement lives in one place, independent of backend internals; guarantees prefill/decode/peak-RAM/peak-VRAM captured uniformly.
- **Consequences**: Backends must expose prefill vs decode boundaries via a small callback/hook the profiler reads.

#### Decision: Paging instrumentation via an Observer event stream
- **Context**: SPEC §3.6 — surface AirLLM layer streaming as paging events for live TUI + static export + replay.
- **Options**: (1) print/log scraping — brittle; (2) monkeypatch AirLLM internals to emit typed `PagingEvent`s to subscribed sinks.
- **Choice**: Observer pattern — a `PagingTracer` publishes `PagingEvent`s; sinks are the Textual TUI, a JSONL writer, and (for tests) an in-memory list.
- **Rationale**: Decouples capture from rendering; replay mode just feeds recorded events to the same sinks, so the visualizer is demonstrable without AirLLM (SPEC §3.6 edge case).
- **Consequences**: A thin, version-guarded adapter around AirLLM's layer loop; if hooks are unavailable, fall back to coarse per-layer timing.

#### Decision: Filesystem as system of record (no DB)
- **Context**: SPEC §5.3 — immutable RunResults/paging streams keyed by spec hash.
- **Options**: (1) SQLite; (2) plain JSON/JSONL under `results/`.
- **Choice**: JSON/JSONL files.
- **Rationale**: Graders inspect artifacts directly; diffs are reviewable in git; no schema migrations; reports regenerate from files.
- **Consequences**: A small `ResultStore` repository encapsulates read/write + spec-hash keying.

#### Decision: OLoRA fallback wrapper
- **Context**: SPEC §11.4 — PEFT OLoRA support varies by version.
- **Choice**: Try PEFT-native OLoRA init; if absent, apply orthonormal (QR) initialization to LoRA A/B as a documented thin wrapper.
- **Rationale**: Guarantees the OLoRA path is demonstrable regardless of installed PEFT version, matching the lecture's QR-init explanation.

### 1.3 Dependency Inventory & Philosophy

**Philosophy — curated, layered:** a light always-installable core; heavy/fragile engines behind extras (`localforge[gpu]`, `[airllm]`, `[ollama]`, `[viz]`, `[dev]`). Fewer than ~20 direct core deps. Every dep below is an ecosystem standard for its job (rationale in §1.1).

## 2. Design Patterns

### 2.1 Strategy + Registry Factory (backends)
**Why:** SPEC §3.2 interchangeable engines.
```python
# backends/base.py
from typing import Protocol, runtime_checkable
from localforge.core.types import RunSpec, GenerationStream

@runtime_checkable
class InferenceBackend(Protocol):
    name: str
    def is_available(self) -> tuple[bool, str]: ...      # (ok, reason)
    def load(self, spec: RunSpec) -> None: ...
    def generate(self, spec: RunSpec) -> GenerationStream: ...  # yields tokens + prefill marker

_REGISTRY: dict[str, type] = {}
def register(cls): _REGISTRY[cls.name] = cls; return cls
def make_backend(name: str) -> InferenceBackend:
    if name not in _REGISTRY: raise UnknownBackend(name)
    return _REGISTRY[name]()
```

### 2.2 Context Manager (profiler)
**Why:** SPEC §3.3 uniform measurement.
```python
# profiling/profiler.py
class Profiler:
    def __enter__(self): self._t0=perf_counter(); self._peak_ram=sample_rss(); return self
    def mark_prefill_done(self): self.prefill_ms=(perf_counter()-self._t0)*1e3
    def __exit__(self, *exc):
        self.decode_tok_s = self._tokens / max(self._decode_seconds, 1e-9)
        self.peak_ram_mb = max_rss_since(self._t0); self.peak_vram_mb = cuda_peak_mb()
```

### 2.3 Observer (paging events)
**Why:** SPEC §3.6 decoupled capture/render/replay.
```python
# paging/tracer.py
class PagingTracer:
    def __init__(self): self._sinks: list[Callable[[PagingEvent], None]] = []
    def subscribe(self, sink): self._sinks.append(sink)
    def emit(self, ev: PagingEvent):
        for s in self._sinks: s(ev)
```

### 2.4 Repository (result store)
**Why:** SPEC §5.3 filesystem-as-record.
```python
# reporting/store.py
class ResultStore:
    def save(self, r: RunResult) -> Path: ...        # results/runs/<spec_hash>.json
    def load_suite(self, suite_id: str) -> list[RunResult]: ...
```

### 2.5 Builder (suite → RunSpecs)
**Why:** SPEC §3.4 matrix expansion of (model × backend × dtype) from one YAML.
```python
# config/suite.py
def expand_suite(doc: dict) -> list[RunSpec]:  # cartesian product → typed specs
    ...
```

### 2.6 Null Object (unavailable backend)
**Why:** SPEC §3.2.1 — skip, never crash.
```python
class UnavailableBackend:
    def __init__(self, name, reason): self.name, self.reason = name, reason
    def generate(self, spec): raise BackendUnavailable(self.name, self.reason)
```

## 3. Project Structure

### 3.1 Directory Layout
```
localforge/
├── pyproject.toml               # uv project, deps, extras, ruff/mypy/pytest config
├── uv.lock                      # pinned, reproducible env
├── README.md                    # quickstart, results showcase
├── .env.example                 # HF_TOKEN, OLLAMA_BASE_URL, cache dirs
├── .gitignore / .gitattributes
├── .github/workflows/ci.yml     # lint → type → test → CPU smoke
├── docs/                        # SPEC/IMPL/TASKS/BRANDING/PROMPT + PRD/PLAN/TODO
├── config/
│   ├── settings.toml            # defaults
│   └── suites/
│       ├── demo.yaml            # CPU-only smoke matrix
│       └── full.yaml            # scale-up matrix
├── data/
│   └── finetune/tiny_sft.jsonl  # tiny instruction dataset for PEFT demo
├── src/localforge/
│   ├── __init__.py
│   ├── __main__.py              # `python -m localforge`
│   ├── core/
│   │   ├── types.py             # RunSpec, RunResult, PagingEvent, enums
│   │   ├── errors.py            # exception hierarchy
│   │   ├── hashing.py           # stable spec_hash
│   │   ├── logging.py           # structured logging setup
│   │   └── capabilities.py      # CUDA/Ollama/AirLLM/bitsandbytes probes
│   ├── config/
│   │   ├── settings.py          # pydantic-settings, .env
│   │   └── suite.py             # YAML → RunSpec builder
│   ├── models/
│   │   ├── acquire.py           # HF snapshot_download + token
│   │   ├── registry.py          # local model registry (json)
│   │   └── formats.py           # SafeTensors/GGUF detection + sizing
│   ├── backends/
│   │   ├── base.py              # Protocol, registry, factory, Null object
│   │   ├── transformers_backend.py
│   │   ├── ollama_backend.py
│   │   ├── airllm_backend.py
│   │   └── runner.py            # orchestrates load+generate under Profiler
│   ├── profiling/
│   │   ├── profiler.py          # context manager
│   │   ├── probes.py            # psutil RSS, CUDA mem
│   │   └── baseline.py          # idle RAM/VRAM
│   ├── finetune/
│   │   ├── adapters.py          # LoRA/QLoRA/OLoRA config factory
│   │   ├── olora.py             # QR orthonormal init fallback
│   │   ├── dataset.py           # jsonl SFT loader
│   │   └── trainer.py           # PEFT training loop + param summary
│   ├── paging/
│   │   ├── events.py            # PagingEvent + sinks (jsonl, memory)
│   │   ├── tracer.py            # Observer
│   │   ├── airllm_hook.py       # version-guarded AirLLM instrumentation
│   │   └── replay.py            # recorded stream → sinks
│   ├── viz/
│   │   ├── tui.py               # Textual app: residency + fault count + timeline
│   │   ├── widgets.py           # hierarchy bar, fault gauge, timeline widgets
│   │   ├── static.py            # matplotlib PNG timeline
│   │   └── html.py              # standalone paging HTML
│   ├── reporting/
│   │   ├── store.py             # ResultStore repository
│   │   ├── matrix.py            # Comparison Matrix (pandas)
│   │   ├── charts.py            # PNG charts (mem/throughput/load)
│   │   └── report.py            # report.html via Jinja2
│   ├── cli/
│   │   ├── app.py               # Typer app wiring
│   │   ├── pull.py  run.py  compare.py  finetune.py  visualize.py  baseline.py
│   └── templates/
│       └── report.html.j2
└── tests/
    ├── conftest.py              # fixtures, fake backend, tmp result dir
    ├── unit/                    # types, hashing, suite expansion, profiler math,
    │                            #   olora init, capabilities, registry, matrix
    ├── integration/             # fake-backend run, compare suite, paging replay,
    │                            #   report generation, finetune tiny CPU
    └── data/                    # recorded paging stream, sample results
```

**Structural philosophy:** layer-based modules with a `src/` package (import-isolation, prevents accidental CWD imports). Each SPEC §4.1 component = one package. Optional engines are isolated so their imports happen only inside their backend module. Tests separate from `src`, split unit/integration. Config and datasets are data, not code.

### 3.2 Module Breakdown (selected)
- **core** — Path `src/localforge/core`. Responsibility: shared types/errors/hashing/capabilities. Exports `RunSpec`, `RunResult`, `PagingEvent`, enums, `probe_capabilities()`. Imports: stdlib, pydantic, torch (for cuda probe, lazily).
- **backends** — Responsibility: the three engines + factory + runner. Exports `make_backend`, `run_spec()`. Imports: core, profiling, paging (airllm only), optional engine libs (lazy).
- **profiling** — Responsibility: measurement. Exports `Profiler`, `baseline()`. Imports: psutil, torch (lazy).
- **finetune** — Responsibility: PEFT pipeline. Exports `train_adapter()`, `build_peft_config()`. Imports: peft, transformers, bitsandbytes (lazy for qlora).
- **paging** — Responsibility: event capture/replay. Exports `PagingTracer`, `PagingEvent`, `replay()`. Imports: core; airllm only inside `airllm_hook`.
- **viz** — Responsibility: TUI + static renders. Exports `run_tui()`, `render_static()`. Imports: textual, rich, matplotlib.
- **reporting** — Responsibility: results persistence + reports. Exports `ResultStore`, `build_matrix()`, `write_report()`. Imports: pandas, matplotlib, jinja2.
- **cli** — Responsibility: command surface. Exports `app`. Imports: typer, all of the above (lazily per command).

### 3.3 Module Dependency Graph
```
cli → {models, backends, finetune, reporting, viz, profiling}
backends → {core, profiling, paging(airllm)}
finetune → {core, models}
paging  → {core}        viz → {core, paging, reporting}
reporting → {core}      everything → core
```
No cycles; `core` is the sink shared by all. Optional-engine coupling is confined to single files.

## 4. Data Layer
No database (SPEC §5.3). On-disk layout:
```
results/
├── baseline/<host>.json
├── runs/<spec_hash>.json                # one RunResult
├── suites/<suite_id>.json               # list of spec_hashes + metadata (git SHA, host)
├── paging/<spec_hash>.jsonl             # PagingEvent stream
└── reports/<suite_id>/{report.html,*.png,matrix.md,matrix.json}
```
`spec_hash` = sha256 of canonical-JSON RunSpec (stable key, dedup, cache). `ResultStore` is the only writer/reader.

## 7. Error Handling Strategy

### 7.1 Error Classification
| Category | Example | Surfaced As | User Sees |
|---|---|---|---|
| Config | bad suite YAML | `ConfigError` | line-pointed message, exit 2 |
| Auth | missing HF_TOKEN | `AuthError` | "set HF_TOKEN in .env" |
| Availability | Ollama down / AirLLM absent | `BackendUnavailable` | skipped row + reason (non-fatal) |
| Capability | NF4 on CPU | warning | "falling back to fp32" |
| Resource | OOM during layer stream | `ResourceError` | suggest smaller model / lower ceiling |
| Internal | unexpected | `LocalforgeError` | message + log path, exit 1 |

### 7.2 Propagation
Backend/engine exceptions are caught at the `runner` boundary; availability/capability issues become typed non-fatal results (`backend_available=False`, `note=...`) so a `compare` suite always completes. Only genuinely fatal config/auth errors abort the command.

## 8. Configuration

### 8.1 Sources (low→high precedence)
`settings.toml` defaults → `.env` → env vars → CLI flags.

### 8.2 Schema (selected)
| Key | Type | Default | Env Var | Description |
|---|---|---|---|---|
| hf_token | str | — | HF_TOKEN | Hugging Face access token |
| ollama_base_url | str | http://localhost:11434/v1 | OLLAMA_BASE_URL | Ollama OpenAI-compat endpoint |
| cache_dir | path | ~/.cache/localforge | LOCALFORGE_CACHE | model/result cache root |
| airllm_ram_ceiling_mb | int | 4096 | — | layer-stream working-set target |
| default_model | str | Qwen2.5-0.5B-Instruct | — | CI/demo model |
| seed | int | 0 | — | global RNG seed |

## 9. Testing Strategy

### 9.1 Test Pyramid
| Level | Tool | Scope | Target |
|---|---|---|---|
| Unit | pytest | types, hashing, suite expansion, profiler math, OLoRA QR init, capabilities, matrix build | ≥80% on `core`/`config`/`reporting` |
| Integration | pytest | fake-backend run, compare suite end-to-end, paging replay→render, report.html generation, tiny CPU LoRA finetune | all CLI commands, happy + skip paths |

### 9.2 Patterns
A `FakeBackend` (registered like real ones) yields deterministic tokens + synthetic prefill marker, so the full run/compare/report pipeline is testable with **zero model downloads and no GPU**. A recorded `paging.jsonl` fixture drives visualizer/replay tests. `tmp_path` isolates `results/`. CUDA/Ollama/AirLLM tests are `@pytest.mark.skipif(not capability)`.

### 9.3 CI Pipeline
```
Push/PR → ruff check + format --check → mypy → pytest (unit+integration, fake backend) → CPU smoke: `localforge compare --suite config/suites/demo.yaml --backend transformers`
```

## 10. Security / Secrets
`.env` only; `.env` git-ignored, `.env.example` committed. HF token never logged (redacted in structured logs). No network services exposed by localforge itself.

## 12. Development Workflow

### 12.1 Local Setup
```bash
git clone <repo> && cd localforge
uv sync                      # core env from uv.lock
uv sync --extra dev          # + test/lint tools
cp .env.example .env         # add HF_TOKEN
uv run localforge baseline
uv run localforge compare --suite config/suites/demo.yaml
```
Optional engines: `uv sync --extra ollama --extra airllm --extra gpu`.

### 12.2 Code Standards
ruff (lint+format), mypy on core packages, Google-style docstrings, conventional commits. Pre-commit hook runs ruff + mypy.

### 12.3 Git Workflow
`main` protected; feature branches `feat/<module>`; PRs run CI. Instructor (`rmisegal@gmail.com`) added as collaborator per submission rules.
