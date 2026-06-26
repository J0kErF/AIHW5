# localforge — Tasks

> Ordered work breakdown derived from IMPLEMENTATION.md. Execute sequentially; each task is one focused session. No task references later work.

## Summary

| Metric | Value |
|---|---|
| Total Tasks | 18 |
| Phases | 6 |
| Estimated Effort | ~9–12 days |
| Foundation Complete | After Task 3 |
| MVP (run + profile + compare on CPU) | After Task 9 |
| Full Release | After Task 18 |

---

## Phase 1: Foundation
> After this phase: package installs via `uv`, imports, lints, types; nothing functional.

### Task 1: Project Scaffolding
**Create the uv project skeleton, config, CI, and directory tree.**
**Files:** `pyproject.toml` (deps + extras `[dev,gpu,ollama,airllm,viz]` + ruff/mypy/pytest config), `uv.lock` (via `uv lock`), `.gitignore`, `.gitattributes`, `.env.example`, `README.md` (stub), `.github/workflows/ci.yml`, full `src/localforge/` + `tests/` + `config/` + `data/` tree with `__init__.py`/`.gitkeep`.
**Commands:** `uv init`, `uv add` core deps, `uv add --group dev pytest pytest-cov ruff mypy`, `uv lock`, `git init`.
**Acceptance:**
- [ ] `uv sync` and `uv sync --extra dev` succeed.
- [ ] `uv run ruff check .` and `uv run mypy src` pass on empty package.
- [ ] `uv run pytest` runs (0 tests, 0 failures).
- [ ] Every directory from IMPLEMENTATION.md §3.1 exists.
**Dependencies:** none **Effort:** 2h **Refs:** IMPL §1, §3.1

### Task 2: Core Types, Errors, Hashing
**Define `RunSpec`, `RunResult`, `PagingEvent`, enums, exception hierarchy, stable spec hash.**
**Files:** `core/types.py`, `core/errors.py`, `core/hashing.py`, `core/logging.py`, `tests/unit/test_types.py`, `tests/unit/test_hashing.py`.
**Implementation:** pydantic models per SPEC §5.1; enums `Backend`, `Dtype`, `Device`, `PageAction`, `PageSource`; `spec_hash()` = sha256 of canonical JSON (key order stable). Exception tree from IMPL §7.1 (`LocalforgeError` base).
**Acceptance:**
- [ ] All entities instantiate + round-trip JSON; mypy clean, no `Any`.
- [ ] `spec_hash` identical for equal specs, differs on any field change.
- [ ] Every error category in IMPL §7.1 has a class.
**Dependencies:** T1 **Effort:** 3h **Refs:** SPEC §2,§5; IMPL §2,§7

### Task 3: Capability Probes & Settings
**Detect CUDA/Ollama/AirLLM/bitsandbytes; load typed settings + `.env`.**
**Files:** `core/capabilities.py`, `config/settings.py`, `config/settings.toml`, `tests/unit/test_capabilities.py`.
**Implementation:** `probe_capabilities()` returns dataclass of `(cuda, ollama, airllm, bitsandbytes)` with reasons, using lazy imports + socket check for `:11434`. pydantic-settings hierarchy per IMPL §8.
**Acceptance:**
- [ ] Probes never raise on a machine lacking any engine (return `(False, reason)`).
- [ ] Settings load defaults < `.env` < env var < explicit override, verified by test.
- [ ] HF token absent → settings still load; token redacted in `repr`.
**Dependencies:** T2 **Effort:** 3h **Refs:** IMPL §1.2,§8,§10

---

## Phase 2: Profiling & Backends Core
> After this phase: a fake backend runs under the profiler and produces a RunResult.

### Task 4: Profiler & Probes
**Context-manager profiler capturing load/prefill/decode/peak RAM/peak VRAM + baseline.**
**Files:** `profiling/profiler.py`, `profiling/probes.py`, `profiling/baseline.py`, `tests/unit/test_profiler.py`.
**Implementation:** Context Manager pattern (IMPL §2.2); `probes.py` wraps psutil RSS + `torch.cuda.max_memory_allocated` (guarded). `baseline()` writes `results/baseline/<host>.json`.
**Acceptance:**
- [ ] Profiler computes decode_tok_s from token count + decode seconds; prefill split via `mark_prefill_done()`.
- [ ] On CPU-only host, `peak_vram_mb is None` (no error).
- [ ] Profiler overhead asserted < 5% in a synthetic test.
**Dependencies:** T3 **Effort:** 3h **Refs:** SPEC §3.3,§10; IMPL §2.2

### Task 5: Backend Protocol, Registry, Runner, FakeBackend
**Define the `InferenceBackend` Protocol, registry/factory, Null object, runner, and a deterministic `FakeBackend`.**
**Files:** `backends/base.py`, `backends/runner.py`, `tests/conftest.py` (FakeBackend fixture), `tests/integration/test_runner.py`.
**Implementation:** Protocol + `register`/`make_backend` (IMPL §2.1), `UnavailableBackend` Null object (§2.6), `runner.run_spec(spec)` wraps backend in `Profiler`, returns `RunResult`. FakeBackend yields fixed tokens + emits a prefill marker.
**Acceptance:**
- [ ] `make_backend("fake")` runs end-to-end → valid `RunResult` with all metrics.
- [ ] Unknown backend → `UnknownBackend`; unavailable backend → `RunResult(backend_available=False, note=...)`, no crash.
- [ ] Run is deterministic for fixed seed.
**Dependencies:** T4 **Effort:** 4h **Refs:** SPEC §3.2; IMPL §2.1,§2.6

---

## Phase 3: Models & Real Backends
> After this phase: real HF/Ollama/AirLLM inference works (when available).

### Task 6: Model Acquisition, Registry, Format Detection
**Pull from Hugging Face, register locally, detect SafeTensors/GGUF + size.**
**Files:** `models/acquire.py`, `models/registry.py`, `models/formats.py`, `tests/unit/test_formats.py`, `tests/integration/test_acquire.py` (skipif no token/network).
**Implementation:** `snapshot_download` with token; registry json under cache; `formats.detect()` inspects files.
**Acceptance:**
- [ ] `pull(model_id)` caches weights + registry entry (id, format, bytes, path); re-pull is a no-op.
- [ ] Missing token → `AuthError` with actionable message.
- [ ] Format detection unit-tested on fixture file listings (no download).
**Dependencies:** T5 **Effort:** 3h **Refs:** SPEC §3.1; IMPL §3.2

### Task 7: Transformers Backend (+ quantization path)
**Implement the `transformers` backend with CPU default and optional fp16/NF4.**
**Files:** `backends/transformers_backend.py`, `tests/integration/test_transformers_backend.py` (skipif slow/no model).
**Implementation:** load model+tokenizer, greedy generate, emit prefill marker after first forward; NF4 via bitsandbytes only when CUDA, else warn + fp32 (SPEC §3.2.2).
**Acceptance:**
- [ ] Generates text + full metrics on the demo model (CPU).
- [ ] Requesting NF4 on CPU logs fallback note, still produces output.
- [ ] Seeded greedy output reproducible.
**Dependencies:** T6 **Effort:** 4h **Refs:** SPEC §3.2; IMPL §1.2

### Task 8: Ollama Backend
**Implement the Ollama backend over the OpenAI-compatible local API.**
**Files:** `backends/ollama_backend.py`, `tests/integration/test_ollama_backend.py` (skipif daemon down).
**Implementation:** client to `OLLAMA_BASE_URL`; `is_available()` pings daemon; stream tokens; map errors to `BackendUnavailable`.
**Acceptance:**
- [ ] With daemon up + model pulled, returns text + metrics.
- [ ] With daemon down, `is_available()` → `(False, reason)`; runner yields skipped result.
**Dependencies:** T7 **Effort:** 3h **Refs:** SPEC §3.2.1; IMPL §4.3

### Task 9: AirLLM Backend (layer streaming)
**Implement the AirLLM CPU backend running layer-by-layer.**
**Files:** `backends/airllm_backend.py`, `tests/integration/test_airllm_backend.py` (skipif airllm absent).
**Implementation:** wrap AirLLM layered model; respect `airllm_ram_ceiling_mb`; expose layer boundaries for paging hook (T12). Soft import.
**Acceptance:**
- [ ] On a machine with airllm + small model, completes generation without exceeding configured ceiling.
- [ ] Absent airllm → skipped result, no crash.
**Dependencies:** T8 **Effort:** 4h **Refs:** SPEC §3.6; IMPL §1.2
> **Milestone — MVP:** `run` + `compare` work on CPU with transformers/fake; others skip cleanly.

---

## Phase 4: Reporting & CLI
> After this phase: full comparison reports and a usable CLI.

### Task 10: Result Store & Suite Builder
**Filesystem repository + YAML→RunSpec matrix expansion.**
**Files:** `reporting/store.py`, `config/suite.py`, `config/suites/demo.yaml`, `config/suites/full.yaml`, `tests/unit/test_suite.py`, `tests/integration/test_store.py`.
**Implementation:** Repository (IMPL §2.4) + Builder (§2.5) cartesian product of model×backend×dtype.
**Acceptance:**
- [ ] `expand_suite(demo.yaml)` yields expected RunSpecs.
- [ ] Store saves/loads RunResults keyed by spec_hash; suite metadata records git SHA + host.
**Dependencies:** T9 **Effort:** 3h **Refs:** SPEC §3.4,§5.3; IMPL §2.4,§2.5

### Task 11: Comparison Matrix, Charts, HTML Report
**Assemble matrix, render PNG charts, bundle `report.html`.**
**Files:** `reporting/matrix.py`, `reporting/charts.py`, `reporting/report.py`, `templates/report.html.j2`, `tests/integration/test_report.py`.
**Implementation:** pandas matrix; matplotlib charts (peak mem, decode tok/s, load time); Jinja2 report with metadata + skipped rows.
**Acceptance:**
- [ ] `compare` produces `matrix.md`, `matrix.json`, ≥3 PNGs, `report.html`.
- [ ] Skipped backends appear as labeled rows, not omitted.
- [ ] Report regenerates deterministically from stored results.
**Dependencies:** T10 **Effort:** 4h **Refs:** SPEC §3.4; IMPL §4

### Task 12: CLI Wiring (pull/baseline/run/compare)
**Typer app exposing the core commands with `--no-tui` plain output.**
**Files:** `cli/app.py`, `cli/pull.py`, `cli/baseline.py`, `cli/run.py`, `cli/compare.py`, `__main__.py`, `tests/integration/test_cli.py`.
**Implementation:** Typer commands; Rich tables; lazy per-command imports so missing extras don't break `--help`.
**Acceptance:**
- [ ] `localforge --help` lists all commands with no heavy imports.
- [ ] `localforge compare --suite config/suites/demo.yaml` runs CPU-only end-to-end and writes a report.
- [ ] All commands have a plain-text path (no TUI required).
**Dependencies:** T11 **Effort:** 4h **Refs:** SPEC §6; IMPL §3

---

## Phase 5: Fine-Tuning & Paging Visualizer (Originality)
> After this phase: PEFT pipeline + the signature visualizer work.

### Task 13: PEFT Adapters & OLoRA Fallback
**LoRA/QLoRA/OLoRA config factory + QR orthonormal init fallback.**
**Files:** `finetune/adapters.py`, `finetune/olora.py`, `tests/unit/test_adapters.py`, `tests/unit/test_olora.py`.
**Implementation:** `build_peft_config(method)`; QLoRA wires bitsandbytes 4-bit base (CUDA) else documented fallback; `olora.py` QR-orthonormalizes A/B when PEFT-native OLoRA unavailable (SPEC §11.4).
**Acceptance:**
- [ ] Each method yields a valid PEFT config; param summary shows only A/B trainable.
- [ ] OLoRA QR init produces orthonormal columns (unit-tested numerically).
**Dependencies:** T12 **Effort:** 4h **Refs:** SPEC §3.5; IMPL §1.2

### Task 14: Dataset Loader & Trainer
**Tiny SFT loader + PEFT training loop with before/after sampling.**
**Files:** `finetune/dataset.py`, `finetune/trainer.py`, `data/finetune/tiny_sft.jsonl`, `tests/integration/test_finetune.py` (CPU, few steps).
**Implementation:** jsonl loader; trainer runs a handful of steps on CPU; logs before/after generation on a held-out prompt; saves adapter.
**Acceptance:**
- [ ] `finetune --method lora` trains on CPU to completion on the tiny dataset and saves an adapter.
- [ ] Before/after generations are logged; adapter loadable by the transformers backend.
**Dependencies:** T13 **Effort:** 4h **Refs:** SPEC §3.5

### Task 15: Paging Tracer, Events, Replay
**Observer event stream + JSONL sink + AirLLM hook + replay engine.**
**Files:** `paging/events.py`, `paging/tracer.py`, `paging/airllm_hook.py`, `paging/replay.py`, `tests/data/sample_paging.jsonl`, `tests/integration/test_paging.py`.
**Implementation:** Observer (IMPL §2.3); version-guarded AirLLM instrumentation emitting `PagingEvent`s; replay feeds recorded JSONL to sinks (SPEC §3.6 edge case).
**Acceptance:**
- [ ] Real AirLLM run (when present) emits fault/hit/evict events to `results/paging/<hash>.jsonl`.
- [ ] Replay reproduces an event stream from fixture with no AirLLM installed.
**Dependencies:** T9, T14 **Effort:** 4h **Refs:** SPEC §3.6; IMPL §2.3

### Task 16: Visualizer (TUI + static export) & `visualize` command
**Textual live dashboard + matplotlib/HTML static render; CLI command.**
**Files:** `viz/tui.py`, `viz/widgets.py`, `viz/static.py`, `viz/html.py`, `cli/visualize.py`, `cli/finetune.py`, `tests/integration/test_viz.py`.
**Implementation:** Textual app subscribing to the tracer (residency bar across memory hierarchy, fault gauge, per-layer timeline); `static.py` renders PNG; `html.py` standalone page; `visualize` supports `--live` and `--replay <jsonl>`.
**Acceptance:**
- [ ] `localforge visualize --replay tests/data/sample_paging.jsonl` renders without a GPU/AirLLM and exports PNG + HTML.
- [ ] TUI degrades to plain summary with `--no-tui`.
- [ ] `finetune` command exposed and runs the T14 pipeline.
**Dependencies:** T15 **Effort:** 5h **Refs:** SPEC §3.6,§6; IMPL §2.3

---

## Phase 6: Quality, Docs, Release
> After this phase: reproducible, documented, graded-ready repo.

### Task 17: Test Hardening & CI Green
**Raise coverage, finalize skip-markers, make CI fully green on CPU-only.**
**Files:** expand `tests/unit/*`, `tests/integration/*`, `.github/workflows/ci.yml`.
**Acceptance:**
- [ ] `uv run pytest --cov` ≥80% on `core`/`config`/`reporting`; suite green with no engines installed.
- [ ] CI runs ruff + mypy + pytest + CPU smoke `compare` and passes.
**Dependencies:** T16 **Effort:** 4h **Refs:** IMPL §9

### Task 18: README, Results Showcase, Reproducibility & Submission
**Author README with reproduction steps + committed demo results; finalize PRD/PLAN/TODO; prep submission.**
**Files:** `README.md`, committed `results/reports/demo/*`, `docs/PRD.md`/`PLAN.md`/`TODO.md` (final), `docs/REPRODUCE.md`.
**Implementation:** end-to-end demo run on CPU, commit artifacts + figures; document the OS-paging analogy with a captured visualizer screenshot/HTML; add instructor (`rmisegal@gmail.com`) collaborator note.
**Acceptance:**
- [ ] A fresh clone reproduces the demo report via documented `uv` commands.
- [ ] README shows the comparison table + a paging-visualizer figure.
- [ ] PRD/PLAN/TODO present at repo per submission rules; `moamteam-ex05.pdf` template filled separately.
**Dependencies:** T17 **Effort:** 4h **Refs:** SPEC §9,§12; submission guidelines V3

---

## Milestones
| Milestone | After Task | Achieved | Demo-able? |
|---|---|---|---|
| Foundation | T3 | installs, lints, types, probes | smoke import |
| Profiled run | T5 | fake run → RunResult | unit/integration |
| Real inference | T9 | HF/Ollama/AirLLM (when present) | `run` |
| MVP | T12 | compare + report on CPU | full CPU demo |
| Originality | T16 | finetune + paging visualizer | TUI + figures |
| Release | T18 | reproducible, documented | grade-ready |

## Dependency Graph
```
T1→T2→T3→T4→T5→T6→T7→T8→T9→T10→T11→T12→T13→T14→T15→T16→T17→T18
                         └────────────→ T15 (also needs T9)
```
