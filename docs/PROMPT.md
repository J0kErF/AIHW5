# localforge — Claude Code Implementation Prompt

> Single-shot, self-contained prompt. Build the entire project from `uv init` to a reproducible, graded-ready repo. Follow the steps in order; do not skip ahead. Run lint/type/test after each phase.

## Project Overview

Build **localforge**, a modular Python CLI/toolkit for running and fine-tuning LLMs **locally** and measuring what it costs. It unifies three backends — Hugging Face `transformers`, **Ollama** (local OpenAI-compatible API), and **AirLLM** (disk-as-virtual-memory layer streaming) — behind one `InferenceBackend` Protocol, wraps every run in a profiler (load/prefill/decode/peak-RAM/peak-VRAM), produces apples-to-apples comparison reports (Markdown+JSON+HTML+PNG), runs a LoRA/QLoRA/OLoRA fine-tuning pipeline, and ships a signature **Paging Visualizer** that turns AirLLM's layer streaming into an OS-style page-fault/residency trace (live Textual TUI + static export + replay). **Hard rule: CPU-first.** Every feature must run on a CPU-only machine with no GPU, no Ollama daemon, and AirLLM not installed — those paths *skip with a reason*, never crash. Managed entirely with `uv`.

## Tech Stack

| Layer | Technology | Version (min, uv-resolved) |
|---|---|---|
| Language | Python | 3.11+ |
| Env/deps | uv | latest |
| CLI | Typer | ≥0.12 |
| Console/TUI | Rich / Textual | ≥13 / ≥0.60 |
| ML core | PyTorch | ≥2.2 |
| Models | transformers / huggingface_hub | ≥4.44 / ≥0.24 |
| PEFT | peft | ≥0.12 |
| Quant | bitsandbytes | ≥0.43 (CUDA-only path) |
| Accel | accelerate | ≥0.33 |
| Layer streaming | airllm | ≥2.11 (optional) |
| Ollama | ollama | ≥0.3 (optional) |
| Probe | psutil | ≥6.0 |
| Config | pydantic / pydantic-settings | ≥2.7 |
| Data/plots | pandas / matplotlib | ≥2.2 / ≥3.9 |
| Templating | Jinja2 | ≥3.1 |
| Test | pytest / pytest-cov | ≥8.0 |
| Lint/types | ruff / mypy | ≥0.6 / ≥1.11 |

## Project Structure
```
localforge/
├── pyproject.toml  uv.lock  README.md  .env.example  .gitignore  .gitattributes
├── .github/workflows/ci.yml
├── docs/  (SPECIFICATION, IMPLEMENTATION, TASKS, PROMPT, BRANDING, PRD, PLAN, TODO, REPRODUCE)
├── config/ settings.toml  suites/{demo,full}.yaml
├── data/finetune/tiny_sft.jsonl
├── src/localforge/
│   ├── __init__.py  __main__.py
│   ├── core/      types.py errors.py hashing.py logging.py capabilities.py
│   ├── config/    settings.py suite.py
│   ├── models/    acquire.py registry.py formats.py
│   ├── backends/  base.py transformers_backend.py ollama_backend.py airllm_backend.py runner.py
│   ├── profiling/ profiler.py probes.py baseline.py
│   ├── finetune/  adapters.py olora.py dataset.py trainer.py
│   ├── paging/    events.py tracer.py airllm_hook.py replay.py
│   ├── viz/       tui.py widgets.py static.py html.py
│   ├── reporting/ store.py matrix.py charts.py report.py
│   ├── cli/       app.py pull.py run.py compare.py finetune.py visualize.py baseline.py
│   └── templates/ report.html.j2
└── tests/  conftest.py  unit/  integration/  data/
```

## Dependencies (install commands)
```bash
uv init --package localforge && cd localforge
uv add typer rich textual torch transformers huggingface_hub peft accelerate \
       psutil pydantic pydantic-settings pandas matplotlib jinja2
uv add --optional gpu bitsandbytes
uv add --optional ollama ollama
uv add --optional airllm airllm
uv add --group dev pytest pytest-cov ruff mypy
uv lock
```

## Configuration Files

### pyproject.toml (key sections)
```toml
[project]
name = "localforge"
requires-python = ">=3.11"
[project.scripts]
localforge = "localforge.cli.app:app"
[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E","F","I","UP","B","SIM"]
[tool.mypy]
python_version = "3.11"
packages = ["localforge"]
strict = true
[tool.pytest.ini_options]
addopts = "-q --cov=localforge --cov-report=term-missing"
testpaths = ["tests"]
markers = ["needs_cuda","needs_ollama","needs_airllm","slow"]
```

### .env.example
```
HF_TOKEN=
OLLAMA_BASE_URL=http://localhost:11434/v1
LOCALFORGE_CACHE=
```

### .gitignore (essentials)
```
.venv/  __pycache__/  *.pyc  .env  .pytest_cache/  .ruff_cache/  .mypy_cache/
results/runs/  results/paging/  results/baseline/  ~/.cache/localforge
!results/reports/demo/
```

### config/suites/demo.yaml
```yaml
suite_id: demo
models: ["Qwen/Qwen2.5-0.5B-Instruct"]
backends: ["transformers", "ollama", "airllm"]
dtypes: ["fp32"]
prompt: "Explain virtual memory in one sentence."
max_new_tokens: 64
seed: 0
```

## Data Model (inline — `src/localforge/core/types.py`)
```python
class Backend(str, Enum): TRANSFORMERS="transformers"; OLLAMA="ollama"; AIRLLM="airllm"; FAKE="fake"
class Dtype(str, Enum): FP32="fp32"; FP16="fp16"; BF16="bf16"; NF4="nf4"
class Device(str, Enum): CPU="cpu"; CUDA="cuda"; AUTO="auto"
class PageAction(str, Enum): FAULT="fault"; HIT="hit"; EVICT="evict"
class PageSource(str, Enum): MMAP="mmap"; DISK="disk"

class RunSpec(BaseModel):           # frozen
    model_id: str; backend: Backend; prompt: str
    max_new_tokens: int = Field(ge=1, le=4096)
    dtype: Dtype = Dtype.FP32; seed: int = 0; device: Device = Device.AUTO

class RunResult(BaseModel):
    spec_hash: str; text: str
    load_s: float; prefill_ms: float; decode_tok_s: float | None
    peak_ram_mb: float; peak_vram_mb: float | None
    backend_available: bool = True; note: str | None = None

class PagingEvent(BaseModel):
    layer: int; action: PageAction; bytes: int; source: PageSource; t_ms: float
```
On-disk record (no DB): `results/runs/<spec_hash>.json`, `results/paging/<spec_hash>.jsonl`, `results/suites/<id>.json`, `results/reports/<id>/{report.html,*.png,matrix.md,matrix.json}`. `spec_hash` = sha256 of canonical-JSON RunSpec.

## Environment Variables
| Variable | Required | Default | Description |
|---|---|---|---|
| HF_TOKEN | for gated/HF pulls | — | Hugging Face token |
| OLLAMA_BASE_URL | no | http://localhost:11434/v1 | Ollama OpenAI-compat endpoint |
| LOCALFORGE_CACHE | no | ~/.cache/localforge | cache root |

## Error Handling
| Category | Class | Behavior |
|---|---|---|
| Config (bad YAML) | `ConfigError` | exit 2, line-pointed |
| Auth (no HF_TOKEN) | `AuthError` | actionable message |
| Availability (Ollama down / AirLLM absent) | `BackendUnavailable` | **non-fatal**: RunResult(backend_available=False, note=reason) |
| Capability (NF4 on CPU) | warning | fallback to fp32, note recorded |
| Resource (OOM) | `ResourceError` | suggest smaller model/ceiling |
| Internal | `LocalforgeError` | exit 1 + log path |
Catch engine exceptions at the `runner` boundary; only config/auth are fatal — a `compare` suite always completes with skipped rows.

## Implementation Order

### Step 1 — Scaffolding (TASKS T1)
Create the tree, configs, CI, empty packages. **🔍 Checkpoint:** `uv sync --extra dev`, `uv run ruff check .`, `uv run mypy src`, `uv run pytest` all succeed (0 tests).

### Step 2 — Core types/errors/hashing (T2)
Implement the data model above + exception tree + `spec_hash()` + structured logging (redact HF token). Tests: JSON round-trip; hash stability/sensitivity.

### Step 3 — Capabilities + settings (T3)
`probe_capabilities()` (lazy imports + socket probe of `:11434`) returns `(ok, reason)` per engine, never raises. pydantic-settings: defaults<.env<env<override. **🔍 Checkpoint:** probes return cleanly on a bare machine.

### Step 4 — Profiler (T4)
Context-manager `Profiler` (pattern below) + psutil/CUDA probes + `baseline()`.
```python
class Profiler:
    def __enter__(self): self.t0=perf_counter(); return self
    def mark_prefill_done(self): self.prefill_ms=(perf_counter()-self.t0)*1e3
    def add_tokens(self, n, secs): self._tok=n; self._dec=secs
    def __exit__(self,*e):
        self.decode_tok_s = (self._tok/self._dec) if self._dec>0 else None
        self.peak_ram_mb = peak_rss_mb(); self.peak_vram_mb = cuda_peak_mb()  # None on CPU
```

### Step 5 — Backend Protocol + registry + runner + FakeBackend (T5)
```python
@runtime_checkable
class InferenceBackend(Protocol):
    name: str
    def is_available(self) -> tuple[bool,str]: ...
    def load(self, spec: RunSpec) -> None: ...
    def generate(self, spec: RunSpec) -> Iterator[Token]: ...  # first yield after prefill marker
```
`runner.run_spec(spec)`: resolve backend → if unavailable return skipped RunResult → else wrap in Profiler, iterate tokens, build RunResult. `FakeBackend` yields deterministic tokens for tests. **🔍 Checkpoint:** `make_backend("fake")` → valid RunResult; unknown→error; unavailable→skipped, no crash.

### Step 6 — Models (T6)
`acquire.pull()` via `snapshot_download(token=...)`; registry json; `formats.detect()` SafeTensors/GGUF + size. Missing token→`AuthError`. Re-pull is no-op.

### Step 7 — Transformers backend (T7)
Load model+tokenizer, greedy generate, `mark_prefill_done()` after first forward; NF4 via bitsandbytes only if CUDA else warn+fp32. Seeded → reproducible.

### Step 8 — Ollama backend (T8)
Client to `OLLAMA_BASE_URL`; `is_available()` pings daemon; stream tokens; daemon down → `(False, reason)`.

### Step 9 — AirLLM backend (T9)
Wrap AirLLM layered model on CPU; respect `airllm_ram_ceiling_mb`; expose layer boundaries for the paging hook; soft import → absent ⇒ skipped. **🔍 Checkpoint (MVP):** `run` works on CPU with transformers/fake; ollama/airllm skip cleanly.

### Step 10 — Result store + suite builder (T10)
Repository `ResultStore.save/load`; `expand_suite(yaml)` = cartesian product model×backend×dtype → RunSpecs; suite metadata captures git SHA + host.

### Step 11 — Matrix + charts + HTML report (T11)
pandas matrix; matplotlib PNGs (peak mem, decode tok/s, load time); Jinja2 `report.html` incl. skipped rows + metadata. Deterministic regeneration.

### Step 12 — CLI (T12)
Typer app; commands `pull/baseline/run/compare`; Rich tables; lazy per-command imports; `--no-tui` plain output. **🔍 Checkpoint:** `localforge compare --suite config/suites/demo.yaml` runs CPU end-to-end and writes a report; `--help` loads with no heavy imports.

### Step 13 — PEFT adapters + OLoRA fallback (T13)
```python
def build_peft_config(method):  # lora|qlora|olora
    base = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05)
    if method=="qlora": ...  # 4-bit base via bitsandbytes when CUDA, else documented fallback
    if method=="olora": ...  # PEFT-native if available else QR-orthonormal init (olora.py)
    return base
```
Param summary shows only A/B trainable; OLoRA QR init numerically orthonormal (tested).

### Step 14 — Dataset + trainer (T14)
jsonl SFT loader; CPU trainer (few steps) on `tiny_sft.jsonl`; log before/after generation on held-out prompt; save adapter; loadable by transformers backend.

### Step 15 — Paging tracer/events/replay (T15)
```python
class PagingTracer:
    def __init__(self): self._sinks=[]
    def subscribe(self,sink): self._sinks.append(sink)
    def emit(self,ev): [s(ev) for s in self._sinks]
```
Version-guarded `airllm_hook` emits fault/hit/evict → `results/paging/<hash>.jsonl`; `replay()` feeds fixture JSONL to sinks with no AirLLM installed.

### Step 16 — Visualizer + visualize/finetune commands (T16)
Textual TUI subscribing to tracer: residency bar across Registers→Cache→RAM→SSD/NVMe, fault gauge, per-layer timeline; `static.py` PNG, `html.py` standalone page; `visualize --live|--replay`, `--no-tui` plain fallback. Wire `finetune` command. **🔍 Checkpoint:** `localforge visualize --replay tests/data/sample_paging.jsonl` renders + exports PNG/HTML with no GPU/AirLLM.

### Step 17 — Test hardening + CI green (T17)
Coverage ≥80% on core/config/reporting; `skipif` markers for cuda/ollama/airllm/slow; CI = ruff+mypy+pytest+CPU smoke `compare`. Suite green with no engines installed.

### Step 18 — README + results + reproducibility + submission (T18)
Run CPU demo, commit `results/reports/demo/*` + a paging figure; README shows comparison table + visualizer figure + `uv` reproduction steps; finalize `docs/PRD.md`/`PLAN.md`/`TODO.md`/`REPRODUCE.md`; note instructor collaborator `rmisegal@gmail.com`.

## Testing Requirements
- Unit: types/hashing/suite expansion/profiler math/OLoRA QR/capabilities/matrix.
- Integration: fake-backend run, compare end-to-end, paging replay→render, report.html, tiny CPU LoRA finetune.
- The whole run→compare→report→visualize pipeline must be testable with **FakeBackend + recorded paging fixture — zero downloads, no GPU**.
- Run: `uv run pytest`.

## Quality Checks (final)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run mypy src` clean.
- [ ] `uv run pytest --cov` ≥80% on core/config/reporting, 0 failures with no engines installed.
- [ ] `uv run localforge compare --suite config/suites/demo.yaml` produces matrix.md/json + ≥3 PNG + report.html on CPU.
- [ ] `uv run localforge visualize --replay tests/data/sample_paging.jsonl` exports PNG + HTML with no AirLLM.
- [ ] Fresh clone reproduces the demo report from documented `uv` commands.
- [ ] README documents setup, env vars, results, and the OS-paging analogy.
