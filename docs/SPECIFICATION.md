# localforge — Specification

> One toolkit to pull, serve, profile, fine-tune, and *see* large language models run on local hardware — from a full GPU down to a memory-starved laptop.

## 1. Overview

### 1.1 What Is localforge?

**localforge** is a modular Python toolkit and CLI for running and fine-tuning large language models **locally**, and for **measuring and comparing** what that actually costs in memory and time. It unifies three very different execution backends — Hugging Face `transformers`, **Ollama**, and **AirLLM** — behind a single inference interface, instruments each run for RAM/VRAM/latency, and produces reproducible comparison reports.

The problem it solves: the L08 lecture covers a sprawling set of ideas — Ollama serving, Hugging Face model formats (SafeTensors/GGUF), quantization (NF4/QLoRA), LoRA/OLoRA fine-tuning, and AirLLM's trick of treating disk as virtual memory so a 70B model can "run" in 4 GB of RAM. These are normally explored in disconnected notebooks. localforge turns them into one coherent, testable system where you can run the *same* prompt through the *same* model on three backends and get an apples-to-apples table of load time, peak memory, prefill latency, and decode throughput.

Its signature feature is the **Paging Visualizer**: AirLLM streams a model layer-by-layer from disk, which is conceptually identical to an operating system paging memory in and out under demand. localforge instruments that layer streaming and renders it as a live memory-hierarchy view — page faults, residency, mmap-backed loads — making the OS-theory framing of the lecture *visible* rather than abstract.

### 1.2 Target Audience

- **Primary:** the course instructor grading HW5, who will clone the repo and must be able to reproduce every benchmark and figure from a clean machine (possibly CPU-only, no GPU) using only `uv` and a Hugging Face token.
- **Secondary:** students learning local LLM deployment who want a runnable reference implementation that ties OS virtual-memory concepts to real transformer inference.

### 1.3 Key Differentiators

- **Three backends, one interface.** The same `generate()` call runs on `transformers`, Ollama, or AirLLM; switching is a config flag, not a rewrite.
- **Measurement is first-class, not an afterthought.** Every run is wrapped by a profiler that captures load time, peak RSS, peak VRAM, prefill ms, and decode tokens/sec into a typed `RunResult`.
- **Runs where the grader runs.** CPU-first design with graceful CUDA detection; every feature degrades cleanly to CPU-only so nothing is "GPU required."
- **The Paging Visualizer.** No comparable teaching tool surfaces AirLLM's layer streaming as an OS-style paging trace (page faults, residency timeline, mmap zero-copy loads).
- **Reproducible by construction.** `uv.lock`-pinned deps, seeded runs, deterministic report artifacts (Markdown + HTML + PNG) committed under `results/`.

### 1.4 Competitive Landscape

| Capability | localforge | raw `transformers` scripts | Ollama alone | `llama.cpp` |
|---|---|---|---|---|
| Unified multi-backend inference | ✅ | ❌ | ❌ | ❌ |
| Built-in RAM/VRAM/latency profiling | ✅ | ❌ (manual) | partial | partial |
| AirLLM layer-streaming + paging trace | ✅ | ❌ | ❌ | ❌ |
| LoRA/QLoRA/OLoRA fine-tuning pipeline | ✅ | ✅ (DIY) | ❌ | ❌ |
| Apples-to-apples comparison reports | ✅ | ❌ | ❌ | ❌ |
| CPU-only fallback for every feature | ✅ | partial | ✅ | ✅ |

## 2. Core Concepts

| Concept | Definition |
|---|---|
| **Backend** | A pluggable inference engine implementing the `InferenceBackend` protocol. Three exist: `transformers`, `ollama`, `airllm`. |
| **RunSpec** | Immutable description of one inference job: model id, backend, prompt, max tokens, dtype/quantization, seed, device. |
| **RunResult** | Typed record of one run's outputs: generated text + metrics (load_s, prefill_ms, decode_tok_s, peak_ram_mb, peak_vram_mb). |
| **Profiler** | Context manager that samples process RSS and CUDA memory around a run and computes timing splits. |
| **Prefill** | The forward pass over the full prompt (compute-bound); measured separately from decode. |
| **Decode** | Autoregressive token-by-token generation (memory-bandwidth-bound); reported as tokens/sec. |
| **Quantization** | Reducing weight precision (e.g., NF4 4-bit) to shrink memory; configured per RunSpec. |
| **LoRA / QLoRA / OLoRA** | Parameter-efficient fine-tuning: train small low-rank adapters (`W = W0 + BA`) instead of full weights. QLoRA = LoRA over a 4-bit base; OLoRA = orthonormal (QR) adapter init. |
| **Layer Stream** | AirLLM's execution model: load one transformer block's weights from disk, run it, evict it, repeat. |
| **Paging Event** | An instrumented record emitted when a layer is loaded (fault), reused (hit), or evicted — the visualizer's data unit. |
| **Memory Hierarchy** | The Registers→Cache→RAM→SSD/NVMe ladder; the visualizer maps layer residency onto it. |
| **Comparison Matrix** | A table of RunResults across backends × models, the core deliverable artifact. |

## 3. Functional Requirements

### 3.1 Model Acquisition

#### 3.1.1 Pull a model from Hugging Face
**User Story:** As a grader, I want to download a named model from the Hub with one command so that all backends share the same weights.

**Acceptance Criteria:**
- [ ] `localforge pull <model_id>` downloads weights to a local cache using `HF_TOKEN` from env/`.env`.
- [ ] SafeTensors is preferred; the resolved format (SafeTensors/GGUF) is reported.
- [ ] A model registry records id, format, size on disk, and local path.
- [ ] Re-pulling a cached model is a no-op with a clear message.

**Edge Cases:** missing/invalid token → actionable error; gated model → explain license acceptance; disk full → fail before partial corruption.

**Constraints:** default demo models must be small enough to run CPU-only in CI (e.g. a ~0.5–1.5B model); large models are opt-in.

### 3.2 Multi-Backend Inference

#### 3.2.1 Generate text via a selectable backend
**User Story:** As a user, I want to run the same prompt through `transformers`, `ollama`, or `airllm` so that I can compare them.

**Acceptance Criteria:**
- [ ] `localforge run --backend {transformers|ollama|airllm} --model <id> --prompt "..."` returns text + a `RunResult`.
- [ ] All three backends implement one `InferenceBackend` protocol; selection is by name via a factory.
- [ ] `ollama` backend talks to the local OpenAI-compatible API (`localhost:11434`) and surfaces a clear error if the daemon is down.
- [ ] `airllm` backend runs layer-by-layer on CPU and completes without exhausting RAM on the demo model.
- [ ] Generation is reproducible for a fixed seed + greedy decoding.

**Edge Cases:** backend unavailable (no Ollama daemon / AirLLM not installed) → skip with reason, never crash the suite; CUDA absent → `transformers` runs on CPU.

#### 3.2.2 Quantized inference
**Acceptance Criteria:**
- [ ] A RunSpec may request NF4 4-bit quantization (bitsandbytes) when CUDA is present.
- [ ] On CPU-only hosts, requesting CUDA-only quantization yields a clear "unsupported on this device, falling back to <dtype>" message rather than a hard failure.

### 3.3 Profiling & Measurement

#### 3.3.1 Establish a baseline and profile every run
**User Story:** As a grader, I want each run measured identically so the comparison is fair.

**Acceptance Criteria:**
- [ ] A baseline command records idle RAM and, if present, idle VRAM before any model loads.
- [ ] Every run captures: load_s, prefill_ms, decode_tok_s, peak_ram_mb, peak_vram_mb.
- [ ] Prefill and decode latency are measured separately.
- [ ] Metrics are persisted as JSON under `results/` keyed by RunSpec hash.

**Edge Cases:** VRAM sampling on CPU-only host returns `null`, not an error; very short generations still report decode_tok_s (or `n/a` if < 2 tokens).

### 3.4 Comparison & Reporting

#### 3.4.1 Run a comparison suite
**Acceptance Criteria:**
- [ ] `localforge compare --suite <yaml>` executes a matrix of (model × backend × dtype) RunSpecs.
- [ ] Output is a Comparison Matrix as Markdown table + machine-readable JSON.
- [ ] Static charts (PNG) are generated: peak memory by backend, decode tok/s by backend, load time by model size.
- [ ] Unavailable backends are reported as "skipped (reason)" rows, not omitted silently.
- [ ] A single `report.html` bundles tables + charts + run metadata (git SHA, host specs, seed).

### 3.5 Fine-Tuning

#### 3.5.1 LoRA / QLoRA / OLoRA adapter training
**User Story:** As a user, I want to fine-tune a small adapter on a tiny dataset so that I can demonstrate PEFT end-to-end.

**Acceptance Criteria:**
- [ ] `localforge finetune --method {lora|qlora|olora} --model <id> --data <path>` trains an adapter via PEFT and saves it.
- [ ] `print_trainable_parameters`-style summary shows only A/B matrices are trainable.
- [ ] A merged or adapter-loaded model can then be run through §3.2 inference.
- [ ] Training runs to completion on CPU for the demo (tiny dataset, few steps); QLoRA path activates only with CUDA, else explains the fallback.
- [ ] Before/after generations on a held-out prompt are logged to show the adapter changed behavior.

### 3.6 Paging Visualizer (Originality Hook)

#### 3.6.1 Instrument and visualize AirLLM layer streaming
**User Story:** As a learner, I want to *see* AirLLM page layers in and out so that the OS virtual-memory analogy becomes concrete.

**Acceptance Criteria:**
- [ ] The AirLLM backend emits a `PagingEvent` stream (layer index, action ∈ {fault,hit,evict}, bytes, source ∈ {mmap,disk}, timestamp).
- [ ] A live **TUI** (Rich/Textual) renders: a residency bar across the memory hierarchy, a running page-fault count, and a per-layer timeline during generation.
- [ ] After a run, the event stream is exported to `results/paging/<run>.json` and rendered to a static **HTML + PNG** timeline so it appears in reports without a live terminal.
- [ ] A synthetic/replay mode can render a recorded event stream when AirLLM is not installed, so the visualizer is demonstrable on any machine.

**Edge Cases:** extremely fast layers → events are batched for render; missing instrumentation hooks → fall back to coarse per-layer timing.

## 4. Architecture Overview

### 4.1 System Components
- **`cli`** — Typer-based command surface (`pull`, `run`, `compare`, `finetune`, `visualize`, `baseline`).
- **`backends`** — `InferenceBackend` protocol + `transformers`, `ollama`, `airllm` implementations + a factory/registry.
- **`profiling`** — `Profiler` context manager, system probes (psutil/CUDA), `RunResult` assembly.
- **`models`** — Hugging Face acquisition, local registry, format detection (SafeTensors/GGUF).
- **`finetune`** — PEFT adapters (LoRA/QLoRA/OLoRA), dataset loading, trainer wrapper.
- **`paging`** — `PagingEvent` model, AirLLM instrumentation hooks, replay engine.
- **`viz`** — Textual TUI dashboard + static HTML/PNG renderers (matplotlib/Jinja2).
- **`reporting`** — Comparison Matrix assembly, Markdown/JSON/HTML report writers.
- **`config`** — Typed settings (pydantic), `.env` loading, suite YAML parsing.
- **`core`** — Shared types (`RunSpec`, `RunResult`, enums), errors, logging.

### 4.2 Component Interactions
`cli` parses a command into one or more `RunSpec`s → `backends` factory resolves a backend → `profiling.Profiler` wraps the backend call → backend (for AirLLM) emits `PagingEvent`s into `paging` → results flow to `reporting`, paging streams to `viz`. Data flow is synchronous and single-process; the only async/IO boundary is the HTTP call to the Ollama daemon. No shared mutable global state — everything is passed as typed values.

### 4.3 External Integrations
- **Hugging Face Hub** — weight + tokenizer download; fallback: clear error instructing token setup.
- **Ollama daemon** — local OpenAI-compatible server on `:11434`; fallback: backend marked unavailable, suite continues.
- **CUDA/bitsandbytes** — optional acceleration & 4-bit; fallback: CPU dtype path.

## 5. Data Model

### 5.1 Core Entities

#### RunSpec
| Field | Type | Required | Description | Constraints |
|---|---|---|---|---|
| model_id | str | Yes | HF model id or Ollama tag | non-empty |
| backend | enum | Yes | transformers \| ollama \| airllm | — |
| prompt | str | Yes | input text | — |
| max_new_tokens | int | Yes | generation cap | 1–4096 |
| dtype | enum | No | fp32 \| fp16 \| bf16 \| nf4 | default fp32 (CPU) |
| seed | int | No | RNG seed | default 0 |
| device | enum | No | cpu \| cuda \| auto | default auto |

#### RunResult
| Field | Type | Required | Description |
|---|---|---|---|
| spec_hash | str | Yes | stable hash of the RunSpec |
| text | str | Yes | generated output |
| load_s | float | Yes | model load seconds |
| prefill_ms | float | Yes | prompt forward-pass ms |
| decode_tok_s | float\|null | Yes | decode throughput |
| peak_ram_mb | float | Yes | peak process RSS |
| peak_vram_mb | float\|null | Yes | peak CUDA memory (null on CPU) |
| backend_available | bool | Yes | false ⇒ skipped row |
| note | str | No | skip reason / fallback note |

#### PagingEvent
| Field | Type | Required | Description |
|---|---|---|---|
| layer | int | Yes | transformer block index |
| action | enum | Yes | fault \| hit \| evict |
| bytes | int | Yes | weight bytes moved |
| source | enum | Yes | mmap \| disk |
| t_ms | float | Yes | ms since run start |

### 5.2 Relationships
- RunSpec → produces one → RunResult (1:1 by spec_hash)
- RunResult (airllm) → has many → PagingEvent (1:N)
- Suite → has many → RunSpec (1:N)

### 5.3 Data Lifecycle
RunResults and PagingEvent streams are written once to `results/` as immutable JSON keyed by spec hash; reports are regenerated from them. Model weights live in the HF cache and are never deleted by localforge. No database; the filesystem under `results/` is the system of record.

## 6. User Interface

### 6.1 Interface Type
Primary: **CLI** (Typer). Secondary: **TUI** (Textual) for the live Paging Visualizer. Reports are static **HTML/PNG/Markdown**. No web server, no browser dependency for core flows.

### 6.2 Key Screens (CLI commands & TUI)
- `pull` — download + register a model (progress + summary).
- `baseline` — print idle RAM/VRAM.
- `run` — single inference; prints text + metrics table.
- `compare` — run a suite; prints Comparison Matrix; writes report.
- `finetune` — train an adapter; prints param summary + before/after sample.
- `visualize` — launch the Textual TUI (live or replay) and export static artifacts.

### 6.3 Accessibility/Output
All TUI output has a `--no-tui` plain-text equivalent so it works over SSH/CI logs; colors degrade to ASCII.

## 9. Deployment Model

### 9.1 Target Environments
Local developer/grader machines: Windows, macOS, Linux. No cloud component.

### 9.2 Distribution Method
Git repository + `uv` project. `uv sync` reproduces the locked environment; `uv run localforge ...` invokes the CLI. Optionally `pipx`/`uv tool install` for a global entry point.

### 9.3 Configuration
`.env` (HF_TOKEN, OLLAMA_BASE_URL, cache dirs) + typed settings + suite YAML files under `config/`. CLI flags override config which overrides env defaults.

### 9.4 System Requirements
Minimum: Python 3.11+, ~8 GB RAM, ~10 GB free disk for demo models, CPU-only OK. Optional: NVIDIA GPU + CUDA for fp16/NF4/QLoRA paths.

## 10. Performance Requirements

### 10.1 Targets (demo model, grader machine)
- Demo end-to-end `compare` suite completes in < 10 min on CPU-only.
- AirLLM demo run completes without exceeding a configurable RAM ceiling (default 4 GB working set for layer streaming).
- Profiler overhead < 5% of run wall time.

### 10.2 Determinism
Seeded greedy runs produce identical text across repeats on the same host.

## 11. Constraints & Non-Goals

### 11.1 Technical Constraints
- Python 3.11+, `uv`-managed (no plain `venv`/`pip install` in docs).
- Must run, with graceful degradation, on a CPU-only machine with no Ollama and no AirLLM installed (those paths skip, not crash).
- Hugging Face access via token; no credentials committed.

### 11.2 Non-Goals
- **Not a production inference server** — no auth, scaling, batching, or multi-tenant serving.
- **Not a model trainer from scratch** — only PEFT adapters, never full pretraining.
- **Not a new quantization kernel** — uses bitsandbytes/transformers, doesn't reimplement them.
- **Not a GPU cluster / distributed system** — single process, single host.
- **Not a chat UI / web app** — CLI + TUI + static reports only.
- **Not a model hub mirror** — relies on Hugging Face, doesn't host weights.

### 11.3 Assumptions
- The grader can obtain a free Hugging Face token and accept any required model licenses.
- **Target is CPU-only (see DECISIONS.md D1).** CPU paths are the design center, not a fallback. The hero comparison is three *real* CPU backends — `transformers` (full residency), AirLLM (layer streaming), and Ollama (quantized GGUF on CPU). CUDA-only paths (NF4/QLoRA/fp16) are demonstrated as code + documented fallbacks with the lecture's rationale, never as fabricated numbers.
- Ollama and AirLLM may be absent; the system must remain demonstrable via replay/skip. Ollama install is recommended (DECISIONS.md open items) to realize the full three-way comparison.
- Real CPU fine-tuning = LoRA + OLoRA (QR init); QLoRA is documented/CUDA-guarded.

### 11.4 Open Questions
- **[TBD: default demo model]** Options: a ~0.5B (TinyLlama/Qwen2.5-0.5B) for CI speed vs a 1.5–3B for more realistic numbers. Leaning small for CI + a documented "scale-up" model for the report.
- **[TBD: OLoRA availability]** PEFT OLoRA init support varies by version; if unavailable, implement orthonormal QR init as a thin custom wrapper and document it.

## 12. Future Considerations
- **v1.1**: GGUF/`llama.cpp` backend as a fourth engine.
- **v1.1**: speculative-decode / KV-cache paging experiments tied to the visualizer.
- **v2.0**: optional web dashboard (FastAPI + HTMX) rendering the same report data.
