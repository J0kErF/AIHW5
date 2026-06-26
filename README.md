# localforge

> Forge LLMs on the hardware you have.

**localforge** is a modular, CPU-first Python CLI/toolkit for running and fine-tuning large language models **locally**, and for **measuring and comparing** what that costs in memory and time. It unifies three execution backends — Hugging Face `transformers`, **Ollama** (quantized GGUF), and **AirLLM** (disk-as-virtual-memory layer streaming) — behind one interface, profiles every run, and produces reproducible comparison reports.

Its signature feature is the **Paging Visualizer**: AirLLM streams a model layer-by-layer from disk, which is conceptually identical to an operating system paging memory in and out under demand. localforge instruments that streaming and renders it as a memory-hierarchy view — page faults, residency, evictions — making the OS-theory framing of the course concrete.

> University HW5 — local LLM inference & fine-tuning (lecture L08: Ollama, LoRA/QLoRA/OLoRA, AirLLM, quantization). See `docs/` for the full design, decisions, and reverse-engineering write-ups.

## Why it's interesting

- **Three real backends, one interface.** The same prompt/model runs through `transformers`, `ollama`, and `airllm`; switching is a flag.
- **Measurement is first-class.** Every run is wrapped by a profiler capturing load time, prefill vs decode latency, decode throughput, and peak RAM/VRAM.
- **CPU-first, honest about limits.** Every feature runs on a CPU-only laptop with no GPU, no Ollama daemon, and AirLLM not installed — those paths *skip with a reason*, never crash. CUDA-only paths (NF4/QLoRA/fp16) degrade to documented fallbacks, never faked numbers.
- **The Paging Visualizer** bridges OS virtual-memory theory and real LLM inference (`docs/RE_AIRLLM.md`).

## Results (CPU-only demo, Qwen2.5-0.5B-Instruct)

`uv run localforge compare --suite config/suites/demo.yaml` →

| Backend | Status | Load (s) | Prefill (ms) | Decode (tok/s) | Peak RAM (MB) |
|---|---|---|---|---|---|
| transformers | ok | 6.3 | 412.6 | 11.9 | 3132 |
| ollama | skipped | — | — | — | *(no daemon — run `ollama serve`)* |
| airllm | skipped | — | — | — | *(optional extra not installed)* |

Skipped rows are explained, not dropped — the comparison is honest about what ran. Full artifacts (Markdown/JSON + charts + `report.html`) in [`results/reports/demo/`](results/reports/demo/).

### Paging visualization

`uv run localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct` captures a **real** per-block execution trace; `--replay`/synthetic modes model AirLLM's bounded working set. Example artifact: [`results/paging/`](results/paging/). The contrast — transformers keeps all blocks resident (no eviction) while AirLLM re-streams a small working set from disk every token — is the heart of `docs/RE_AIRLLM.md`.

## Quickstart

```bash
uv sync --group dev          # reproducible env from uv.lock
cp .env.example .env         # add your HF_TOKEN (free, from huggingface.co)

uv run localforge baseline                                   # idle RAM/VRAM
uv run localforge run --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct
uv run localforge compare --suite config/suites/demo.yaml    # 3-backend report
uv run localforge finetune --method lora                     # tiny CPU LoRA train
uv run localforge visualize --no-tui                         # paging visualizer (synthetic)
```

Optional engines: `uv sync --extra ollama` (then `ollama serve`), `uv sync --extra airllm`, `uv sync --extra gpu`.

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for a full from-clone reproduction, and [`docs/SUBMISSION.md`](docs/SUBMISSION.md) for the assignment checklist.

## Commands

| Command | What it does |
|---|---|
| `baseline` | Record idle RAM/VRAM |
| `pull <model>` | Download a model from the HF Hub |
| `run` | Single inference + profiled metrics |
| `compare --suite <yaml>` | Run a model×backend×dtype matrix → report |
| `finetune --method {lora,qlora,olora}` | Train a PEFT adapter (before/after) |
| `visualize [--model/--replay]` | Paging visualizer (TUI / static PNG+HTML) |

## Architecture

CPU-first, layer-based modules under `src/localforge/`: `core` (types/errors/capabilities), `config` (settings + suite builder), `models` (HF acquisition), `backends` (Protocol + registry + transformers/ollama/airllm + runner), `profiling` (Profiler), `finetune` (PEFT), `paging` (tracer/events/replay), `viz` (TUI + static), `reporting` (matrix/charts/report). Design patterns and rationale in [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not slow"        # offline suite (real-model tests are -m slow)
```

CI (`.github/workflows/ci.yml`) runs lint → types → offline tests → CPU smokes on every push.

## Documentation

| Doc | Purpose |
|---|---|
| `docs/SPECIFICATION.md` | what it is and does |
| `docs/IMPLEMENTATION.md` | architecture, patterns, module map |
| `docs/DECISIONS.md` | locked decisions (ADR log) |
| `docs/REVERSE_ENGINEERING.md` · `RE_AIRLLM.md` · `RE_OLLAMA.md` | the RE arm |
| `docs/PRD.md` · `PLAN.md` · `TODO.md` | submission artifacts |
| `docs/REPRODUCE.md` · `SUBMISSION.md` | reproduction + submission |

## License

MIT.
