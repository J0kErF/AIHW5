# localforge

> Forge LLMs on the hardware you have.

**localforge** is a modular Python CLI/toolkit for running and fine-tuning large language models **locally**, and for **measuring and comparing** what that costs in memory and time. It unifies three execution backends — Hugging Face `transformers`, **Ollama** (quantized GGUF), and **AirLLM** (disk-as-virtual-memory layer streaming) — behind one interface, profiles every run (load / prefill / decode / RAM / VRAM), and produces reproducible comparison reports.

Its signature feature is the **Paging Visualizer**: AirLLM streams a model layer-by-layer from disk, which is conceptually identical to an operating system paging memory in and out under demand. localforge instruments that streaming and renders it as a live memory-hierarchy view — page faults, residency, mmap-backed loads — making the OS-theory framing of the course concrete.

> **Status:** under construction. Planning is complete (`docs/`); implementation follows `docs/TASKS.md`. This README is fleshed out with results and figures in the final task (T18).

## Design at a glance

- **CPU-first.** Every feature runs on a CPU-only machine; GPU/Ollama/AirLLM paths *skip with a reason* rather than crash. See `docs/DECISIONS.md`.
- **Three real CPU backends** compared apples-to-apples: `transformers`, `ollama`, `airllm`.
- **PEFT fine-tuning:** LoRA / QLoRA / OLoRA.
- **Reverse-engineering arm:** empirical analysis of AirLLM/Ollama internals, validated by the visualizer (`docs/REVERSE_ENGINEERING.md`).

## Quickstart (will be runnable as tasks land)

```bash
uv sync --group dev          # reproducible env from uv.lock
cp .env.example .env         # add your HF_TOKEN
uv run localforge version
```

## Documentation

| Doc | Purpose |
|---|---|
| `docs/SPECIFICATION.md` | what it is and does |
| `docs/IMPLEMENTATION.md` | architecture, patterns, module map |
| `docs/TASKS.md` | ordered build plan |
| `docs/DECISIONS.md` | locked decisions (ADR log) |
| `docs/REVERSE_ENGINEERING.md` | the RE deliverable |
| `docs/PRD.md` / `PLAN.md` / `TODO.md` | submission artifacts |

## License

MIT.
