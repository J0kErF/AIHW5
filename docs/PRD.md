# localforge — Product Requirements (PRD)

> Submission-required artifact (V3 guidelines). Companion to `SPECIFICATION.md` (full detail). Course HW5 — local LLM inference & fine-tuning (L08).

## 1. Problem & Motivation
L08 covers local LLM deployment across many disconnected ideas: Ollama serving, HF model formats (SafeTensors/GGUF), quantization (NF4/QLoRA), LoRA/OLoRA fine-tuning, and AirLLM's disk-as-virtual-memory trick for running giant models in tiny RAM. There is no single tool that lets you run the *same* prompt/model across these execution strategies and **measure** the trade-offs fairly. localforge is that tool.

> **Deliverable shape:** *hybrid* (see `REVERSE_ENGINEERING.md`). No `ex05` spec file exists in the folder; the formal מטלה may live in the lecture recording. Our **working interpretation** is the L08 §9 practical guide, and the hybrid design (RE analysis of AirLLM/Ollama + the `localforge` build on top) is deliberately robust to either a build- or RE-style grading.

## 2. Goal
A reproducible Python CLI/toolkit that satisfies our working interpretation of the L08 §9 practical guide end-to-end:
1. Pull a model from Hugging Face.
2. Run it via Ollama (local OpenAI-compatible API).
3. Measure RAM/GPU baseline.
4. Run AirLLM on CPU (layer-by-layer) and measure latency.
5. Compare loading/GPU/CPU/AirLLM across model sizes — covering SafeTensors/GGUF, prefill vs decode, VRAM, and LoRA/QLoRA/OLoRA fine-tuning.

## 3. Users
- **Instructor (primary):** clones the repo, reproduces every benchmark/figure on a possibly CPU-only machine using only `uv` + an HF token.
- **Students (secondary):** a runnable reference that ties OS virtual-memory theory to real transformer inference.

## 4. Requirements (must-have)
- R1 Pull + register HF models; detect SafeTensors/GGUF.
- R2 Three interchangeable backends (transformers/Ollama/AirLLM) behind one interface.
- R3 Profiler: load_s, prefill_ms, decode_tok_s, peak RAM, peak VRAM — per run.
- R4 Comparison suite → Markdown + JSON + PNG charts + `report.html`.
- R5 LoRA/QLoRA/OLoRA fine-tuning pipeline (PEFT) with before/after demo.
- R6 **Paging Visualizer** (originality): AirLLM layer streaming → OS-style page-fault/residency trace, live TUI + static export + replay.
- R7 **CPU-first**: every feature degrades to "skip with reason" with no GPU/Ollama/AirLLM; nothing GPU-required.
- R8 `uv`-managed, seeded/deterministic, `results/` committed for the demo.

## 5. Success Criteria
- A fresh clone reproduces the demo comparison report and a visualizer figure via documented `uv` commands on CPU only.
- CI (lint + types + tests + CPU smoke `compare`) is green with no engines installed.
- README presents the comparison table + paging figure + the OS-paging analogy.

## 6. Non-Goals
Production serving; full pretraining; new quantization kernels; distributed/multi-GPU; web/chat UI; hosting weights. (See SPECIFICATION.md §11.2.)

## 7. Risks & Mitigations
- AirLLM/bitsandbytes fragility on CPU/Windows → optional extras + replay mode + skip-with-reason.
- OLoRA support varies by PEFT version → QR orthonormal-init fallback wrapper.
- Large model downloads → small default demo model (Qwen2.5-0.5B) for CI; documented scale-up.

## 8. Self-Grade Policy
Honest self-grade per course guidance (no inflation). Grading rigor scales with the self-grade; we target a realistic number reflecting actual completeness and reproducibility.
