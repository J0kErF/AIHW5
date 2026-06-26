# localforge — Decision Log (ADR)

> Locked decisions from the requirements pass (2026-06-26). These constrain SPECIFICATION/IMPLEMENTATION/TASKS. Source of truth for HW5 = `L08-summary-Lora-AirLLM.pdf` (esp. §9), as the formal `ex05` spec is not in the folder and the recording is inaccessible.

## D1 — Target hardware: CPU-only laptop
- **Decision:** CPU-only is the **primary design center**, not just a fallback.
- **Consequences:**
  - The hero comparison is **three real CPU backends**: `transformers` (full residency, fp32), **AirLLM** (layer-by-layer streaming), and **Ollama** (quantized GGUF on CPU). This is a genuine, runnable apples-to-apples story — the best possible framing of L08 §9 on a laptop.
  - CUDA-only paths (NF4, QLoRA 4-bit, fp16) are **documented fallbacks** with explicit "why unsupported on this device" notes + the theory from the lecture — demonstrated via code + a CUDA-guarded path, not faked numbers.
  - Real fine-tuning shown on CPU = **LoRA** (tiny model, few steps) + **OLoRA** (QR init runs on CPU). **QLoRA** is documented/guarded (needs bitsandbytes+CUDA).
  - AirLLM and the **Paging Visualizer** become the centerpiece (memory pressure is the whole point). Working-set ceiling default 4 GB.

## D2 — Ambition: top-grade maximal
- **Decision:** Build the full hybrid (real RE + complete `localforge`), substantial modular codebase, rich reproducible report.
- **Consequences:** All 18 build tasks + 2 RE tasks executed; depth + reproducibility prioritized over raw line count (the `~10k LOC` figure is an ambition, not a gate — see `REVERSE_ENGINEERING.md` §7). Honest self-grade per course guidance.

## D3 — Reverse-engineering depth: full empirical
- **Decision:** Clone **AirLLM** (primary) and **Ollama** source at pinned commits, statically map them, **dynamically trace a real CPU run**, and validate the Paging Visualizer against the traced behavior.
- **Consequences:**
  - New `external/` (git submodules or vendored at pinned commit, license-respecting; no weights committed).
  - RE tasks produce `RE_AIRLLM.md` + `RE_OLLAMA.md` with **file:line citations** to the pinned source, Mermaid diagrams, and a captured `results/traces/airllm_demo.jsonl`.
  - The visualizer's measured fault/residency timeline is reported **against** the RE's predictions; discrepancies are disclosed, not hidden.
  - Requires AirLLM to actually run on CPU with a small model — confirmed feasible (AirLLM supports CPU/compression mode); version pinned in `pyproject.toml`.

## D4 — Timeline: flexible / no fixed date
- **Decision:** Plan for maximal quality without time pressure; still ship in demo-able milestones (MVP after T12, originality after T16).
- **Consequences:** No scope trimming forced; each phase ends in a runnable, committable state so progress is always defensible.

## D5 — Demo model (derived, not asked)
- **Decision:** CI/smoke model = `Qwen/Qwen2.5-0.5B-Instruct` (fast on CPU); **showcase** run uses a documented larger model (e.g. a 3B) to make AirLLM's "big model, small RAM" point visible in the report. Both pinned in suites.

## Open items to confirm with user (non-blocking)
- **Ollama install:** the real Ollama-CPU datapoint needs the Ollama binary/daemon installed locally. If you'd rather not install it, Ollama gracefully becomes a skipped/replayed row and the comparison runs on transformers vs AirLLM. **Recommend installing Ollama** for the full three-way story.
- **Submission identity:** group code `moamteam` (from prior `moamteam-exNN` files) + team member details for the locked `moamteam-ex05.pdf` template — needed only at submission time, not for building.
