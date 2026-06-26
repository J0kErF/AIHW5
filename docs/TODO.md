# localforge — TODO

> Submission-required tracker. Mirrors TASKS.md. Mark `[x]` as completed. Status: **planning complete, implementation not started.**

## Phase 1 — Foundation ✅
- [x] T1 Scaffold uv project, configs, CI, dir tree — `uv sync`/`ruff`/`mypy`/`pytest` green
- [x] T2 Core types/errors/hashing/logging — JSON round-trip + stable spec_hash (100% cov)
- [x] T3 Capability probes + typed settings — never raise on bare machine; token redacted

## Phase 2 — Profiling & backend core ✅
- [x] T4 Profiler context manager + probes + baseline — VRAM=None on CPU; bg RSS sampler
- [x] T5 Backend Protocol/registry/runner/FakeBackend — skip-not-crash on unavailable

## Phase 3 — Models & real backends ✅
- [x] T6 HF acquire/registry/format detection — re-pull no-op, missing token→AuthError
- [x] T7 Transformers backend (+NF4 path) — **real CPU greedy run validated** (Qwen2.5-0.5B, transformers 5.x); NF4-on-CPU falls back to fp32 with note
- [x] T8 Ollama backend — OpenAI-compatible `/v1` (stdlib urllib+SSE); daemon down → skipped; RE_OLLAMA.md written
- [x] T9 AirLLM backend (layer streaming) — absent→skipped cleanly; tracer hook reserved for T15  **← MVP backends done**

## Phase 4 — Reporting & CLI
- [ ] T10 Result store + suite builder — cartesian expansion, spec_hash keying
- [ ] T11 Matrix + charts + report.html — ≥3 PNG, skipped rows shown, deterministic
- [ ] T12 CLI (pull/baseline/run/compare) — `--help` light, CPU compare end-to-end

## Phase 5 — Fine-tuning & Paging Visualizer (originality)
- [ ] T13 PEFT adapters + OLoRA QR fallback — only A/B trainable; orthonormal init tested
- [ ] T14 Dataset loader + trainer — CPU LoRA completes, before/after logged, adapter loadable
- [ ] T15 Paging tracer/events/replay — real AirLLM emits events; replay works without AirLLM
- [ ] T16 Visualizer TUI + static + visualize/finetune cmds — `--replay` exports PNG/HTML, no GPU

## Phase 6 — Quality, docs, release
- [ ] T17 Test hardening + CI green — ≥80% core coverage, green with no engines
- [ ] T18 README + results + reproducibility + submission — fresh clone reproduces demo report

## Cross-cutting / submission checklist
- [ ] CPU-first verified: full demo runs with no GPU, no Ollama, no AirLLM (replay covers viz)
- [ ] Seeded determinism verified
- [ ] `.env` ignored, `.env.example` committed, HF token never logged
- [ ] `results/reports/demo/*` committed (report + paging figure)
- [ ] Instructor `rmisegal@gmail.com` added as GitHub collaborator
- [ ] PRD/PLAN/TODO + root README present
- [ ] `moamteam-ex05.pdf` filled from locked template (no extra text), honest self-grade
