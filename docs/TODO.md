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

## Phase 4 — Reporting & CLI ✅
- [x] T10 Result store + suite builder — cartesian expansion, ollama aliases, spec_hash keying
- [x] T11 Matrix + charts + report.html — 3 PNG, skipped rows shown, Jinja2 template
- [x] T12 CLI (pull/baseline/run/compare) — `--help` light; **real CPU `compare` validated** (transformers 9.7 tok/s, 3147 MB; ollama+airllm skip cleanly → report.html)

## Phase 5 — Fine-tuning & Paging Visualizer (originality) ✅
- [x] T13 PEFT adapters + OLoRA QR fallback — only A/B trainable; orthonormal init numerically tested
- [x] T14 Dataset loader + trainer — **real CPU LoRA train validated** (25.6s, <5% trainable, before/after, adapter saved)
- [x] T15 Paging tracer/events/replay — Observer + JSONL + real forward-hook instrumentation + synthetic model; replay works without AirLLM
- [x] T16 Visualizer TUI + static + visualize/finetune cmds — **real per-block capture validated** (Qwen 144 events); PNG/HTML/JSONL export; ASCII-safe plain mode
- [x] T9a/T15a Empirical AirLLM RE — RE_AIRLLM.md: measured transformers trace (24/24 resident, 0 evict) vs modeled AirLLM (8/24, 40 evict); airllm 2.11/optimum 2.2 incompat documented + handled by skip-with-reason

## Phase 6 — Quality, docs, release ✅
- [x] T17 Test hardening + CI green — core/config/reporting **95%** cov; CI = ruff+mypy+offline tests+CPU smokes; green with no engines
- [x] T18 README + results + reproducibility + submission — comprehensive README, committed showcase (compare report + paging figure), REPRODUCE.md, SUBMISSION.md

## Cross-cutting / submission checklist
- [x] CPU-first verified: full demo runs with no GPU, no Ollama, no AirLLM (replay covers viz)
- [x] Seeded determinism (greedy runs)
- [x] `.env` ignored, `.env.example` committed, HF token in SecretStr (never logged)
- [x] `results/reports/demo/*` + `results/paging/*` committed (report + paging figure)
- [ ] **(user)** Push to GitHub + add `rmisegal@gmail.com` as collaborator
- [x] PRD/PLAN/TODO + root README present
- [ ] **(user)** `moamteam-ex05.pdf` from locked template (no extra text), honest self-grade — see docs/SUBMISSION.md
