# HW5 Submission Checklist

> Maps localforge to the course submission rules (`submit.txt`, software guidelines V3). Group code: **moamteam**. Assignment: local LLM inference & fine-tuning (L08).

## What's in the repo (required artifacts)
- [x] Root `README.md` with setup, results, and architecture.
- [x] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` (the mandated PRD/PLAN/TODO markdown).
- [x] Full design set: `SPECIFICATION.md`, `IMPLEMENTATION.md`, `TASKS.md`, `DECISIONS.md`, `PROMPT.md`, `BRANDING.md`.
- [x] Reverse-engineering write-ups: `REVERSE_ENGINEERING.md`, `RE_AIRLLM.md`, `RE_OLLAMA.md`.
- [x] `docs/REPRODUCE.md` — from-clone reproduction.
- [x] Committed showcase: `results/reports/demo/` (comparison report) and `results/paging/` (paging visualization).
- [x] Tests (`uv run pytest -m "not slow"` green) + CI workflow.

## Self-assessment vs the L08 §9 brief
| §9 requirement | Status |
|---|---|
| Pull a model from Hugging Face | `localforge pull` / `models/acquire.py` |
| Run via Ollama (OpenAI-compatible API) | `ollama` backend (`/v1`, stdlib) |
| Measure RAM/GPU baseline | `localforge baseline` |
| Run AirLLM on CPU + measure latency | `airllm` backend + profiler (engine dep-broken upstream → skip-with-reason, documented in RE_AIRLLM.md; real per-block trace captured via instrumentation) |
| Compare load/GPU/CPU/AirLLM across sizes | `localforge compare` → report |
| SafeTensors/GGUF, prefill/decode, VRAM | format detection; profiler prefill/decode split; VRAM probe |
| LoRA/QLoRA/OLoRA fine-tuning | `localforge finetune` (real CPU LoRA validated) |
| Tooling: uv, HF token | `uv`-managed; `.env` HF token |
| Originality / "מעוף" | the Paging Visualizer + empirical RE bridging OS theory and LLM inference |

## Steps you (the student) must do
1. **Push to GitHub** and **add the lecturer as a collaborator**: `rmisegal@gmail.com` (and confirm the repo is shared per the rules).
   ```bash
   git remote add origin <your-github-url>
   git push -u origin main
   ```
2. **Fill the locked Word template** (`moamteam-ex05.docx`, from the course "Intro" module — same template family as your `moamteam-ex04.docx`). Do **not** change fields or layout; only fill in: group code `moamteam`, members, the GitHub URL, and the **self-grade**. Save as **`moamteam-ex05.pdf`** and submit on Moodle. No extra text/attachments.
3. **Self-grade honestly** (suggested **89**). Per course guidance the grading rigor scales with the self-grade; a fair number yields a more accurate result. See the rubric self-assessment below.
4. Each group member submits separately on Moodle (the time window is per-person).

## Rubric self-assessment (software guidelines V3)
| Rubric criterion | Status |
|---|---|
| §2 README (root) + `docs/` + PRD/PLAN/TODO | ✅ + SPEC/IMPL/TASKS/DECISIONS/RE/REPRODUCE |
| §3.2 ≤150 lines per file | ✅ all 56 source files comply |
| §3 docstrings + type hints | ✅ throughout; mypy strict clean |
| §4.2 OOP / SOLID | ✅ Protocol, Strategy+Factory, Observer, Repository, Null-Object, Context-Manager, Builder |
| §5.1 API Gatekeeper | ✅ `core/gatekeeper.py` (rate limit + retry) wired into HF + Ollama calls |
| §6.2 ≥85% coverage | ✅ **88%** overall offline (core/config/reporting 95%) |
| §7.1 Ruff linter | ✅ + mypy strict |
| §8 uv + Git | ✅ uv-locked; clean per-phase commits |
| §14.2 `__init__.py` | ✅ every package |
| §15 Multithreading | ✅ profiler sampler + generation streamer threads |
| Originality ("מעוף") | ✅ Paging Visualizer + empirically-verified AirLLM RE |

**Why 89, not higher:** AirLLM's headline CPU run is *modeled* (its memory dynamics), not executed — verified to be GPU-bound after a five-blocker investigation (RE_AIRLLM §6). The real per-block execution trace and all other §9 items run for real. Honest, complete, original — but not a flawless 95+.

## Notes for the grader
- Everything runs **CPU-only**; no GPU/Ollama/AirLLM required to reproduce the core report and visualizer (replay/synthetic modes cover the absent engines).
- The HF token lives only in a git-ignored `.env`; it is never committed.
- `airllm` is genuinely broken upstream against current `optimum`; this is documented as a finding and handled gracefully, not hidden.
