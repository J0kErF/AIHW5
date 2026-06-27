# HW5 Submission Checklist

> Maps localforge to the course submission rules (`submit.txt`, software guidelines V3). Group code: **moamteam**. Assignment: local LLM inference & fine-tuning (L08).

## What's in the repo (required artifacts)
- [x] Root `README.md` with setup, results, architecture, and **hardware spec** (CPU-only, 4 cores, 16 GB RAM, no GPU).
- [x] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` (the mandated PRD/PLAN/TODO markdown).
- [x] Full design set: `SPECIFICATION.md`, `IMPLEMENTATION.md`, `TASKS.md`, `DECISIONS.md`, `PROMPT.md`, `BRANDING.md`.
- [x] Reverse-engineering write-ups: `REVERSE_ENGINEERING.md`, `RE_AIRLLM.md`, `RE_OLLAMA.md`.
- [x] `docs/REPRODUCE.md` — from-clone reproduction.
- [x] Deep-dive report: `reports/REPORT.md` — the README *is* the report, with embedded visuals in `figures/` (roofline, breakeven).
- [x] Committed showcase: `results/reports/demo/` (comparison report) and `results/paging/` (paging visualization).
- [x] Real AirLLM CPU run: `experiments/airllm_real_run.py` (7-fix recipe, transformers 4.41 env).
- [x] Figures: `figures/roofline.png` (§5.6), `figures/breakeven.png` (§5.5).
- [x] Tests (`uv run pytest -m "not slow"` green) + CI workflow.

## Self-assessment vs the L08 §9 brief
| §9 requirement | Status |
|---|---|
| Pull a model from Hugging Face | ✅ `localforge pull` / `models/acquire.py` |
| Run via Ollama (OpenAI-compatible API) | ✅ `ollama` backend (`/v1`, stdlib); skips with reason when daemon absent |
| Measure RAM/GPU baseline | ✅ `localforge baseline` |
| Run AirLLM on CPU + measure latency | ✅ Real run achieved via `experiments/airllm_real_run.py` (7-fix recipe, pinned transformers 4.41). TTFT ~4.1 s, TPOT ~3.8 s/tok, peak RAM ~370 MB |
| Compare load/GPU/CPU/AirLLM across sizes | ✅ `localforge compare` → `results/reports/demo/` (matrix + charts) |
| TTFT / TPOT / ITL metrics (§5.4) | ✅ Properly labeled in profiler and report: TTFT = Time To First Token, TPOT = Time Per Output Token, ITL = Inter-Token Latency |
| Roofline analysis (§5.6) | ✅ `figures/roofline.png` — prefill compute-bound, decode memory-bound, AirLLM disk-bound |
| Economic analysis (§5.5) | ✅ `localforge econ` → `figures/breakeven.png`; breakeven ≈ 6.8B tokens; API wins for most users |
| SafeTensors/GGUF, prefill/decode, VRAM | ✅ Format detection; profiler prefill/decode split; VRAM probe (null on CPU) |
| LoRA/QLoRA/OLoRA fine-tuning | ✅ `localforge finetune` (real CPU LoRA validated; QLoRA/OLoRA documented as CUDA-only fallback) |
| Tooling: uv, HF token | ✅ `uv`-managed; `.env` HF token (git-ignored) |
| Originality / "מעוף" | ✅ Paging Visualizer + empirically-verified AirLLM RE + OS-theory bridge |

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
| §5.4 TTFT / TPOT / ITL labeling | ✅ Properly named in profiler output and `reports/REPORT.md` |
| §5.5 Economic analysis | ✅ API-vs-OnPrem breakeven (`figures/breakeven.png`, ≈ 6.8B tokens); caching & PagedAttention discussed |
| §5.6 Roofline analysis | ✅ `figures/roofline.png` — compute/memory/disk ceilings with measured data points |
| §6.2 ≥85% coverage | ✅ **88%** overall offline (core/config/reporting 95%) |
| §7/§8 Report with embedded visuals | ✅ `reports/REPORT.md` *is* the report; figures in `figures/` (roofline, breakeven); README links everything |
| §7.1 Ruff linter | ✅ + mypy strict |
| §8 uv + Git | ✅ uv-locked; clean per-phase commits |
| §9 Directory structure | ✅ `src/`, `docs/`, `tests/`, `config/`, `results/`, `reports/`, `figures/`, `experiments/`, `data/` |
| §14.2 `__init__.py` | ✅ every package |
| §15 Multithreading | ◑ used where it matters (profiler RSS sampler + generation streamer threads), not as a standalone concurrency showcase |
| Originality ("מעוף") | ✅ Paging Visualizer + empirically-verified AirLLM RE |

**Why 89, not higher (disclosed honestly):**
- **The model (0.5B) is deliberately small.** On a CPU-only laptop with 16 GB RAM, Qwen2.5-0.5B doesn't OOM even under `transformers` full-residency. This means the experiment is more "AirLLM cost-benefit analysis" than "AirLLM rescues a model that can't otherwise run." The memory savings (~8.7×) and latency penalty (~42×) are real and measured, but the existential need for layer streaming only kicks in at 7B+ — which we can't practically run on this hardware.
- **Quantization comparison is conceptual, not measured across FP16/Q8/Q4 levels.** bitsandbytes requires CUDA (no GPU), and Ollama requires a running daemon we don't bundle. The quantization discussion in the report is accurate and well-sourced, but it's not backed by our own FP16→Q8→Q4 sweep — it's documented as a known gap with fallback reasoning, not faked numbers.
- **AirLLM required a separate dedicated env** (pinned to transformers 4.41) rather than running within the main project's environment. The 7-fix recipe in `experiments/airllm_real_run.py` is reproducible and documented, but it's not integrated into the `localforge` CLI — the CLI correctly skips AirLLM with a reason when the engine is broken upstream.
- 88% overall coverage is concentrated in core/config/reporting (95%); the originality-feature modules (`trainer` 34%, `transformers_backend` 65%, `visualize_cmd` 66%) are exercised mainly by the **slow/real-model tests** (run `uv run pytest -m slow`), not the offline suite — a deliberate trade-off so CI stays download-free.
- These are honest limitations that prevent a 95+, but the core analysis, engineering, economic work, roofline modeling, and reverse-engineering are solid and original.

## Notes for the grader
- Everything runs **CPU-only**; no GPU/Ollama/AirLLM required to reproduce the core report and visualizer (replay/synthetic modes cover the absent engines).
- **Real AirLLM run is available** in `experiments/airllm_real_run.py` using a documented 7-fix recipe (separate transformers-4.41 venv). It produces real TTFT/TPOT/peak-RAM numbers used in the report.
- **Hardware:** CPU-only laptop, 4 cores, 16 GB RAM, NVMe SSD, no discrete GPU. Documented in `README.md` and `reports/REPORT.md §1`.
- The HF token lives only in a git-ignored `.env`; it is never committed.
- `airllm` is genuinely broken upstream against `transformers ≥5.x` and current `optimum`; this is documented as a finding and handled gracefully, not hidden.
- Directory structure: `src/` (code), `docs/` (design + RE), `tests/`, `config/` (suites), `results/` (runs + reports + paging), `reports/` (REPORT.md), `figures/` (roofline + breakeven), `experiments/` (real AirLLM run), `data/` (datasets).
