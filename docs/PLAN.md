# localforge — Plan (PLAN)

> Submission-required artifact. Architecture + phased delivery. Full detail in `IMPLEMENTATION.md` and `TASKS.md`.

## Architecture (one process, layer-based modules)
```
cli → {models, backends, finetune, reporting, viz, profiling}
backends → {core, profiling, paging(airllm)}      finetune → {core, models}
paging → {core}      viz → {core, paging, reporting}      reporting → {core}
everything → core   (no cycles; optional engines isolated to single files)
```
Patterns: Strategy+Registry Factory (backends), Context Manager (profiler), Observer (paging events), Repository (result store), Builder (suite→specs), Null Object (unavailable backend). Rationale per IMPLEMENTATION.md §2.

## Module map (SPEC feature → module)
| Feature | Module |
|---|---|
| HF pull / formats | `models/` |
| 3 backends + runner | `backends/` |
| measurement | `profiling/` |
| comparison report | `reporting/` |
| PEFT fine-tune | `finetune/` |
| paging capture/replay | `paging/` |
| TUI + static viz | `viz/` |
| commands | `cli/` |
| types/errors/caps | `core/`, `config/` |

## Tech stack
Python 3.11+ / uv; Typer + Rich/Textual; PyTorch + transformers + huggingface_hub + peft + accelerate; bitsandbytes/airllm/ollama as **optional extras**; psutil; pydantic(-settings); pandas + matplotlib + Jinja2; pytest + ruff + mypy. (IMPLEMENTATION.md §1.)

## Deliverable shape
**Hybrid** (`REVERSE_ENGINEERING.md`): a reverse-engineering analysis of AirLLM (primary) + Ollama, **plus** `localforge` as the original instrumentation/extension layer that empirically validates the RE via the Paging Visualizer. Adds RE tasks T9a (AirLLM static+dynamic RE → `RE_AIRLLM.md` + trace) and T15a (predicted-vs-measured cross-check); `RE_OLLAMA.md` alongside T8. The `~10k LOC / ~70 files` figure is a **top-grade ambition, not a hard requirement** (it was the HW4 bring-your-own-repo threshold) — depth and reproducibility win over line count.

## Phased delivery (18 tasks + 2 RE tasks, ~10–13 days)
1. **Foundation** (T1–T3): scaffold, core types, capabilities/settings.
2. **Profiling & backend core** (T4–T5): profiler, Protocol/registry/runner, FakeBackend.
3. **Models & real backends** (T6–T9): HF pull, transformers, Ollama, AirLLM. → MVP.
4. **Reporting & CLI** (T10–T12): store, matrix/charts/report, CLI.
5. **Fine-tuning & visualizer** (T13–T16): PEFT + OLoRA, trainer, paging tracer/replay, TUI+static. → Originality.
6. **Quality, docs, release** (T17–T18): coverage/CI, README/results/reproducibility/submission.

## Reproducibility plan
`uv sync` from `uv.lock`; seeded greedy runs; demo uses Qwen2.5-0.5B for CPU speed; committed `results/reports/demo/*`; CI runs the CPU smoke `compare`. Optional engines via `uv sync --extra {gpu,ollama,airllm}`.

## Testing plan
FakeBackend + recorded paging fixture make the full run→compare→report→visualize pipeline testable with zero downloads and no GPU. CUDA/Ollama/AirLLM tests guarded by `skipif`. Target ≥80% coverage on core/config/reporting.

## Submission
GitHub repo (group code `moamteam`), instructor `rmisegal@gmail.com` added as collaborator; repo contains PRD/PLAN/TODO + root README; deliverable `moamteam-ex05.pdf` from the locked Word template (no extra text); honest self-grade.
