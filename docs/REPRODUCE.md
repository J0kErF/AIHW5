# Reproducing localforge

Every result in this repo regenerates from a clean clone with `uv` and a free Hugging Face token. CPU-only; no GPU required.

## 0. Prerequisites
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- A free Hugging Face token (https://huggingface.co/settings/tokens). Qwen2.5 models are public, but a token avoids download rate limits.

## 1. Environment
```bash
git clone <repo-url> localforge && cd localforge
uv sync --group dev          # exact env from uv.lock
cp .env.example .env         # set HF_TOKEN=...
```

## 2. Offline checks (no downloads, no GPU)
```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not slow"          # full offline suite, green
```

## 3. The comparison report (downloads ~1 GB once)
```bash
uv run localforge baseline
uv run localforge compare --suite config/suites/demo.yaml
# -> results/reports/demo/{report.html, matrix.md, matrix.json, *.png}
```
On a CPU-only host the `transformers` backend runs; `ollama`/`airllm` appear as explained skipped rows. To make the three-way comparison real:
```bash
uv sync --extra ollama && ollama serve &      # separate terminal
ollama pull qwen2.5:0.5b
uv run localforge compare --suite config/suites/demo.yaml
```

## 4. Fine-tuning (CPU)
```bash
uv run localforge finetune --method lora --steps 12
# trains a LoRA adapter on data/finetune/tiny_sft.jsonl; prints trainable % and before/after.
# try also --method olora (QR-orthonormal init) and --method qlora (CUDA -> documented fallback on CPU).
```

## 5. The paging visualizer
```bash
# real per-block execution trace from an actual model:
uv run localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct --no-tui
# modeled AirLLM paging (no model/airllm needed):
uv run localforge visualize --no-tui
# replay a recorded stream:
uv run localforge visualize --replay tests/data/sample_paging.jsonl --no-tui
# -> results/paging/{paging.png, paging.html, paging.jsonl}
# drop --no-tui for the live Textual dashboard.
```

## 6. Real-model / engine-gated tests
```bash
uv run pytest -m slow            # real transformers run + real CPU LoRA train (downloads a model)
uv run pytest -m needs_ollama    # requires `ollama serve` + the model pulled
uv run pytest -m needs_airllm    # requires `uv sync --extra airllm`
```

## Notes
- **CPU-first:** nothing requires a GPU. CUDA paths (fp16/NF4/QLoRA) are exercised as code and reported as documented fallbacks on CPU.
- **AirLLM:** `airllm==2.11` currently fails to import against `optimum>=2.2` (removed `bettertransformer`); localforge reports it as a skipped backend with the exact reason. See `docs/RE_AIRLLM.md`.
- Determinism: greedy runs are seeded; the same host reproduces the same text.
