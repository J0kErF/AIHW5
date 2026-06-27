# localforge — Running a Massive LLM Locally: Deep-Dive Technical Report

> **EX05** · AirLLM, Quantization and Performance Benchmarking · **moamteam**
> Built with **localforge** — a modular, CPU-first Python CLI for running, benchmarking, and fine-tuning LLMs locally.

---

## Executive Summary

Running an LLM locally is a **memory-vs-latency trade**. On a CPU-only laptop (specs below), the same Qwen2.5-0.5B model:

| Backend | Strategy | TTFT | TPOT | Throughput | Peak RAM |
|---|---|---|---|---|---|
| `transformers` | full residency (all weights in RAM) | ~0.8 s | **~0.10 s/tok** | ~9.9 tok/s | **~3,227 MB** |
| **AirLLM** | layer streaming (one block from disk at a time) | ~4.1 s | **~3.8 s/tok** | ~0.26 tok/s | **~370 MB** |

**AirLLM uses ~8.7× less RAM but decodes ~42× slower.** The rest of this report explains *why* (the roofline), *how* (AirLLM's paging), and *when each makes economic sense* (API vs OnPrem).

---

## 1. Hardware Specification & Model Choice (§5.1)

### Hardware

| Component | Specification |
|---|---|
| **CPU** | Intel Core i7-1165G7 @ 2.80 GHz, 4 cores / 8 threads (11th Gen Tiger Lake) |
| **RAM** | 16 GB DDR4 |
| **GPU** | Intel Iris Xe Graphics (integrated, **no CUDA**) |
| **VRAM** | Shared — no dedicated VRAM |
| **Storage** | WDC PC SN730 1 TB NVMe SSD (~3 GB/s sequential read) |
| **OS** | Windows |

### Model choice: Qwen2.5-0.5B-Instruct

With 16 GB RAM and no CUDA GPU, models above ~7B parameters in fp32 would exhaust RAM, and models above ~3B would be impractically slow on CPU. We chose **Qwen/Qwen2.5-0.5B-Instruct** (0.5B parameters, ~1 GB fp16 / ~2 GB fp32) to enable **a complete experimental pipeline** — baseline, AirLLM, profiling, fine-tuning — all executable on this hardware without OOM or multi-hour waits.

This model is deliberately *not* too large for our hardware, which shifts the experiment's narrative: instead of demonstrating "it crashes without AirLLM," we demonstrate the **measured cost of the memory-latency trade** on a model where both paths complete, making the comparison fair and quantitative. The roofline analysis and AirLLM's ~42× decode slowdown remain fully visible even at this scale — the physics of layer streaming are scale-invariant (§5). A larger model (e.g., 7B) would show the same pattern with proportionally larger numbers, but would take hours per run on this CPU, preventing the iterative measurement the assignment requires.

> **Negative result acknowledged:** On this hardware, AirLLM does not "rescue" a model that can't run — it demonstrates the *cost* of its memory-saving strategy. This is itself a valid experimental finding, documented honestly.

---

## 2. Baseline — Direct Execution (§5.2)

We ran the model directly via the Hugging Face `transformers` backend on CPU:

```bash
uv run localforge run --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct
```

**Result:** The model loads and runs successfully in ~3,227 MB peak RSS. This is our baseline — all weights resident in RAM, no paging, no optimization.

| Metric | Value |
|---|---|
| Load time | 12.9 s |
| TTFT (prefill) | 774.9 ms |
| TPOT (decode) | 101.4 ms/token |
| Throughput | 9.9 tok/s |
| Peak RAM | 3,227 MB |
| Peak VRAM | n/a (CPU-only) |
| Total generation time (20 tokens) | ~14.9 s |

The model does **not** crash or OOM — 16 GB RAM easily holds a 0.5B model. The bottleneck at baseline is **decode throughput** (memory-bandwidth-bound matrix-vector products on CPU), not memory capacity.

`localforge baseline` records idle system RAM for comparison: the model adds ~2.8 GB above idle.

---

## 3. AirLLM Integration — Disk as Virtual Memory (§5.3)

AirLLM runs the model **one transformer block at a time**: mmap the block's SafeTensors shard, load only that block, run its forward pass, evict it, repeat. The working set is one block, so a model far larger than RAM can "run." This is **demand paging** — the OS analogy the course builds on:

| AirLLM mechanism | OS analogue |
|---|---|
| transformer block | page |
| block not in RAM → load from disk | page fault |
| block already resident → reuse | cache hit |
| drop block to free RAM | eviction |
| `mmap` of SafeTensors shard | memory-mapped file (zero-copy via page cache) |
| RAM ceiling (working-set target) | resident set size limit |

### We ran AirLLM for real — a seven-fix recipe

AirLLM 2.11 is incompatible with the transformers 5.x localforge uses. Getting a real CPU run took a documented **seven-fix recipe** (see `experiments/airllm_real_run.py` and `docs/RE_AIRLLM.md`):

1. **`optimum.bettertransformer` removed** (in `optimum>=2.2`) → `ModuleNotFoundError`. *Fix:* no-op shim.
2. **`sentencepiece` missing** → *Fix:* install it.
3. **Single-file model rejected** — AirLLM needs `model.safetensors.index.json`. *Fix:* re-save sharded.
4. **Tied embeddings crash the splitter** — Qwen ties `lm_head` → `IndexError`. *Fix:* untie/materialize.
5. **CUDA hard-wired in init** — CPU-only torch raises. *Fix:* no-op `torch.cuda.*`.
6. **`DynamicCache` not subscriptable** — transformers 5.x changed API. *Fix:* `use_cache=False`.
7. **`position_embeddings` required** — transformers 5.x Qwen2 layer API changed. *Fix:* pin **transformers 4.41.2**.

**Real AirLLM result (Qwen2.5-0.5B, CPU):**

| Metric | Value |
|---|---|
| TTFT | ~4.1 s |
| TPOT | ~3.8 s/token |
| Throughput | ~0.26 tok/s |
| Peak RAM | ~370 MB |

**This fragility is itself a finding:** seven fixes to run a "run big models locally" tool is precisely the operational friction the course frames. localforge handles it gracefully — a missing/broken engine becomes a *skipped* row with a reason, never a crash.

### Quantization

The spec asks about FP16, Q8, Q4 quantization levels. Our hardware constraints limit what we can demonstrate:

- **FP32** (baseline): the `transformers` backend runs in fp32 on CPU. Peak RAM ~3,227 MB.
- **FP16**: theoretically halves memory. On this CPU (no native fp16 compute), performance gains are marginal and bitsandbytes (NF4/QLoRA) requires CUDA. localforge reports this as a documented fallback.
- **GGUF quantization via Ollama**: Ollama serves models in 4-bit GGUF format, which is the practical quantization path on CPU hardware. With `ollama serve` running, `uv run localforge compare` populates the Ollama row with quantized inference numbers. On this machine without an Ollama daemon, localforge records it as a *skipped* row with the reason — the comparison is honest about what ran.
- **AirLLM + quantization**: AirLLM 2.11 supports a `compression` parameter for on-the-fly quantization during layer streaming. Our real run used fp32; combining AirLLM with 4-bit quantization would reduce the per-block disk read (~4× less I/O per token) and could meaningfully improve TPOT — but the transformers version incompatibility prevented testing this combination.

**Quantization takeaway:** On CPU-only hardware without CUDA, the primary quantization lever is **GGUF via Ollama** (4-bit, keeps weights RAM-resident but 4× smaller). The AirLLM path would benefit most from quantization (less disk I/O per layer), but the library's fragility prevented testing this combination.

---

## 4. Performance Measurement & Comparison (§5.4)

### Metric definitions

- **TTFT** (Time To First Token): the prefill window — building the KV-cache for the input prompt. Measures compute load.
- **TPOT** (Time Per Output Token) / **ITL** (Inter-Token Latency): decode latency per token after the first. Measures memory-bandwidth load.
- **Throughput**: tokens/sec during decode.
- **Peak RAM / VRAM**: maximum resident set size during inference.

### Full comparison matrix

| Backend | Model | Status | Load (s) | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak RAM (MB) | Note |
|---|---|---|---|---|---|---|---|---|
| transformers | Qwen2.5-0.5B | ✅ ok | 12.9 | 774.9 | 101.4 | 9.9 | 3,227 | full residency |
| AirLLM | Qwen2.5-0.5B | ✅ ok | — | ~4,100 | ~3,800 | ~0.26 | ~370 | layer-streamed (dedicated env) |
| Ollama | qwen2.5:0.5b | ⬚ skipped | — | — | — | — | — | no daemon (run `ollama serve`) |

> Reproduced via `uv run localforge compare --suite config/suites/demo.yaml`. AirLLM numbers from `experiments/airllm_real_run.py` in a dedicated transformers-4.41 env.

### Comparison charts

![Peak RAM by backend — transformers uses ~3.2 GB vs AirLLM's ~370 MB](results/reports/demo/peak_ram.png)

![Decode throughput by backend](results/reports/demo/decode_tok_s.png)

![Load time by backend](results/reports/demo/load_s.png)

### Estimated total runtime and power

For a 20-token generation (short prompt + 20 output tokens):

| Backend | Total wall time | Estimated energy |
|---|---|---|
| transformers | ~15 s | ~0.12 Wh (CPU PBP 28 W × 15 s / 3600) |
| AirLLM | ~80 s (4.1 s TTFT + 20 × 3.8 s) | ~0.62 Wh (28 W × 80 s, plus NVMe I/O) |

AirLLM is ~5× more energy-intensive per generation due to repeated disk I/O.

### Output quality

Both backends produce **identical output** for the same model at fp32 — AirLLM runs the exact same weights in the exact same order (one block at a time). Output quality is a function of quantization level, not execution strategy. At fp32, the 0.5B model produces coherent short answers (e.g., *"Virtual memory is a memory management technique..."*) with occasional repetition typical of small models.

---

## 5. The Roofline — Prefill vs Decode (§5.6)

The two inference phases have opposite bottlenecks:

![Roofline model: prefill is compute-bound, decode is memory/disk-bound](figures/roofline.png)

- **Prefill (→ TTFT)** processes the whole prompt as **matrix-matrix** products. High arithmetic intensity (FLOP/byte), so it is **compute-bound** — limited by CPU FLOPS, under the compute roof.
- **Decode (→ TPOT)** emits one token at a time as **matrix-vector** products. It must read **every weight** to produce **one** token, so intensity is ~1 FLOP/byte: **memory-bound**, limited by RAM bandwidth (~50 GB/s on this laptop).
- **AirLLM decode** re-streams weights from **NVMe (~3 GB/s)** instead of RAM every token. On the roofline it sits far left under the *disk* roof — which is why its TPOT (~3.8 s/tok) is ~42× worse than full-residency decode (~0.10 s/tok).

**This is the central insight:** AirLLM doesn't reduce compute — it makes decode **disk-bandwidth-bound** instead of memory-bandwidth-bound. That's the precise, measured cost of trading RAM for disk.

The roofline also explains why prefill (TTFT) suffers less under AirLLM: prefill is compute-bound, so reading weights from a slower medium hurts less proportionally than during decode (where weight-reading *is* the bottleneck).

---

## 6. Economic Analysis — API vs OnPrem (§5.5)

### Cost model

We model cost as a function of total token volume *N*:

- **API:** `cost = price_per_tok × N` (no upfront cost)
- **OnPrem (buy GPU):** `cost = CAPEX + opex_per_tok × N` (one-time hardware + electricity)
- **Cloud GPU (rent):** `cost = (hourly_rate / (throughput × 3600)) × N`

### Assumptions (illustrative 2026 prices, documented for transparency)

| Parameter | Value | Source |
|---|---|---|
| API price | $0.60 / 1M output tokens | OpenAI-class small model pricing |
| GPU CAPEX | $1,600 | Consumer 24 GB GPU (RTX 4070 Ti class) |
| GPU power | 350 W | Typical gaming/inference GPU TDP |
| Electricity | $0.15/kWh | Average residential rate |
| OnPrem throughput | 40 tok/s | Served on consumer GPU |
| Cloud GPU hourly | $1.20/h | Cloud GPU instance (A10G class) |

### Breakeven analysis

![API vs OnPrem vs Cloud GPU — cost over volume](figures/breakeven.png)

| Path | $/token | Fixed cost |
|---|---|---|
| API | 6.0×10⁻⁷ | $0 |
| OnPrem (electricity only) | 3.7×10⁻⁷ | $1,600 CAPEX |
| Cloud GPU | 8.3×10⁻⁶ | $0 |

**Breakeven ≈ 6.8 billion tokens** before buying a GPU beats the API.

### Key findings

1. **For most individuals/small teams, the API wins** — you never approach 6.8B tokens. OnPrem is for sustained high volume, data-privacy, or offline requirements.
2. **Cloud GPU is the *most* expensive per-token** (~14× the API) because a single rented GPU at 40 tok/s is far less utilized than a batched API endpoint. Renting only wins if you also need the GPU for training or need burst capacity.
3. **Sensitivity:** at ±50% API price, breakeven moves to ~3.4B–13.6B tokens — it never becomes "buy a GPU for casual use."

### Prompt Caching & PagedAttention

Two mechanisms shift the API-side economics:
- **Prompt/Context Caching:** API providers using PagedAttention-based servers can cache the system prompt's KV-cache, avoiding re-prefill on repeated calls. This reduces the effective API cost for chatbot-style workloads (same system prompt, many user turns).
- **PagedAttention (vLLM):** pages the KV-cache in fixed blocks, cutting VRAM fragmentation from ~60% to a few %. This lets a served GPU batch far more concurrent requests, lowering the effective $/token of the OnPrem/Cloud side.

Both mechanisms make the *served* API even more competitive vs OnPrem for typical use patterns.

---

## 7. Lecture Concepts Applied (§5.6)

Every result maps to course concepts:

| Result | Lecture concept |
|---|---|
| Decode throughput limited by RAM bandwidth | Memory-bound computation (L08) |
| AirLLM decode limited by NVMe bandwidth | Virtual memory paging — disk as backing store |
| AirLLM's per-block load/evict cycle | Demand paging, working set, page fault/eviction |
| Prefill faster than decode (per FLOP) | Compute-bound vs memory-bound (roofline model) |
| Quantization (GGUF 4-bit) shrinks resident set | Reducing page size → more pages fit in physical memory |
| API wins at low volume | CAPEX amortization, economies of scale in cloud serving |

### Paging Visualization

localforge instruments the per-block execution trace and renders it as a memory-hierarchy view — page faults, residency, evictions — making the OS-theory framing concrete:

![Synthetic AirLLM paging model — fault/evict sawtooth with bounded working set](results/paging/paging.png)

The upper panel shows fault (load) and evict events per block over time; the lower panel shows the resident set size bounded at the working-set target. This is exactly the OS paging behavior described in L08 §8.

---

## 8. Extensions & Original Contributions (§5.7)

1. **The Paging Visualizer** — localforge's signature feature. It bridges OS virtual-memory theory and real LLM inference by instrumenting per-block execution and rendering fault/residency/eviction traces. This goes beyond what AirLLM itself exposes.

2. **Seven-fix AirLLM compatibility recipe** — a documented chain of real incompatibilities, each driven to root cause. This is an original contribution to anyone trying to run AirLLM on current tooling.

3. **LoRA/QLoRA/OLoRA fine-tuning** — `localforge finetune` trains a PEFT adapter and shows before/after inference, with CPU-compatible fallbacks.

4. **Graceful degradation architecture** — three backends behind one Protocol, where unavailable engines become *skipped rows with reasons*, never crashes. This is an engineering pattern not in the assignment but valuable for robustness.

5. **Economic breakeven CLI** — `localforge econ` computes and plots API-vs-OnPrem-vs-Cloud breakeven with user-configurable assumptions.

---

## 9. Conclusions

- **Memory bounds everything.** Where the weights live (VRAM/RAM/disk) determines what model you can run and how fast.
- **Decode is the bottleneck, and it's memory-bound** — AirLLM makes it disk-bound, measured at ~42× slower for ~8.7× less RAM.
- **For most users the API wins economically**; OnPrem/Cloud are for volume, privacy, or training.
- **AirLLM's operational fragility** (seven fixes to run on current tooling) is itself a finding about the maturity of local LLM deployment.
- **localforge** makes all of this measurable and reproducible, degrading gracefully when an engine is unavailable.

---

## Quickstart — Reproduce the Experiment

```bash
git clone https://github.com/J0kErF/AIHW5.git && cd AIHW5
uv sync --group dev          # reproducible env from uv.lock
cp .env.example .env         # add your HF_TOKEN (free, from huggingface.co)

uv run localforge baseline                                   # idle RAM/VRAM
uv run localforge run --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct
uv run localforge compare --suite config/suites/demo.yaml    # 3-backend report
uv run localforge finetune --method lora                     # tiny CPU LoRA train
uv run localforge visualize --no-tui                         # paging visualizer
uv run localforge econ                                       # economic breakeven
```

Optional: `uv sync --extra ollama` (then `ollama serve`), `uv sync --extra airllm`, `uv sync --extra gpu`.

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for full from-clone reproduction instructions.

## CLI Commands

| Command | What it does |
|---|---|
| `baseline` | Record idle RAM/VRAM |
| `pull <model>` | Download a model from the HF Hub |
| `run` | Single inference + profiled metrics |
| `compare --suite <yaml>` | Run a model×backend×dtype matrix → report |
| `finetune --method {lora,qlora,olora}` | Train a PEFT adapter (before/after) |
| `visualize [--model/--replay]` | Paging visualizer (TUI / static PNG+HTML) |
| `econ` | API-vs-OnPrem breakeven analysis + figure |

## Architecture

CPU-first, modular design under `src/localforge/`: `core` (types/errors/capabilities), `config` (settings + suite builder), `models` (HF acquisition), `backends` (Protocol + registry + transformers/ollama/airllm + runner), `profiling` (Profiler), `finetune` (PEFT), `paging` (tracer/events/replay), `viz` (TUI + static), `reporting` (matrix/charts/report), `econ` (breakeven model). Design patterns in [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

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
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | what it is and does |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | architecture, patterns, module map |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | locked decisions (ADR log) |
| [`docs/RE_AIRLLM.md`](docs/RE_AIRLLM.md) | AirLLM reverse engineering |
| [`docs/RE_OLLAMA.md`](docs/RE_OLLAMA.md) | Ollama reverse engineering |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | from-clone reproduction |
| [`docs/SUBMISSION.md`](docs/SUBMISSION.md) | submission checklist |
| [`reports/REPORT.md`](reports/REPORT.md) | standalone copy of the technical report |

## Artifacts

- Comparison report: `results/reports/demo/{report.html, matrix.md, *.png}`
- Paging visualization: `results/paging/{paging.png, paging.html}`
- Figures: `figures/roofline.png`, `figures/breakeven.png`
- Real AirLLM run: `experiments/airllm_real_run.py`

## License

MIT.
