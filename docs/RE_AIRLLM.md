# Reverse-Engineering: AirLLM

> The primary RE deliverable (docs/REVERSE_ENGINEERING.md). How AirLLM runs a model larger than RAM by streaming it layer-by-layer, mapped to OS virtual-memory theory, and validated against a real per-block execution trace captured by localforge. Claims are tagged *(measured)* — from our instrumentation — or *(architectural)* — from AirLLM's design.

## 1. The idea
AirLLM (`airllm`, Apache-2.0) runs a decoder-only LLM **one transformer block at a time**. Instead of holding all N blocks in RAM, it:
1. memory-maps the block's SafeTensors shard,
2. loads only that block's weights, runs the forward pass for the current position,
3. releases the block and moves to the next.

A 70B model whose weights are ~140 GB can thus "run" in a few GB of RAM: the **working set** is one block, not the whole model. This is exactly demand paging — the OS analogy the course builds on (L08 §8).

## 2. OS virtual-memory mapping *(architectural)*
| AirLLM mechanism | OS analogue |
|---|---|
| transformer block | page |
| block not in RAM → load from disk | page fault |
| block already resident → reuse | cache hit |
| drop block to free RAM | eviction |
| `mmap` of a SafeTensors shard | memory-mapped file, zero-copy via the page cache |
| RAM ceiling (working-set target) | resident set size limit |
| disk → page cache → RAM → cache → registers | the memory hierarchy |

localforge's visualizer renders precisely these events (`PageAction ∈ {fault, hit, evict}`, `PageSource ∈ {mmap, disk}`) across the memory hierarchy (`src/localforge/viz`).

## 3. Instrumentation methodology
localforge captures paging behavior two ways (`src/localforge/paging`):

1. **Real per-block hooks** (`instrument_layers`): a PyTorch `forward_pre_hook` on every decoder block emits one event as the block executes. Because AirLLM loads a block immediately before running it, block-execution order *is* the fault sequence. The identical hook runs on the `transformers` backend, giving a measured baseline.
2. **Synthetic pager** (`synthesize_airllm_trace`): an explicit model of AirLLM's bounded working set — fault a block in, evict the oldest when the RAM ceiling is hit — used to predict AirLLM's memory dynamics and to demonstrate the visualizer where `airllm` cannot run (see §6).

## 4. Empirical result — measured execution *(measured)*
Captured with `localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 6` (artifacts in `results/paging/`):

- **24 transformer blocks**, **144 block-executions** over **769 ms** (24 blocks × 6 forward passes: 1 prefill + 5 decode).
- **Peak resident = 24/24 blocks, 0 evictions.** The `transformers` backend keeps every block in RAM and re-runs them each token — *full residency*, no paging.

This is the baseline: same compute AirLLM does, but with the whole model resident.

## 5. Predicted vs measured — the contrast *(measured ⊕ architectural)* — T15a
Synthetic AirLLM model for a 24-block model with a working-set target of ~8 blocks (`results/.../sample_paging` style), 2 passes:

| Metric | transformers (measured) | AirLLM model (predicted) |
|---|---|---|
| block-executions | 144 (6 passes) | 48 (2 passes) |
| peak resident blocks | **24 / 24** | **8 / 24** |
| evictions | **0** | **40** |
| bytes streamed from disk | ~0 (resident) | ~16.8 GB |
| memory footprint | full model | one working set |
| cost traded | RAM | disk I/O per token |

**The finding:** the two backends do *identical* per-block compute in the *same order*; they differ only in the memory policy. AirLLM converts a RAM constraint into repeated disk I/O — it re-faults every block on every decode step, which is why its decode latency is dominated by I/O, not compute (L08 §8.3). The visualizer makes this visible: a flat "24 resident" line for transformers vs a sawtooth bounded at 8 for AirLLM.

## 6. We ran it for real — a verified seven-fix recipe + real numbers

Getting AirLLM to actually run was a chain of **seven** real incompatibilities, each driven to root cause and fixed (script: `experiments/airllm_real_run.py`):

1. **`optimum.bettertransformer` removed** (in `optimum>=2.2`) → `ModuleNotFoundError`. *Fix:* a no-op `BetterTransformer.transform` shim (modern transformers has SDPA built in).
2. **`sentencepiece` missing** (airllm imports a Baichuan tokenizer at load). *Fix:* install it.
3. **Single-file model rejected** — airllm asserts `model.safetensors.index.json`; Qwen2.5-0.5B ships unsharded. *Fix:* re-save with `max_shard_size` to force an index.
4. **Tied embeddings crash the splitter** — Qwen ties `lm_head` to `embed_tokens`, so `lm_head` is absent from the index and `split_and_save_layers` does `shards[0]` on `[]` → `IndexError`. *Fix:* untie/materialize `lm_head` before sharding.
5. **CUDA hard-wired in init** — on a CPU-only torch build, `init_model()` reaches `torch.cuda.empty_cache()` → "Torch not compiled with CUDA enabled". *Fix:* no-op `torch.cuda.empty_cache/reset_peak_memory_stats/synchronize`.
6. **`DynamicCache` not subscriptable** — transformers 5.x changed the KV-cache type. *Fix:* `use_cache=False`.
7. **`position_embeddings` required** — transformers 5.x's Qwen2 layer forward expects rotary `(cos,sin)` passed in, which airllm 2.11's hand-rolled loop never provides → `cannot unpack NoneType`. *Fix:* pin **transformers 4.41.2** (airllm 2.11's era), in a dedicated env.

**Real result (Qwen2.5-0.5B, CPU, transformers 4.41):** `TTFT ≈ 4.1 s, TPOT ≈ 3.8 s/token, peak RSS ≈ 370 MB` — vs the transformers backend's ~3,200 MB and ~0.09 s/token. So AirLLM trades **~8.7× less RAM for ~42× slower decode**, exactly the disk-bandwidth-bound prediction of the roofline (`reports/REPORT.md` §4). §4's per-block execution trace is *measured* on the real model; §5's bounded-working-set fault/evict dynamics are *modeled* (the synthetic pager) and labeled as such — both are now corroborated by the real run's RAM footprint.

**The fragility is itself the finding:** seven fixes to run a "run a big model locally" tool is precisely the operational friction L08 frames. localforge makes it non-fatal — because airllm 2.11 cannot import under the project's transformers 5.x, `AirLLMBackend.is_available()` detects the unusable import and the runner records a **skipped** row with the exact reason; the comparison never crashes (`src/localforge/backends/airllm_backend.py`). Reproduce the real run via the dedicated env in `experiments/airllm_real_run.py`.

## 7. Reproduce
```bash
# measured per-block trace + artifacts (PNG/HTML/JSONL)
uv run localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct --no-tui
# modeled AirLLM paging (no airllm needed)
uv run localforge visualize --no-tui            # synthetic
uv run localforge visualize --replay results/paging/paging.jsonl --no-tui
```
