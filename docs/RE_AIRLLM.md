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

## 6. Honesty: what we could and could not run — a *verified* fragility chain
We seriously attempted a real AirLLM CPU run on `airllm==2.11.0`. Rather than one blocker, we empirically hit **five**, each fixed in turn until a sixth, architectural one:

1. **`optimum.bettertransformer` removed** (in the resolved `optimum>=2.2`). AirLLM imports it at module load → `ModuleNotFoundError`. *Fixed* with a ~6-line no-op `BetterTransformer.transform` shim (modern transformers has SDPA built in).
2. **`sentencepiece` missing** — AirLLM eagerly imports a Baichuan tokenizer at package load. *Fixed* by installing it.
3. **Single-file model rejected** — AirLLM asserts `model.safetensors.index.json` exists; Qwen2.5-0.5B ships unsharded. *Fixed* by re-saving with `max_shard_size` to force an index.
4. **Tied embeddings crash the splitter** — Qwen2.5-0.5B ties `lm_head` to `embed_tokens`, so `lm_head` is absent from the index and AirLLM's `split_and_save_layers` does `shards[0]` on an empty list → `IndexError`. *Fixed* by untying/materializing `lm_head` before sharding.
5. **CUDA hard-wired in init** — even with `device="cpu", dtype=float32`, `AirLLMBaseModel.init_model()` reaches a CUDA call (`Torch not compiled with CUDA enabled`). AirLLM's design streams layers *to a GPU* with limited VRAM; clean CPU-only execution is not supported in 2.11 on this platform without patching internals.

**This is the verified finding:** running a giant-model tool like AirLLM locally is operationally fragile — it took five fixes to get to a sixth, architectural wall (it wants a GPU). This is precisely the friction L08 frames; localforge's value is making it *visible and non-fatal*. Consequently:

- §4 execution numbers are **measured** (real per-block hooks on the actual model — the same instrumentation AirLLM's streaming would feed).
- §5 bounded-working-set memory dynamics are **modeled**, explicitly labeled, because AirLLM itself would not run CPU-only here.

localforge degrades gracefully: `AirLLMBackend.is_available()` detects the unusable import and the runner records a **skipped** row with the exact reason — the comparison never crashes (`src/localforge/backends/airllm_backend.py`). Reproduce the chain: `uv sync --extra airllm` then `uv run localforge run --backend airllm --model Qwen/Qwen2.5-0.5B-Instruct` → skipped with the optimum reason. The full investigation script and the five fixes are recorded here for a GPU host, where steps 1–4 plus a real device would complete the run.

## 7. Reproduce
```bash
# measured per-block trace + artifacts (PNG/HTML/JSONL)
uv run localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct --no-tui
# modeled AirLLM paging (no airllm needed)
uv run localforge visualize --no-tui            # synthetic
uv run localforge visualize --replay results/paging/paging.jsonl --no-tui
```
