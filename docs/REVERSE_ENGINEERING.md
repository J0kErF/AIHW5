# localforge — Reverse-Engineering Deliverable (RE)

> HW5 hybrid deliverable. Mirrors the HW4 pattern (RE of an existing system, à la Graphify/Obsidian) applied to the L08 systems, then **validated empirically** by localforge's instrumentation. The build (`localforge`) is the original extension layer on top of the reverse-engineered internals.

## 1. Why hybrid
The course runs two assignment modes: build-from-scratch (HW1–3) and reverse-engineering an existing repo (HW4 — `ex04-...Reverse-engineering.pdf`). HW5 ships in the same Jun-16 material drop as HW4 and has **no `ex05` spec file** in the folder, so the deliverable shape is genuinely ambiguous. The hybrid satisfies both: a rigorous RE report **and** an original tool. If the grader expects RE, §§3–5 deliver it; if they expect a build, `localforge` delivers it; the Paging Visualizer ties them together by *measuring* what the RE claims.

## 2. Target systems (in priority order)
| System | Why | RE depth |
|---|---|---|
| **AirLLM** | The lecture's headline: disk-as-virtual-memory, layer-by-layer execution, mmap/SafeTensors zero-copy. Core originality hook. | Deep — call graph, memory model, latency model. |
| **Ollama** (+ underlying `llama.cpp`/GGUF) | Local serving, OpenAI-compatible API, quantized GGUF runtime. | Medium — request path, model loading, API surface. |
| **transformers `generate()`** | Reference baseline; KV-cache, prefill/decode. | Light — enough to contrast prefill/decode against the above. |

## 3. RE questions to answer (AirLLM, primary)
- **Layering:** How is a model decomposed into independently-loadable transformer blocks? Where does the split happen, and how are block weights persisted (SafeTensors layout, sharding)?
- **Loading:** How is each layer brought into RAM — `mmap` vs explicit read? Is it zero-copy via the OS page cache? When is a layer evicted?
- **Memory model:** What is the working-set ceiling, and how does AirLLM keep it bounded while streaming a 70B model? Map each step to OS concepts (page, page table, page fault, MMU, hit/miss, hierarchy).
- **Compute path:** How do hidden states flow GPU↔CPU↔disk between layers? What is moved vs recomputed?
- **Latency model:** Where does time go — I/O (load) vs compute? How does this differ from full-residency `transformers`?
- **Instrumentation points:** Exactly which functions/objects must be hooked to emit `PagingEvent`s (fault/hit/evict, bytes, source, t_ms) without forking AirLLM?

## 4. RE methodology (reproducible)
1. **Acquire source** at a pinned version (record commit/tag) under `external/` (git submodule or vendored, license-respecting) — never committed weights.
2. **Static read:** map packages → modules → key classes; produce an annotated architecture diagram (Mermaid in this repo).
3. **Dynamic trace:** run the demo model under a tracer (`sys.settrace`/targeted logging) to capture the real per-layer load→compute→evict sequence; save a trace artifact.
4. **Synthesize:** write findings with evidence (file:line citations to the pinned source) and diagrams; state confidence per claim.
5. **Validate:** localforge's `paging/airllm_hook.py` instruments the identified points; the Paging Visualizer's measured fault/residency timeline is the **empirical proof** of the RE claims. Discrepancies between predicted and measured behavior are reported, not hidden.

## 5. RE deliverables (artifacts)
- `docs/RE_AIRLLM.md` — AirLLM architecture, memory model, latency model, OS-paging mapping, with diagrams + source citations.
- `docs/RE_OLLAMA.md` — Ollama request/serving path + GGUF loading + API surface.
- `docs/diagrams/` — Mermaid: AirLLM layer-stream sequence, memory-hierarchy map, Ollama request flow.
- `results/traces/airllm_demo.jsonl` — captured dynamic trace backing the claims.
- Empirical cross-check section: predicted vs measured (page faults, peak working set, I/O-vs-compute split) from the localforge profiler + visualizer.

## 6. How this maps onto the existing plan
- The RE findings **define** `paging/airllm_hook.py` instrumentation (TASKS T15) and the AirLLM backend boundaries (T9).
- The build docs (SPECIFICATION/IMPLEMENTATION/TASKS/PROMPT) stand as-is for the `localforge` arm; this doc adds the RE arm and two new analysis tasks:
  - **T9a (after T9):** AirLLM static + dynamic RE → `RE_AIRLLM.md` + trace artifact.
  - **T15a (after T15):** empirical cross-check (predicted vs measured) appended to `RE_AIRLLM.md` and surfaced in `report.html`.
- `RE_OLLAMA.md` is authored alongside T8 (Ollama backend).

## 7. Scope honesty
The **`~10k LOC / ~70 files`** figure is a *top-grade ambition* (it was the threshold for the HW4 "bring-your-own-repo" alternative), **not** a hard requirement — we will not generate bloat to hit a number; depth and reproducibility win. Likewise, treating **L08 §9 as the brief is our working interpretation** (the formal מטלה may live in the lecture recording, which is not in this folder); the hybrid design is deliberately robust to either interpretation.
