# Reverse-Engineering: Ollama

> Part of the HW5 RE arm (docs/REVERSE_ENGINEERING.md). Focus: how Ollama serves a local model and exposes it through an OpenAI-compatible API — the path localforge's `ollama` backend drives. Claims marked *(empirical)* are validated by our own backend interaction; *(architectural)* are from Ollama's public design.

## 1. What Ollama is
Ollama is a local model-serving daemon. It wraps a `llama.cpp`-based inference engine, manages a library of **GGUF** quantized models, and exposes two HTTP surfaces on `:11434`:
- its **native** API (`/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`), and
- an **OpenAI-compatible** API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).

The lecture (L08 §6) uses the OpenAI-compatible surface so existing OpenAI clients point at `OPENAI_BASE_URL=http://localhost:11434/v1`. localforge targets the same surface.

## 2. Serving architecture *(architectural)*
```
client ──HTTP──> ollama daemon ──> model scheduler ──> llama.cpp runner (GGUF, mmap) ──> tokens
                      │                  │
                  model store        keep-alive (model stays resident ~5m)
              (~/.ollama/models)
```
- **Model store:** pulled models live as GGUF blobs + a manifest under `~/.ollama/models`. `ollama pull <tag>` fetches them; tags look like `qwen2.5:0.5b`, *not* Hugging Face repo ids — this is why localforge treats the Ollama `model_id` as a distinct tag.
- **Scheduler / keep-alive:** the daemon loads a model on first request and keeps it resident for a short idle window, so the *first* request pays load cost and later ones do not. This matters for fair profiling: our profiler sees near-zero "load" for an already-warm model.
- **GGUF + mmap:** `llama.cpp` memory-maps the GGUF file, so weights page in from disk through the OS page cache — the same OS-paging theme AirLLM exploits, but at whole-file granularity with quantized weights.

## 3. The wire protocol localforge uses *(empirical)*
Endpoint: `POST {base}/chat/completions`. Request:
```json
{ "model": "qwen2.5:0.5b", "messages": [{"role":"user","content":"..."}],
  "stream": true, "max_tokens": 64, "temperature": 0, "seed": 0 }
```
With `stream:true` the daemon emits Server-Sent Events, one JSON chunk per line prefixed `data: `, terminated by `data: [DONE]`:
```
data: {"choices":[{"delta":{"content":"Virtual"}}]}
data: {"choices":[{"delta":{"content":" memory"}}]}
data: [DONE]
```
localforge's `ollama_backend.py` parses these deltas and yields each `content` piece, so the same prefill/decode profiling applies as for the transformers backend (first chunk = end of prefill).

## 4. How this contrasts with the other backends
| Aspect | Ollama | transformers | AirLLM |
|---|---|---|---|
| Weights | GGUF, quantized | SafeTensors, full precision | SafeTensors, streamed per layer |
| Residency | whole model (kept warm) | whole model in RAM | one layer at a time |
| Memory trick | mmap of a quantized file | none (full load) | mmap + layer eviction |
| Interface | HTTP (OpenAI-compatible) | in-process Python | in-process Python |
| localforge load cost | ~0 if warm | full load each run | per-layer, repeated |

## 5. Reproduce
```bash
ollama pull qwen2.5:0.5b
ollama serve            # exposes :11434
uv run localforge run --backend ollama --model qwen2.5:0.5b --prompt "Explain virtual memory."
```
If the daemon is down or the tag isn't pulled, localforge reports a *skipped* row with the exact remedy, and any comparison suite still completes.

## 6. Confidence & gaps
- §3 is directly verified by our backend. §2 is from Ollama's documented architecture; we did not fork `llama.cpp`, so internal scheduler details (exact keep-alive, batching) are stated at architectural confidence, not line-level. The full empirical line-level RE in this project targets **AirLLM** (docs/RE_AIRLLM.md), where the OS-paging behavior is the originality hook.
