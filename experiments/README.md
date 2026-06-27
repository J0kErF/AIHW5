# experiments/

Reproducible benchmark/analysis scripts that sit on top of the `localforge` library.

| File | What it does |
|---|---|
| `airllm_real_run.py` | A **real** AirLLM layer-streamed CPU run (spec §5.3), printing TTFT/TPOT/peak RAM. Needs a dedicated transformers-4.41 env (instructions in the file header) because airllm 2.11 is incompatible with the transformers 5.x the main project uses. |

Most benchmarking is driven directly through the CLI rather than bespoke scripts:

```bash
uv run localforge baseline                                    # §5.1 baseline
uv run localforge compare --suite config/suites/demo.yaml     # §5.2/5.4 backends + metrics
uv run localforge finetune --method lora                      # §5.7 PEFT
uv run localforge visualize --model Qwen/Qwen2.5-0.5B-Instruct # §5.6 paging
uv run localforge econ                                        # §5.5 API-vs-OnPrem breakeven
```

Outputs land in `results/` (per-run) and `figures/` (roofline, breakeven); the
synthesis is `reports/REPORT.md`.
