# What is Nemorix?

Nemorix is a **memory management layer for LLM inference systems** that run many simultaneous AI agents.

## The Problem

Each AI agent (a chatbot session, a coding assistant, an autonomous task worker) builds up a
**KV-cache** — the compressed memory of everything it has processed so far. For a large model
like Llama-3-70B at a 64K-token context window, that cache is roughly **20 GB per agent**.

A single H100 GPU has 80 GB of VRAM. Without memory management, you can host **at most 4–6
agents simultaneously** — and when any of them need to come back after being evicted, the GPU
must spend **1–2 seconds recomputing** their entire context from scratch.

## The Solution

Nemorix acts like a **virtual memory system** for AI agents — the same concept an OS uses to
make programs think they have more RAM than physically exists:

```
┌─────────────────────────────────────────────────────────┐
│  GPU VRAM      80 GB   3,000 GB/s   $40/GB/mo  (active) │
│  CXL Memory  512 GB      64 GB/s    $4/GB/mo   (ready)  │
│  CPU RAM     256 GB      50 GB/s    $2/GB/mo   (sleeping)│
│  NVMe SSD     4 TB        7 GB/s   $0.10/GB/mo (cold)   │
└─────────────────────────────────────────────────────────┘
```

When an agent goes idle, Nemorix moves its KV-cache down to cheaper, slower storage.
When it wakes up, Nemorix fetches just the layers it needs first — so inference starts in
**milliseconds instead of seconds**.

## Proven Results

| Metric | Without Nemorix | With Nemorix |
|---|---|---|
| Agents under 200ms SLA | **0** out of 50 | **50** out of 50 |
| Average resume latency | 1,205 ms | **9.9 ms** |
| P99 resume latency | 1,638 ms | **15.6 ms** |
| Cost per agent-hour | $7.01 | **$0.17** |

> All results from a deterministic simulation (seed=42, 50 agents, 64K tokens, 24h).
> Reproduce with: `python benchmarks/run_simulation.py`

## What Makes It Different

| Feature | vLLM | NVIDIA Dynamo | **Nemorix** |
|---|---|---|---|
| Cross-tier offloading | ❌ GPU only | ✅ GPU→RAM→SSD | ✅ GPU→CXL→RAM→SSD |
| CXL warm tier | ❌ | ❌ | ✅ |
| Semantic eviction | ❌ LRU only | ❌ LRU only | ✅ Attention + priority |
| Agent lifecycle model | ❌ | ❌ | ✅ Sleep/wake/migrate |
| Progressive compression | ❌ | ❌ | ✅ FP16→FP8→INT4 |

## Next Steps

- [Quick Start →](guide/quickstart.md) — up and running in 2 minutes
- [Memory Tiers →](guide/tiers.md) — understand the four-tier hierarchy
- [Eviction Policies →](guide/eviction.md) — how Nemorix decides what to evict
