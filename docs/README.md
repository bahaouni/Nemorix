# Nemorix

> **Virtual Memory for AI Agents**  
> CXL-aware KV-cache tiering that treats agent memory like an OS memory hierarchy.

---

## At a Glance

| | Without Nemorix | With Nemorix |
|---|---|---|
| Agents under 200ms SLA | **0** out of 50 | **50** out of 50 |
| Average resume latency | 1,205 ms | **9.9 ms** |
| P99 resume latency | 1,638 ms | **15.6 ms** |
| Cost per agent-hour | $7.01 | **$0.17** |

**122× faster · 98% cheaper · 8× more agents on the same GPU.**

---

## Why Nemorix?

Modern AI inference stacks assume all agents are always active. In practice:

- An AI assistant answers 3 questions per hour → **87% idle**
- An autonomous coding agent runs 10 min, waits 20 min for CI → **67% idle**
- A customer service bot runs only while a customer is typing → **90%+ idle**

Each idle agent's KV-cache still occupies 10–20 GB of expensive GPU VRAM.  
At 50 agents, you need 12–13 H100 GPUs just to keep everyone's cache warm.

Nemorix solves this with a four-tier memory hierarchy — **GPU → CXL → RAM → SSD** —
moving idle agents to cheaper storage and fetching them back in milliseconds.

---

## Install

```bash
pip install -e ".[dev]"
```

Then run the benchmark in one command:

```bash
python benchmarks/run_simulation.py
```

→ [Quick Start](guide/quickstart.md) for a full 5-step walkthrough.

---

## Project Status

> **Research-grade simulator (May 2026)**  
> All numbers come from physics-based equations using published hardware specs.  
> 74 automated tests validate every formula.  
> Real hardware validation (H100 + CXL DIMM) is the planned next phase.

---

## License

MIT — free to use, modify, and distribute.  
See [CONTRIBUTING](https://github.com/bahaouni/nemorix/blob/main/CONTRIBUTING.md) to get involved.
