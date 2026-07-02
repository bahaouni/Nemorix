# Nemorix

> **Virtual Memory for LLM Agent State** — CXL-aware KV-cache tiering.
> Keep hundreds of idle AI agents warm on one GPU: **~120× faster resume, 85% cheaper.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-74%20passing-brightgreen.svg)](tests/)

---

## What It Is

A single 80 GB GPU fits only ~6 Llama-3-70B agents (each needs ~21 GB of KV-cache).
Yet agents are idle 75–98% of the time. Nemorix treats agent KV-cache like OS
virtual memory, paging idle agents across a four-tier hierarchy:

| Tier | Capacity | Recall latency | Cost/GB/mo |
|------|----------|----------------|------------|
| GPU VRAM | 80 GB | — (active) | $40 |
| CXL Memory | 512 GB | ~10 ms | $4 |
| CPU RAM | 256 GB | ~20 ms | $2 |
| NVMe SSD | 4 TB | ~230 ms | $0.10 |

<<<<<<< Updated upstream
=======
When an idle agent wakes, Nemorix pages in just what it needs first —
**~10 ms** instead of **~1,200 ms** recomputation.

This repository is a **physics-based analytical simulator** validated by 74 tests,
plus a reference API. It is the research artifact behind the paper
([PAPER.md](PAPER.md)).
>>>>>>> Stashed changes

---

## Setup

```bash
# Clone and enter
git clone https://github.com/bahaouni/nemorix.git
cd nemorix

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# Install (pure standard library — no runtime dependencies)
pip install -e ".[dev]"
```

Requires **Python 3.10+**.

---

## Use

```bash
# 1. Run the tests (74 pass in ~3 min)
python -m pytest tests/ -q

# 2. Run the main benchmark (50 agents, 24 h)
python benchmarks/run_simulation.py

# 3. Reproduce the multi-seed robustness numbers
python benchmarks/run_robustness.py --seeds 8

# 4. See where semantic eviction wins (10 → 500 agents)
python benchmarks/compare_policies.py
```

From Python:

```python
from nemorix.simulation.runner import SimulationRunner, SimulationConfig

runner = SimulationRunner(SimulationConfig(num_agents=100))
metrics = runner.run("semantic")
print(metrics.sla_agents, metrics.avg_resume_latency_ms)
```

**Full simulator guide:** [benchmarks/README.md](benchmarks/README.md)

---

## Results (reproducible, seed 42)

| Metric | No Offload | LRU | **Nemorix** |
|--------|-----------|-----|-------------|
| Agents under 200 ms SLA | 0 / 50 | 50 / 50 | **50 / 50** |
| Avg resume latency | 1,205 ms | 10.4 ms | **9.9 ms** |
| P99 resume latency | 1,638 ms | 15.6 ms | **15.6 ms** |
| Cost per agent-hour | $7.01 | $0.17 | **$0.17** |

At **500 agents**, semantic eviction keeps **4.2× more agents under SLA** than LRU
(273 vs 65) once the CXL pool saturates.

---

## Project Layout

```
src/nemorix/        Core library (tiers, agents, scheduler, policies, API)
benchmarks/         Simulator entry points + JSON results  (see benchmarks/README.md)
tests/              74 tests validating hardware physics & simulation invariants
docs/               Full documentation site (docsify)
PAPER.md            The research manuscript
```

---

## Status

Research-grade simulator. All latency/cost numbers derive from published hardware
specs (H100 HBM3, Samsung CMM-D CXL, NVMe Gen4, MLPerf v4.0). Real hardware
validation (H100 + CXL DIMM) is Phase 2.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
