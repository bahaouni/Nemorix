# Quick Start

Get Nemorix running and see the benchmark results in under 2 minutes.

## 1. Install

```bash
pip install -e ".[dev]"
```

> **Requirements:** Python 3.10+. No external dependencies for the core — pure standard library.

## 2. Run the Test Suite

Verify everything works:

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_accuracy.py ................... [ 52%]
tests/test_eviction.py .....         [ 65%]
tests/test_tier_manager.py .....      [ 78%]
tests/test_scheduler.py ...           [ 86%]
tests/test_simulation_integrity.py .................... [100%]

74 passed in ~60s
```

## 3. Run the Benchmark

```bash
python benchmarks/run_simulation.py
```

This runs a 24-hour simulation with 50 AI agents and prints a comparison table:

```
  Metric                         No Offload        LRU      Nemorix
  ─────────────────────────────────────────────────────────────────
  Agents under SLA (<200ms)               0         50         50
  Max GPU-resident agents                 6          6          6
  Avg resume latency               1205.5ms      10.4ms      9.9ms
  P50 resume latency               1279.3ms      10.0ms      9.6ms
  P99 resume latency               1638.1ms      15.6ms     15.6ms
  GPU utilization                       90%        90%        91%
  Eviction accuracy                     N/A        42%        41%
  Cost per agent-hour (incl. GPU)     $7.01       $0.17      $0.17
```

## 4. Use the Python API

```python
from nemorix.core.tier_manager import MemoryTierManager
from nemorix.core.scheduler import AgentScheduler
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator

# Create the tier manager (defaults match H100 + CXL + DDR5 + NVMe)
manager = MemoryTierManager()
policy = SemanticEvictionPolicy()
scheduler = AgentScheduler(manager, policy)

# Build an agent with its per-layer KV blocks
agent = WorkloadGenerator(seed=42).create_agent(
    context_tokens=32_768,    # 32K tokens
    priority=5,               # 0 (highest) - 10 (lowest)
    activation_prob=0.1,      # 10% chance active per step
)
policy.set_agent_priority(agent.agent_id, agent.priority)
scheduler.register_agent(agent)

# Activate the agent (pages blocks to GPU). Time is in seconds.
latency_ms = scheduler.activate_agent(agent.agent_id, current_time=0.0)
print(f"Agent resumed in {latency_ms:.1f} ms")

# When done for now, demote it to CXL
scheduler.deactivate_agent(agent.agent_id, "cxl", current_time=60.0)
```

## 5. Run the Policy Sweep

Compare all three policies across multiple agent counts (10 / 25 / 50 / 75 / 100):

```bash
python benchmarks/compare_policies.py
```

## 6. Launch the Interactive API (optional)

```bash
pip install -e ".[api]"
uvicorn nemorix.api.server:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) — interactive Swagger UI where you can:

- `POST /agents` — register a new agent
- `POST /agents/{id}/resume` — wake it up and see measured latency
- `GET /agents` — list all agents and their current tier
- `GET /metrics` — live GPU/CXL/RAM/SSD utilization

---

**Next:** [Installation options →](guide/installation.md) or jump straight to [Core Concepts →](guide/tiers.md)
