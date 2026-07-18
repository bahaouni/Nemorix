# Eviction Policies

When the GPU is full and a new agent needs space, Nemorix must decide which sleeping agent
to evict (move back down to CXL/RAM/SSD). This decision is controlled by the eviction policy.

For a deep theoretical explanation of the Retention Law, see
[`src/nemorix/policies/README.md`](../../src/nemorix/policies/README.md).

## Available Policies

### LRU (Least-Recently-Used)

The baseline. Evicts the agent that was last active the longest time ago.

```python
from nemorix.policies.lru import LRUEvictionPolicy

policy = LRUEvictionPolicy()
# Sorts blocks by last_accessed timestamp ascending
# → block idle the longest gets evicted first
```

**Pros:** Simple, predictable, O(n log n) sort.  
**Cons:** Ignores what is being evicted. A high-priority VIP agent idle for 6 minutes
is evicted before a background job idle for 5 minutes. At 500 agents, only 63 stay
under the 200 ms SLA.

---

### Semantic — The Retention Law (Nemorix default)

The **Retention Score** R(b) ∈ [0, 1] estimates the value of *keeping* a block resident.
Lowest score gets evicted first.

```
R(b) = w_H × H(b)   +  w_I × I(b)   +  w_P × P(b)   +  w_C × C(b)
       0.25            0.30             0.20             0.25
       wake hazard     attention        priority         recompute cost
```

#### The four terms

| Term | Symbol | What it measures | Formula |
|---|---|---|---|
| **Wake hazard** | H(b) | Survival probability the agent is needed soon | `exp(-τ / μ_a)` — where τ is idle time and μ_a is the *learned* mean idle period |
| **Attention salience** | I(b) | Semantic importance of the block's tokens | Mean attention score + 0.10 boost if the block is an attention sink (score ≥ 0.90) |
| **Priority floor** | P(b) | Business importance of the owning agent | `1 - priority/10` — ensures VIP agents can never decay to zero |
| **Recompute cost** | C(b) | How expensive to lose this block | Rising with layer depth, context length, and tier reload time |

#### Knapsack layer: Cognitive Value Density

Eviction must free a specific number of **bytes**, not just a number of blocks.
Dividing by block size makes the selection optimal for that constraint:

```
CVD(b) = R(b) / size_MiB(b)
```

Evict ascending CVD — shed the least retention value per byte freed.

#### Pressure gate

Under light memory pressure, LRU is near-optimal and cheaper to compute.
Nemorix interpolates:

```
CVD_gated(b) = (1 − ρ) × LRU_rank(b) + ρ × CVD(b)

ρ = GPU occupancy ∈ [0, 1]
```

At ρ → 0: pure LRU. At ρ → 1: pure value-density. This explains why Nemorix ≈ LRU at
50 agents (CXL has room for everyone) but wins **4.4× at 500 agents** for seed 42 (CXL saturates).

#### Usage

```python
from nemorix.policies.semantic import SemanticEvictionPolicy

policy = SemanticEvictionPolicy(
    w_recency=0.25,       # wake-hazard weight
    w_importance=0.30,    # attention salience weight
    w_priority=0.20,      # priority floor weight
    w_recompute=0.25,     # recompute cost weight
    idle_mean_default=50.0,   # initial μ_a before first observation (seconds)
    idle_ewma_alpha=0.3,      # EWMA learning rate for μ_a
    sink_threshold=0.90,      # attention score threshold for sink boost
    sink_boost=0.10,          # additive boost for attention sinks
    reload_tier="cxl",        # tier to price reload cost against
    sla_ms=200.0,             # SLA threshold (ms)
)

# Tell the policy each agent's business priority (1 = most important)
policy.set_agent_priority("agent_001", priority=1)

# Called on every agent activation — updates the online μ_a model
policy.observe_access("agent_001", current_time=3600.0)
```

---

### Predictive Prefetcher (experimental)

Located in `src/nemorix/policies/prefetch.py`. Uses per-agent activation history to predict
which agents will wake up next and pre-loads them before the request arrives.

> **Status:** Implemented but not yet wired into the main simulation loop.
> Connecting it is a Phase 2 task.

---

## Writing a Custom Policy

Implement the `EvictionPolicy` protocol — it selects **KV blocks** (not agents)
to evict until `required_bytes` are freed:

```python
from typing import List
from nemorix.core.kv_block import KVBlock

class MyPolicy:
    def select_victims(
        self, blocks: List[KVBlock], required_bytes: int, current_time: float
    ) -> List[KVBlock]:
        """Return blocks to evict (ascending priority) until required_bytes freed."""
        victims, freed = [], 0
        for b in sorted(blocks, key=lambda b: b.size_bytes, reverse=True):
            victims.append(b)
            freed += b.size_bytes
            if freed >= required_bytes:
                break
        return victims
```

---

## Policy Comparison (50 agents, 24 h, seed 42)

| Policy | SLA Agents | Avg Latency | P99 Latency | $/agent-hr |
|---|---|---|---|---|
| No Offload | 0 / 50 | 1,205 ms | 1,638 ms | $7.01 |
| LRU | 50 / 50 | 16.8 ms | 27.8 ms | $0.17 |
| **Semantic (Nemorix)** | **50 / 50** | **16.2 ms** | **27.8 ms** | **$0.17** |

At 50 agents, both LRU and Semantic serve all agents under SLA — the CXL pool has room for
everyone, so the pressure gate keeps the policy near-LRU. The semantic advantage emerges at
**500 agents** when CXL saturates: Nemorix serves **4.4× more agents under SLA** than LRU
(276 vs 63, seed 42) by protecting blocks with higher modeled retention value in the CXL tier.
