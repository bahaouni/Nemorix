# Eviction Policies

When the GPU is full and a new agent needs space, Nemorix must decide which sleeping agent
to evict (move back down to CXL/RAM/SSD). This decision is controlled by the eviction policy.

## Available Policies

### LRU (Least-Recently-Used)

The baseline. Evicts the agent that was last active the longest time ago.

```python
from nemorix.policies.lru import LRUEvictionPolicy

policy = LRUEvictionPolicy()
# Sorts agents by last_accessed timestamp ascending
# → agent idle the longest gets evicted first
```

**Pros:** Simple, predictable, O(n log n) sort.
**Cons:** Ignores the cost of a wrong eviction. A rarely-used agent with a 60K-token context
costs much more to recompute than a frequently-used agent with a 4K context — LRU treats
them identically.

### Semantic (Nemorix's default)

A four-factor weighted score that considers more than just recency:

```
evict_score = 0.25 × recency
            + 0.30 × importance
            + 0.20 × priority
            + 0.25 × recompute_cost
```

The agent with the **lowest** score is evicted first.

| Factor | Description | Weight |
|---|---|---|
| `recency` | How recently was this agent last active? (lower = evict sooner) | 0.25 |
| `importance` | Average attention score across KV layers (lower = safe to evict) | 0.30 |
| `priority` | Business priority — higher number = protect more | 0.20 |
| `recompute_cost` | How expensive to recompute? (function of token count × layers) | 0.25 |

```python
from nemorix.policies.semantic import SemanticEvictionPolicy

policy = SemanticEvictionPolicy(
    recency_weight=0.25,
    importance_weight=0.30,
    priority_weight=0.20,
    recompute_weight=0.25,
)
```

**Pros:** Protects high-value agents (complex reasoning state, high priority). Sends agents
with high recompute cost to CXL (25ms recall) rather than SSD (230ms recall).

**Cons:** Requires attention scores to be set on KV blocks — in simulation, these are randomly
assigned; in production, they come from the model's attention output.

### Predictive Prefetcher (experimental)

Located in `src/nemorix/policies/prefetch.py`. Uses per-agent activation history to predict
which agents will wake up next and pre-loads them before the request arrives.

> **Status:** The prefetcher logic is implemented but not yet wired into the main simulation loop.
> Connecting it is a Phase 2 task.

```python
from nemorix.policies.prefetch import PredictivePrefetcher

prefetcher = PredictivePrefetcher(history_window=20)
# Call each step to pre-stage agents predicted to activate soon
prefetcher.step(agents, current_time, scheduler)
```

## Writing a Custom Policy

Implement the `EvictionPolicy` protocol — it selects **KV blocks** (not agents)
to evict until `required_bytes` are freed:

```python
from typing import Protocol, List
from nemorix.core.kv_block import KVBlock

class EvictionPolicy(Protocol):
    def select_victims(
        self, blocks: List[KVBlock], required_bytes: int, current_time: float
    ) -> List[KVBlock]:
        """Return the blocks to evict (in order) until required_bytes are freed."""
        ...
```

Example — evict the largest blocks first (free the most bytes per eviction):

```python
class LargestFirstPolicy:
    def select_victims(self, blocks, required_bytes, current_time):
        victims, freed = [], 0
        for b in sorted(blocks, key=lambda b: b.size_bytes, reverse=True):
            victims.append(b)
            freed += b.size_bytes
            if freed >= required_bytes:
                break
        return victims
```

Pass your policy to the scheduler:

```python
from nemorix.core.scheduler import AgentScheduler

scheduler = AgentScheduler(manager, LargestFirstPolicy())
```

## Policy Comparison (50 agents, 24h)

| Policy | SLA Agents | Avg Latency | P99 Latency | $/agent-hr |
|---|---|---|---|---|
| No Offload | 0 / 50 | 1,205 ms | 1,638 ms | $7.01 |
| LRU | 50 / 50 | 10.4 ms | 15.6 ms | $0.17 |
| **Semantic (Nemorix)** | **50 / 50** | **9.9 ms** | **15.6 ms** | **$0.17** |

At 50 agents, both LRU and semantic serve all agents under SLA. The semantic advantage
emerges at 500+ agents when CXL saturates: Nemorix serves 4.2× more agents under SLA
than LRU (273 vs 65) by preferentially keeping high-value agents in the CXL tier.
