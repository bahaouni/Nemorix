# Python API Reference

## `MemoryTierManager`

Manages allocation, migration, and eviction of KV-cache blocks across the four tiers.

```python
from nemorix.core.tier_manager import MemoryTierManager, TierConfig

manager = MemoryTierManager()
```

### Constructor

```python
MemoryTierManager(tier_configs: dict[str, TierConfig] | None = None)
```

If `tier_configs` is omitted, uses defaults (H100 + Samsung CMM-D CXL + DDR5 + NVMe Gen4).

### Methods

```python
# Migrate a KV block from its current tier to target_tier
# Returns the transfer latency in milliseconds
manager.migrate_block(block: KVBlock, target_tier: str) -> float

# Ensure enough space exists in target_tier for n_bytes
# Evicts blocks (calling the policy) if necessary
manager.ensure_space(n_bytes: int, tier: str, policy: EvictionPolicy) -> None

# Get total cost per hour across all tiers
manager.total_cost_per_hour() -> float

# Get current utilization fraction for a tier
manager.utilization(tier: str) -> float  # 0.0 – 1.0

# Get available space in bytes for a tier
manager.available_bytes(tier: str) -> int
```

---

## `AgentMemoryObject`

Represents a single AI agent and all its KV-cache blocks.

```python
from nemorix.core.agent import AgentMemoryObject

agent = AgentMemoryObject(
    agent_id="my-agent",
    total_context_tokens=65_536,
    activation_probability=0.10,
    priority=7,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str` | required | Unique identifier |
| `total_context_tokens` | `int` | required | Context window size |
| `activation_probability` | `float` | `0.1` | Chance of activation per sim step |
| `priority` | `int` | `5` | 1 (lowest) to 10 (highest) |
| `num_layers` | `int` | `80` | Model layers (Llama-3-70B default) |
| `num_kv_heads` | `int` | `8` | KV attention heads |
| `head_dim` | `int` | `128` | Head dimension |

### Properties

```python
agent.total_size_bytes    # int  — total KV-cache size across all blocks
agent.primary_tier        # str  — 'gpu' | 'cxl' | 'ram' | 'ssd'
agent.state               # str  — 'running' | 'sleeping' | 'suspended'
agent.avg_resume_latency_ms  # float — rolling average over all resumes
```

---

## `AgentScheduler`

OS-style process scheduler for agent lifecycle management.

```python
from nemorix.core.scheduler import AgentScheduler
from nemorix.policies.semantic import SemanticEvictionPolicy

scheduler = AgentScheduler(
    tier_manager=manager,
    eviction_policy=SemanticEvictionPolicy(),
    idle_threshold_secs=300,
)
```

### Methods

```python
# Activate an agent (loads blocks from current tier to GPU if not already there)
# Returns resume latency in milliseconds
scheduler.activate_agent(agent: AgentMemoryObject) -> float

# Mark agent as idle (triggers migration after idle_threshold_secs)
scheduler.deactivate_agent(agent: AgentMemoryObject) -> None

# Scan all agents; migrate those past idle threshold down the tier hierarchy
scheduler.suspend_idle_agents(current_time_secs: float) -> None
```

---

## `KVBlock`

Atomic unit of KV-cache storage — one transformer layer for one agent.

```python
from nemorix.core.kv_block import KVBlock

block = KVBlock(
    block_id="block-001",
    agent_id="my-agent",
    layer_idx=0,
    num_tokens=65_536,
    dtype="fp16",
    attention_score=0.72,
)
```

### Key Fields

| Field | Type | Description |
|---|---|---|
| `block_id` | `str` | Unique block identifier |
| `agent_id` | `str` | Owning agent |
| `layer_idx` | `int` | Transformer layer index (0 = embedding-adjacent) |
| `num_tokens` | `int` | Number of tokens represented |
| `dtype` | `str` | `'fp16'` | `'fp8'` | `'int4'` |
| `attention_score` | `float` | 0.0–1.0 — higher = block is more important |
| `last_accessed` | `float` | Unix timestamp of last access |
| `tier` | `str` | Current tier: `'gpu'` \| `'cxl'` \| `'ram'` \| `'ssd'` |

### Methods

```python
block.size_bytes()             # int — current size in bytes
block.compressed_size("fp8")  # int — projected size after compression
block.compress_to("fp8")       # KVBlock — returns new block at target dtype
block.copy()                   # KVBlock — deep copy
```

---

## `SemanticEvictionPolicy`

The default eviction policy. Scores agents by four weighted factors.

```python
from nemorix.policies.semantic import SemanticEvictionPolicy

policy = SemanticEvictionPolicy(
    recency_weight=0.25,
    importance_weight=0.30,
    priority_weight=0.20,
    recompute_weight=0.25,
)

# Select the best agent to evict from a list
victim = policy.select_eviction_candidate(agents)
```

---

## `LRUEvictionPolicy`

Baseline — evicts the agent idle the longest.

```python
from nemorix.policies.lru import LRUEvictionPolicy

policy = LRUEvictionPolicy()
victim = policy.select_eviction_candidate(agents)
```

---

## `SimulationConfig` and `SimulationRunner`

For running a full discrete-event simulation programmatically:

```python
from nemorix.simulation.runner import SimulationRunner, SimulationConfig

config = SimulationConfig(
    num_agents=50,
    max_tokens=65_536,
    sim_hours=24,
    seed=42,
    gpu_gb=80,
    cxl_gb=512,
    ram_gb=256,
    ssd_gb=4000,
)

runner = SimulationRunner(config)
results = runner.run()   # returns SimulationResults dataclass

print(f"Avg latency: {results.nemorix.avg_latency_ms:.1f} ms")
print(f"Agents under SLA: {results.nemorix.sla_agents}")
print(f"Cost/agent-hr: ${results.nemorix.cost_per_agent_hour:.2f}")
```
