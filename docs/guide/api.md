# Python API Reference

## `MemoryTierManager`

Manages allocation, migration, and eviction of KV-cache blocks across the four tiers.

```python
from nemorix.core.tier_manager import MemoryTierManager

manager = MemoryTierManager()   # defaults: 80 / 512 / 256 / 4000 GB
```

### Constructor

```python
MemoryTierManager(gpu_gb=80, cxl_gb=512, ram_gb=256, ssd_gb=4000)
```

Defaults model H100 HBM3 + Samsung CMM-D CXL + DDR5 + NVMe Gen4. Set `cxl_gb` to a
tiny value (e.g. `0.001`) to disable the CXL tier for ablations.

### Methods

```python
# Migrate a KV block to target_tier (applies tier compression).
# Returns the transfer latency in milliseconds.
manager.migrate_block(block: KVBlock, target_tier: str) -> float

# Migrate all of an agent's blocks; returns on-demand resume latency (ms).
manager.migrate_agent_blocks(blocks: list[KVBlock], target_tier: str) -> float

# Evict blocks from a tier (via the policy) until required_bytes fit.
# Returns the evicted blocks (cascaded to the next colder tier).
manager.ensure_space(tier_name, required_bytes, all_blocks, policy, current_time) -> list[KVBlock]

# Total storage cost per hour across all tiers.
manager.total_cost_per_hour() -> float

# Access a single tier object for utilization / free space.
tier = manager.get_tier("gpu")
tier.utilization      # float 0.0-1.0
tier.free_bytes       # int
```

---

## `AgentMemoryObject`

Represents a single AI agent and all its KV-cache blocks. In practice you create
agents with `WorkloadGenerator`, which builds the per-layer `KVBlock` list for you:

```python
from nemorix.simulation.workload import WorkloadGenerator

gen = WorkloadGenerator(seed=42)
agent = gen.create_agent(
    context_tokens=65_536,
    priority=7,
    activation_prob=0.10,
)
```

You can also construct one directly (an empty `blocks` list is allowed):

```python
from nemorix.core.agent import AgentMemoryObject

agent = AgentMemoryObject(
    agent_id="my-agent",
    priority=7,
    total_context_tokens=65_536,
    activation_probability=0.10,
)
```

### Key Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str` | `""` | Unique identifier |
| `blocks` | `list[KVBlock]` | `[]` | Per-layer KV-cache blocks |
| `state` | `str` | `"suspended"` | `running` / `ready` / `sleeping` / `suspended` |
| `priority` | `int` | `5` | 0 (highest) to 10 (lowest) |
| `total_context_tokens` | `int` | `0` | Context window size |
| `activation_probability` | `float` | `0.1` | Chance of activation per sim step |

### Properties

```python
agent.total_size_bytes       # int   — total KV-cache size across all blocks
agent.total_size_mb          # float — same, in MiB
agent.primary_tier           # str   — 'gpu' | 'cxl' | 'ram' | 'ssd' | 'none'
agent.avg_resume_latency_ms  # float — average over all recorded resumes
```

---

## `AgentScheduler`

OS-style process scheduler for agent lifecycle management.

```python
from nemorix.core.scheduler import AgentScheduler
from nemorix.policies.semantic import SemanticEvictionPolicy

scheduler = AgentScheduler(manager, SemanticEvictionPolicy())
scheduler.register_agent(agent)
```

### Methods

```python
# Register an agent so the scheduler can manage it.
scheduler.register_agent(agent: AgentMemoryObject) -> None

# Activate an agent by id (pages its blocks to GPU, evicting if needed).
# Returns resume latency in milliseconds. Time is in seconds.
scheduler.activate_agent(agent_id: str, current_time: float) -> float

# Move an agent by id down to a colder tier ("cxl" | "ram" | "ssd").
scheduler.deactivate_agent(agent_id: str, target_tier: str, current_time: float) -> None

# Scan all running agents; demote those idle past the threshold. Returns count.
scheduler.suspend_idle_agents(current_time: float, idle_threshold_s: float = 300.0) -> int

# Introspection
scheduler.get_running_count() -> int
scheduler.get_state_counts() -> dict[str, int]
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
    importance_score=0.72,
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
| `importance_score` | `float` | 0.0–1.0 — policy-agnostic importance (higher = keep) |
| `last_accessed` | `float` | Unix timestamp of last access |
| `tier` | `str` | Current tier: `'gpu'` \| `'cxl'` \| `'ram'` \| `'ssd'` |

### Fields & Methods

```python
block.size_bytes               # int   — current size in bytes (field; mutated on compress)
block.size_mb                  # float — current size in MiB (property)
block.compressed_size("fp8")  # int   — projected size after compression (no mutation)
block.compress_to("fp8")       # None  — compress in place (updates size_bytes + dtype)
block.copy()                   # KVBlock — copy with a fresh block_id
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
