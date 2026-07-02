# Agent Lifecycle

Nemorix tracks every agent as an OS-style process through four states:

```
  suspended  ──activate──►  running  ──idle timeout──►  sleeping
      ▲                                                      │
      └──────────────────── evict ◄──────────────────────────┘
```

## States

| State | Blocks live in | Resume cost | Description |
|---|---|---|---|
| `running` | GPU VRAM | 0 ms | Currently serving inference |
| `sleeping` (ready) | CXL Memory | ~25 ms | Idle 5–10 min; fastest recall |
| `sleeping` | CPU RAM | ~32 ms | Idle 10–15 min |
| `suspended` | NVMe SSD | ~230 ms | Idle >15 min; cold storage |
| `suspended` | (nowhere, no_offload) | 820–1,638 ms | Must recompute from scratch |

## Idle Thresholds

Agents migrate down the tier hierarchy based on how long they've been idle:

| Idle time | Destination | New state |
|---|---|---|
| > 5 minutes (300s) | CXL Memory | `sleeping` (ready) |
| > 10 minutes (600s) | CPU RAM | `sleeping` |
| > 15 minutes (900s) | NVMe SSD | `suspended` |

## Using the Scheduler

```python
from nemorix.core.scheduler import AgentScheduler
from nemorix.core.tier_manager import MemoryTierManager
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator

manager = MemoryTierManager()
policy = SemanticEvictionPolicy()
scheduler = AgentScheduler(manager, policy)

# Build an agent (with its per-layer KV blocks) and register it
agent = WorkloadGenerator(seed=42).create_agent(
    context_tokens=65_536, priority=7, activation_prob=0.05,
)
policy.set_agent_priority(agent.agent_id, agent.priority)
scheduler.register_agent(agent)

# Activate (pages blocks to GPU). Time is in seconds.
latency_ms = scheduler.activate_agent(agent.agent_id, current_time=0.0)
print(f"Resumed in {latency_ms:.1f} ms; now in {agent.primary_tier}")

# Demote to CXL explicitly...
scheduler.deactivate_agent(agent.agent_id, "cxl", current_time=60.0)

# ...or let the scheduler demote idle agents automatically.
scheduler.suspend_idle_agents(current_time=400.0, idle_threshold_s=300.0)

print(f"Agent is now in: {agent.primary_tier}")
```

## Agent Fields

```python
@dataclass
class AgentMemoryObject:
    agent_id: str
    total_context_tokens: int           # e.g. 65_536
    activation_probability: float       # 0.0–1.0, per simulation step
    priority: int                       # 1 (low) to 10 (critical)
    state: str                          # 'running' | 'sleeping' | 'suspended'

    # Computed properties
    total_size_bytes: int               # sum of all KV block sizes
    primary_tier: str                   # 'gpu' | 'cxl' | 'ram' | 'ssd'
    avg_resume_latency_ms: float        # rolling average across all resumes
```

## Activation Probability

Every simulation step (= 1 minute of simulated time), each agent rolls a dice against its
`activation_probability`. A roll below the probability triggers an `activate_agent` call.

Typical values:
- **0.02** — background monitoring agent (active ~1.2 min/hr)
- **0.10** — interactive coding assistant (active ~6 min/hr)
- **0.25** — busy customer service bot (active ~15 min/hr)

You can set per-agent probabilities to model heterogeneous workloads:

```python
agents = [
    AgentMemoryObject("bot-1", tokens=32768, activation_probability=0.25, priority=9),
    AgentMemoryObject("bot-2", tokens=65536, activation_probability=0.05, priority=5),
    AgentMemoryObject("bot-3", tokens=16384, activation_probability=0.02, priority=1),
]
```
