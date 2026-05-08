# Nemorix — CXL-Aware Virtual Memory for Persistent LLM Agent State

> **A hardware-software co-designed memory architecture that gives long-running AI agents a persistent, pageable brain — not just a cache.**

> [!NOTE]
> **This is a research-grade simulator, not yet validated against real CXL/GPU hardware.**
> All performance numbers are derived from physics-grounded simulation using published
> hardware specifications (H100 SXM5, Samsung CMM-D CXL 2.0, NVMe Gen4). The simulator
> has been verified with 74 automated tests. Real hardware validation is the next step.

[![CI](https://github.com/nemorix-project/nemorix/actions/workflows/ci.yml/badge.svg)](https://github.com/nemorix-project/nemorix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-7c3aed)](https://bahaouni.github.io/nemorix/)

---

## Documentation

**Full docs site:** [bahaouni.github.io/nemorix](https://bahaouni.github.io/nemorix/)

Covers: [Quick Start](https://bahaouni.github.io/nemorix/#/guide/quickstart) · [Memory Tiers](https://bahaouni.github.io/nemorix/#/guide/tiers) · [Eviction Policies](https://bahaouni.github.io/nemorix/#/guide/eviction) · [CLI Reference](https://bahaouni.github.io/nemorix/#/guide/cli) · [Python API](https://bahaouni.github.io/nemorix/#/guide/api) · [Benchmarks](https://bahaouni.github.io/nemorix/#/guide/benchmarks)

> To enable GitHub Pages: go to your repo → **Settings → Pages → Source: Deploy from branch → Branch: main, Folder: /docs**.

---

## Quick Start

```bash
# Install (zero dependencies for core)
pip install -e .

# Run the test suite (74 tests)
pip install -e ".[dev]"
python -m pytest tests/ -v

# Run the benchmark
python benchmarks/run_simulation.py

# Start the API server (optional)
pip install -e ".[api]"
uvicorn nemorix.api.server:app --reload
```

---

## The One-Liner

**Nemorix is a virtual memory subsystem for LLM inference state that treats KV-cache as a schedulable, persistent resource across GPU VRAM, CXL-pooled memory, system RAM, and NVMe — enabling 10× more concurrent autonomous agents per GPU cluster at 60-80% lower memory cost.**

---

## Table of Contents

1. [Why This Matters Now](#1-why-this-matters-now)
2. [What Exists Today (Honest Assessment)](#2-what-exists-today-honest-assessment)
3. [What Nemorix Does Differently](#3-what-Nemorix-does-differently)
4. [Technical Architecture](#4-technical-architecture)
5. [The CXL Innovation (imec Fit)](#5-the-cxl-innovation-imec-fit)
6. [POC Plan & Deliverables](#6-poc-plan--deliverables)
7. [Business Case](#7-business-case)
8. [Competitive Landscape](#8-competitive-landscape)
9. [Risk Analysis](#9-risk-analysis)
10. [Team & Resources Needed](#10-team--resources-needed)
11. [Roadmap](#11-roadmap)

---

## 1. Why This Matters Now

### The Problem in One Number

A single long-context LLM agent (128K tokens, Llama-3-70B) consumes **~40 GB of GPU VRAM** just for its KV-cache. An H100 has 80 GB. That means **2 agents max per $30,000 GPU** — and most of the time those agents are *idle*, waiting for human input.

### The Industry Shift

The AI industry is moving from **stateless chat** (question → answer → forget) to **stateful agents** (multi-hour coding assistants, autonomous researchers, always-on enterprise workflows). This changes the economics fundamentally:

| Paradigm | Session Duration | KV-Cache Lifetime | Memory Model |
|---|---|---|---|
| Chatbot (2023) | 30 seconds | Ephemeral | Discard after response |
| Agent (2025-2026) | Hours to days | Persistent | Must survive idle periods |

**No production system today treats an agent's KV-cache as a persistent, resumable object.** They either keep it in VRAM (expensive) or delete it (requires costly recomputation).

### Why imec Should Care

This is a **memory hierarchy problem** — the exact class of problems imec's Compute & System Architecture (CSA) group has been solving for decades in different domains (processor caches, DRAM controllers, 3D-stacking). The AI inference memory wall is the next frontier for this expertise.

---

## 2. What Exists Today (Honest Assessment)

**We are honest about what already exists.** The novelty of Nemorix is NOT in basic KV-cache offloading — it is in the specific architecture and the CXL/hardware-aware layer.

### What IS Already Built

| System | What It Does | Limitation |
|---|---|---|
| **vLLM PagedAttention** | Paged KV-cache within GPU memory, prefix caching, LRU eviction | GPU-only; no cross-device memory hierarchy |
| **vLLM CPU Offloading** | Simple GPU→CPU offload connector | No intelligent policy; no SSD tier; no agent lifecycle |
| **NVIDIA Dynamo KVBM** | GPU→CPU→SSD→remote storage tiering | Request-centric (batch serving), not agent-centric; LRU/LFU only |
| **Mooncake** | Distributed KV-cache pool across cluster | Focuses on throughput for request serving, not agent persistence |
| **LMCache** | External KV-cache store for vLLM | Store/retrieve interface; no automated lifecycle management |
| **Dynamo v1.0 Agentic Hints** | Cache pinning TTL, per-request priority | Hints only — no automated agent scheduling or semantic eviction |

### What is NOT Solved (Our Opportunity)

1. **No CXL-aware KV-cache tier** — All systems treat CPU RAM as the only intermediate tier. CXL-attached memory (10× the capacity of VRAM at 3× the latency — vs 100× for SSD) is not used anywhere. This is a perfect fit for "warm" agent state.

2. **No semantic eviction** — Every system uses LRU (Least Recently Used) or LFU (Least Frequently Used). Nobody looks at *what* is in the KV-cache blocks to decide what to evict. A block containing a critical system prompt should not be evicted just because it was accessed less recently than a block of chat filler.

3. **No agent-level process scheduling** — Existing systems see "requests." Nemorix sees "agent processes" with priorities, deadlines, and sleep/wake cycles — like an operating system scheduler.

4. **No predictive pre-hydration** — Nobody predicts *which* agent will be needed next and starts pre-fetching its state from SSD→RAM→GPU before the request arrives.

---

## 3. What Nemorix Does Differently

### The Core Abstraction: Agent as a Process

Nemorix introduces a new abstraction: the **Agent Memory Object (AMO)**. An AMO encapsulates:

```
AgentMemoryObject {
    agent_id:           UUID
    kv_blocks:          List[KVBlock]        # Paged KV-cache blocks
    total_tokens:       int                  # Total context length
    priority:           float                # Business-level priority
    last_active:        timestamp            # Last inference call
    predicted_next:     timestamp            # ML-predicted next activation
    semantic_score:     float                # Importance of cached content
    current_tier:       GPU | CXL | RAM | SSD
    compression_state:  FP16 | FP8 | INT4   # Current quantization level
}
```

### The Four Innovations

#### Innovation 1: CXL as First-Class Memory Tier

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────┐     ┌──────────┐
│  GPU VRAM   │────▶│  CXL Pooled Memory  │────▶│ CPU RAM  │────▶│ NVMe SSD │
│  (L1 Hot)   │     │  (L1.5 Warm)        │     │ (L2 Cool)│     │(L3 Cold) │
│  80 GB      │     │  512 GB - 2 TB      │     │ 256 GB   │     │ 4+ TB    │
│  ~1 μs      │     │  ~3-5 μs            │     │ ~10 μs   │     │ ~100 μs  │
│  $$$$$      │     │  $$                  │     │ $        │     │ ¢        │
└─────────────┘     └─────────────────────┘     └──────────┘     └──────────┘
                            ▲
                            │
                    imec INNOVATION
                    (No one does this yet)
```

CXL (Compute Express Link) memory creates a **revolutionary intermediate tier** that doesn't exist in any current KV-cache system:
- **10-20× cheaper** per GB than HBM/VRAM
- **10-30× faster** than NVMe SSD
- **Poolable** across multiple GPUs (unlike local DRAM)
- **Cache-coherent** with the CPU (unlike RDMA)

This tier is uniquely suited for "warm" agent KV-cache — agents that are idle for minutes but need sub-millisecond resume.

#### Innovation 2: Semantic-Aware Eviction Policy

Instead of blind LRU, Nemorix scores each KV-cache block:

```
eviction_score(block) = α × recency_score
                      + β × attention_importance_score
                      + γ × agent_priority
                      + δ × reconstruction_cost

Where:
  - attention_importance_score: derived from the average attention weight
    received by tokens in this block during the last N forward passes
  - reconstruction_cost: how expensive it would be to recompute this block
    (longer prefixes = more expensive to recompute = keep them longer)
```

A system prompt block that every inference references gets a high attention_importance_score and a high reconstruction_cost → it stays in fast memory even if it was "written" long ago.

#### Innovation 3: Agent Process Scheduler

Nemorix's scheduler treats agents like OS processes:

```
┌──────────────────────────────────────────┐
│           Agent Scheduler                 │
│                                          │
│  RUNNING   [Agent-A] ──── GPU VRAM       │
│  READY     [Agent-B] ──── CXL Memory     │
│  SLEEPING  [Agent-C] ──── CPU RAM        │
│  SUSPENDED [Agent-D] ──── NVMe SSD       │
│                                          │
│  Policy: Priority + Predicted-Next-Use   │
│                                          │
│  On WAKE(Agent-C):                       │
│    1. Async prefetch SSD→RAM→CXL         │
│    2. On-demand page CXL→GPU             │
│    3. Mark Agent-C as RUNNING            │
│    4. If VRAM full: SLEEP lowest-priority│
└──────────────────────────────────────────┘
```

#### Innovation 4: Compression-on-Migration

When KV-cache moves to a colder tier, Nemorix applies progressive quantization:

```
GPU (FP16/BF16) → CXL (FP8) → RAM (INT4) → SSD (INT4 + zstd)

Compression ratios:     1×        2×         4×          8-12×
Quality loss:          0%       <0.5%      <2%          <2%
```

This means Nemorix doesn't just move data — it **shrinks it** as it gets colder, multiplying the effective capacity of each tier.

---

## 4. Technical Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                     │
│   create_agent()  pause_agent()  resume_agent()  list_agents()│
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                    Agent Process Scheduler                     │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Priority   │  │  Predictive  │  │  Semantic Importance  │ │
│  │  Queue      │  │  Prefetcher  │  │  Scorer               │ │
│  └────────────┘  └──────────────┘  └───────────────────────┘ │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                 Memory Tier Manager (C++ Core)                │
│  ┌──────┐  ┌───────┐  ┌──────┐  ┌───────┐  ┌─────────────┐ │
│  │ GPU  │  │  CXL  │  │ RAM  │  │  SSD  │  │ Compression │ │
│  │Alloc │  │ Pool  │  │Alloc │  │ I/O   │  │ Engine      │ │
│  └──────┘  └───────┘  └──────┘  └───────┘  └─────────────┘ │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│              Inference Engine Adapter                          │
│         ┌──────┐  ┌───────────┐  ┌───────────┐              │
│         │ vLLM │  │ llama.cpp │  │ TensorRT  │              │
│         └──────┘  └───────────┘  └───────────┘              │
└───────────────────────────────────────────────────────────────┘
```

### Core Data Structures

```python
@dataclass
class KVBlock:
    block_id: int
    layer_idx: int
    num_tokens: int
    data: bytes                          # Serialized KV tensors
    dtype: Literal["fp16", "fp8", "int4"]
    attention_score: float               # Running avg attention received
    last_accessed: float                 # Timestamp
    tier: Literal["gpu", "cxl", "ram", "ssd"]
    checksum: str                        # Integrity verification

@dataclass
class AgentMemoryObject:
    agent_id: str
    blocks: list[KVBlock]
    state: Literal["running", "ready", "sleeping", "suspended"]
    priority: int                        # 0 = highest
    total_context_tokens: int
    created_at: float
    last_inference_at: float
    next_predicted_activation: float     # From prediction model
    memory_footprint_bytes: int          # Across all tiers

class MemoryTierManager:
    """Manages the four-tier memory hierarchy."""

    def migrate_block(self, block: KVBlock, target_tier: str) -> None:
        """Move a KV block between tiers with optional compression."""

    def evict(self, required_bytes: int, tier: str) -> list[KVBlock]:
        """Evict blocks using semantic-aware policy."""

    def prefetch(self, agent_id: str, target_tier: str) -> None:
        """Async prefetch agent state to target tier."""

class AgentScheduler:
    """OS-style process scheduler for LLM agents."""

    def schedule(self, agent_id: str) -> None:
        """Bring an agent to RUNNING state, managing tier migrations."""

    def suspend(self, agent_id: str) -> None:
        """Move agent state to cold storage."""

    def predict_next_activation(self, agent_id: str) -> float:
        """Predict when this agent will be needed next."""
```

### Key Algorithms

#### Semantic Eviction Algorithm

```
FUNCTION select_eviction_candidates(required_bytes, tier):
    candidates = all_blocks_in(tier)

    FOR each block in candidates:
        block.eviction_score = compute_eviction_score(block)

    SORT candidates BY eviction_score ASCENDING  # Lowest score = evict first

    selected = []
    freed = 0
    FOR block in candidates:
        IF block.agent.state == "running":
            CONTINUE  # Never evict running agent blocks
        selected.append(block)
        freed += block.size
        IF freed >= required_bytes:
            BREAK

    RETURN selected

FUNCTION compute_eviction_score(block):
    recency    = time_decay(now() - block.last_accessed)
    importance = block.attention_score  # Higher = more important
    priority   = block.agent.priority   # Lower number = higher priority
    recompute  = block.num_tokens * block.layer_idx  # Cost to recompute

    RETURN (0.3 * recency) + (0.3 * importance) + (0.2 * (1/priority)) + (0.2 * recompute)
```

#### Predictive Pre-Hydration

```
FUNCTION background_prefetch_loop():
    WHILE True:
        FOR agent in sleeping_agents:
            predicted = predict_next_activation(agent)
            time_to_activation = predicted - now()

            IF time_to_activation < PREFETCH_THRESHOLD:
                # Start warming up: SSD → RAM → CXL
                async prefetch(agent, target_tier="cxl")

            ELIF time_to_activation < WARM_THRESHOLD:
                # At least get it to RAM
                async prefetch(agent, target_tier="ram")

        SLEEP(POLL_INTERVAL)
```

---

## 5. The CXL Innovation (imec Fit)

### Why CXL Is the Missing Piece

Current KV-cache systems have a "latency gap" in their memory hierarchy:

```
Without CXL:
  GPU VRAM (~1 μs) ──── 100× gap ──── CPU RAM (~10 μs) ──── 10× gap ──── SSD (~100 μs)
                         ▲
                    Agent wakes up here = noticeable delay

With CXL (Nemorix):
  GPU VRAM (~1 μs) ── 3-5× ── CXL (~3-5 μs) ── 2-3× ── RAM (~10 μs) ── 10× ── SSD
                                    ▲
                              Agent wakes up here = barely noticeable
```

CXL-attached memory gives us a **512 GB - 2 TB pool** that is:
- **Shared across multiple GPUs** (unlike each GPU's local DRAM)
- **Cache-coherent** (CPU and GPU see consistent data)
- **Hot-swappable** and expandable

### imec's Unique Position

imec has active research in:
- **CXL controller IP** and memory pooling architectures
- **3D-stacked memory** technologies
- **Datacenter system modeling** (kelis)
- **Hardware-software co-design** methodology

Nemorix provides a **real workload** that justifies CXL memory pooling in datacenters — a concrete application that imec can demonstrate to partners (Samsung, SK Hynix, Intel, AMD) as a killer use case for CXL memory products.

### Hardware-Software Co-Design Opportunities

| Hardware Knob | Software Adaptation |
|---|---|
| CXL latency characteristics | Eviction policy tuning — how "warm" does an agent need to be to justify CXL placement? |
| CXL bandwidth per device | Migration scheduling — how many agents can be paged concurrently? |
| CXL pool topology | Agent placement — which GPU-CXL pairing minimizes migration cost? |
| CXL memory capacity | Capacity planning — what ratio of CXL:VRAM maximizes $/agent? |

These are the exact questions imec's CSA group is built to answer.

---

## 6. POC Plan & Deliverables

### Phase 1: Simulation-Based MVP (Weeks 1-4)

**Goal:** Prove the concept works with measurable benefits, without real GPU hardware.

#### Deliverables

1. **Nemorix Simulator** (Python)
   - Simulates the 4-tier memory hierarchy with configurable latencies
   - Implements the Agent Process Scheduler
   - Implements semantic eviction vs LRU comparison
   - Generates latency/cost/concurrency metrics

2. **Workload Generator**
   - Simulates N concurrent agents with realistic activation patterns
   - Configurable: bursty, periodic, random arrival
   - Models real KV-cache sizes (measured from Llama-3-8B/70B)

3. **Benchmark Dashboard**
   - Agents-per-GPU vs baseline (no offloading, LRU offloading)
   - Resume latency per tier
   - GPU utilization over time
   - Cost per agent-hour

#### Technical Implementation

```
Nemorix/
├── README.md
├── pyproject.toml
├── src/
│   └── Nemorix/
│       ├── __init__.py
│       ├── core/
│       │   ├── agent.py              # AgentMemoryObject
│       │   ├── kv_block.py           # KVBlock dataclass
│       │   ├── tier_manager.py       # Memory tier management
│       │   └── scheduler.py          # Agent process scheduler
│       ├── policies/
│       │   ├── lru.py                # Baseline LRU eviction
│       │   ├── semantic.py           # Semantic-aware eviction
│       │   └── prefetch.py           # Predictive prefetcher
│       ├── simulation/
│       │   ├── memory_tier.py        # Simulated memory tiers
│       │   ├── workload.py           # Agent workload generator
│       │   └── runner.py             # Simulation orchestrator
│       ├── compression/
│       │   ├── quantize.py           # FP16→FP8→INT4 conversion
│       │   └── codec.py              # zstd compression wrapper
│       └── api/
│           ├── server.py             # FastAPI control plane
│           └── schemas.py            # API request/response models
├── benchmarks/
│   ├── run_simulation.py             # Main benchmark script
│   ├── compare_policies.py           # LRU vs Semantic comparison
│   └── plot_results.py              # Visualization
├── tests/
│   ├── test_scheduler.py
│   ├── test_tier_manager.py
│   └── test_eviction.py
└── docs/
    ├── architecture.md
    └── cxl_analysis.md
```

#### Key Metrics to Prove

| Metric | Baseline (No Offloading) | With LRU Offloading | Nemorix (Semantic + CXL) |
|---|---|---|---|
| Max concurrent agents per 80GB GPU | 2 | 5-8 | **15-30** |
| Agent resume latency | N/A (recompute: 5-30s) | 200-500ms (from SSD) | **5-50ms** (from CXL) |
| GPU VRAM utilization | 40% (idle waste) | 70% | **90%+** |
| Cost per agent-hour | $2.50 | $1.00 | **$0.30** |

### Phase 2: Real Engine Integration (Weeks 5-12)

**Goal:** Hook into vLLM's KV-cache lifecycle and demonstrate real offloading.

#### Deliverables

1. **vLLM KVConnector Plugin**
   - Implements vLLM's `KVConnectorBase` interface
   - Intercepts KV-cache allocation/deallocation events
   - Routes blocks through Nemorix's tier manager

2. **Real Memory Offloading**
   - GPU→CPU offloading via CUDA async memcpy
   - CPU→SSD offloading via async I/O
   - CXL tier simulated via NUMA-remote memory (for hardware-less testing)

3. **FastAPI Control Plane**
   - `POST /agents` — Create agent session
   - `POST /agents/{id}/pause` — Hibernate agent
   - `POST /agents/{id}/resume` — Wake agent
   - `GET /agents` — List agents with tier placement
   - `GET /metrics` — Memory utilization dashboard

### Phase 3: CXL Hardware Validation (Weeks 13-24)

**Goal:** Run on actual CXL memory hardware (available through imec partners).

#### Deliverables

1. **CXL Memory Integration**
   - Interface with CXL Type-3 memory devices
   - Measure real latency characteristics
   - Validate simulation predictions

2. **Datacenter TCO Model**
   - Use imec.kelis or equivalent to model full-rack economics
   - Compare: GPU-only vs GPU+CXL configurations
   - Publish results showing ROI of CXL for AI inference

3. **Partner Demo**
   - Live demonstration for CXL memory vendors
   - Benchmark report suitable for publication

---

## 7. Business Case

### The Market Pain

| Customer Segment | Pain Point | Willingness to Pay |
|---|---|---|
| **AI Agent Platforms** (Cognition, Devin, etc.) | Each coding agent ties up an entire GPU | $$$$ — their unit economics depend on it |
| **Enterprise AI** (banks, legal, healthcare) | On-prem agents idle 95% of the time | $$$ — GPU overprovisioning is their #1 cost |
| **Cloud Providers** (AWS, Azure, GCP) | Memory fragmentation limits batch sizes | $$$$ — directly impacts $/GPU/hour revenue |
| **GPU Cluster Operators** | Utilization stuck at 30-40% | $$$ — more agents per GPU = more revenue |

### Revenue Model

**Scenario: 1000 enterprises, each running 50 agents on 10 GPUs**

| Revenue Stream | Unit Price | Annual Revenue |
|---|---|---|
| **SaaS API** (managed memory service) | $0.10 per agent-hour managed | ~$43M |
| **Enterprise License** (on-prem) | $50K per cluster per year | ~$50M |
| **Hardware IP License** (CXL controller) | Partnership royalty | TBD |

### Cost Savings for Customers

A company running 100 concurrent coding agents (Llama-3-70B, 128K context):

| Configuration | GPUs Needed | Annual GPU Cost | Nemorix Savings |
|---|---|---|---|
| No offloading | 50× H100 | $1.5M | — |
| Basic CPU offloading | 20× H100 | $600K | 60% |
| **Nemorix (CXL-aware)** | **8× H100 + CXL** | **$350K** | **77%** |

### imec Valorization Paths

1. **Technology License to Cloud Providers** — License the CXL-aware KV-cache management IP
2. **Partner Programs** — Joint development with CXL memory vendors (Samsung CXL DRAM, SK Hynix CMM)
3. **imec.istart / xpand** — Spin out as inference infrastructure startup
4. **Research Programs** — Anchor a new bilateral research program with AI cloud customers

---

## 8. Competitive Landscape

```
                          High
                            │
              Nemorix       │
         ┌──────●──────┐   │
 Agent   │  CXL-aware  │   │        NVIDIA Dynamo KVBM
 Focus   │  Semantic    │   │          ┌─────●─────┐
         │  eviction    │   │          │ Multi-tier │
         └─────────────┘   │          │ but request│
                            │          │ -centric   │
                            │          └────────────┘
                            │
                            │   LMCache        Mooncake
                            │    ●               ●
              ──────────────┼──────────────────────────── Memory
              Request-      │                    Hierarchy
              centric       │                    Sophistication
                            │
                            │
                    vLLM     │
              ┌────●────┐   │
              │ PagedAttn│   │
              │ GPU-only │   │
              └─────────┘   │
                            │
                          Low
```

### Key Differentiators vs Dynamo KVBM

| Capability | Dynamo KVBM | Nemorix |
|---|---|---|
| Memory tiers | GPU→CPU→SSD→S3 | GPU→**CXL**→CPU→SSD |
| Eviction policy | LRU/LFU | **Semantic-aware** (attention + priority + recompute cost) |
| Agent lifecycle | Request-level hints | **Full agent process model** (sleep/wake/schedule) |
| Compression | None during migration | **Progressive quantization** (FP16→FP8→INT4) |
| Predictive prefetch | None | **ML-based activation prediction** |
| Hardware co-design | Software-only | **CXL hardware parameters inform policy** |
| Open source | Yes (Apache 2.0) | Proprietary IP (imec) |

---

## 9. Risk Analysis

| Risk | Severity | Mitigation |
|---|---|---|
| **CXL hardware not yet widely deployed** | Medium | POC works without CXL (CPU NUMA simulation); CXL adoption accelerating in 2026-2027 |
| **Dynamo adds agent-level features** | Medium | Our CXL + semantic eviction + compression are hardware innovations, not easily replicated in software-only |
| **KV-cache formats vary across engines** | Low | Start with vLLM (standardized block format); add adapters incrementally |
| **Restore latency from SSD too high** | Medium | CXL tier eliminates this; compression reduces transfer sizes; prefetching hides latency |
| **Quantization degrades model quality** | Low | Published research shows FP8 KV-cache has <0.5% quality loss; INT4 has <2% |
| **Integration complexity with inference engines** | Medium | vLLM already has pluggable KVConnector API; design for this interface from day 1 |

---

## 10. Team & Resources Needed

### Immediate (Phase 1: Simulation MVP)

| Role | Effort | Skills |
|---|---|---|
| Systems Engineer (you) | Full-time, 4 weeks | Python, systems architecture, LLM inference |
| Innovation Coach | Part-time advisory | imec innovation process guidance |

### Phase 2 (Real Integration)

| Role | Effort | Skills |
|---|---|---|
| Systems Engineer | Full-time, 8 weeks | Python, CUDA, vLLM internals |
| GPU Compute Access | 2× A100/H100 GPUs | For real vLLM integration testing |

### Phase 3 (CXL Validation)

| Role | Effort | Skills |
|---|---|---|
| Hardware Engineer | Part-time, 12 weeks | CXL protocol, memory controller design |
| Systems Engineer | Full-time, 12 weeks | C++, device drivers, benchmarking |
| CXL Hardware | 1× CXL dev kit | Through imec partners |

### Compute Resources

| Phase | Hardware | Purpose |
|---|---|---|
| Phase 1 | Laptop/workstation | Python simulation |
| Phase 2 | 2× A100/H100 (cloud OK) | vLLM integration |
| Phase 3 | CXL-capable server | Hardware validation |

---

## 11. Roadmap

### EXPLORE Phase (Now → Month 1)

- [x] Literature review of KV-cache management systems
- [x] Competitive analysis (vLLM, Dynamo, Mooncake, LMCache)
- [x] Define innovative concept and architecture
- [ ] **Build simulation-based MVP**
- [ ] **Generate benchmarks proving 10× agent density improvement**
- [ ] **Publish internal tech note with results**

### ASSESS Phase (Month 2-3)

- [ ] Validate concept with internal key opinion leaders (CSA group)
- [ ] Validate with external KOLs (CXL consortium contacts)
- [ ] Identify pilot deployment partner
- [ ] Submit to imec Innovation Challenge (Seed Project application)

### INVEST Phase — R&D Timebox (Month 4-6)

- [ ] vLLM KVConnector integration
- [ ] Real GPU offloading benchmarks
- [ ] CXL simulation with NUMA-remote memory
- [ ] Partner demo preparation

### INVEST Phase — AAA Project (Month 7-18)

- [ ] CXL hardware validation
- [ ] Datacenter TCO model (kelis integration)
- [ ] Publication: IEEE/ACM conference paper
- [ ] Partner bilateral discussion (Samsung, SK Hynix, AMD)

### VALORIZE Phase (Month 12+)

- [ ] Technology licensing to cloud providers
- [ ] imec.istart evaluation for spin-out
- [ ] Patent filing for CXL-aware KV-cache management

---

## How to Run the POC (Phase 1)

### Prerequisites

```bash
# Python 3.11+
pip install numpy matplotlib fastapi uvicorn pydantic
```

### Quick Start

```bash
# Clone and install
cd Nemorix
pip install -e .

# Run the simulation benchmark
python benchmarks/run_simulation.py \
  --num-agents 50 \
  --gpu-memory-gb 80 \
  --cxl-memory-gb 512 \
  --ram-gb 256 \
  --ssd-gb 4000 \
  --agent-context-tokens 65536 \
  --model-layers 80 \
  --simulation-hours 24

# Compare eviction policies
python benchmarks/compare_policies.py

# Start the control plane API
uvicorn Nemorix.api.server:app --port 8000
```

### Expected Output

```
=== Nemorix Simulation Results ===

Configuration:
  GPU VRAM: 80 GB | CXL: 512 GB | RAM: 256 GB | SSD: 4 TB
  Model: Llama-3-70B (80 layers, ~500 MB KV per 64K tokens)
  Agents: 50 concurrent, bursty activation pattern

Results:
  ┌────────────────────┬───────────┬──────────┬────────────┐
  │ Metric             │ No Offload│ LRU      │ Nemorix    │
  ├────────────────────┼───────────┼──────────┼────────────┤
  │ Max active agents  │ 2         │ 8        │ 28         │
  │ Avg resume latency │ 12.4s     │ 340ms    │ 18ms       │
  │ GPU utilization    │ 38%       │ 72%      │ 94%        │
  │ Eviction accuracy  │ N/A       │ 61%      │ 89%        │
  │ Cost per agent-hr  │ $2.50     │ $0.95    │ $0.28      │
  └────────────────────┴───────────┴──────────┴────────────┘

  14× improvement in agent density
  60× improvement in resume latency
  89% reduction in inference cost
```

---

## Key Takeaway for Innovation Challenge

**Nemorix is NOT just another KV-cache offloading system — those exist.**

Nemorix is:
1. **A CXL-native memory architecture** — the first KV-cache system designed for CXL pooled memory, a hardware technology imec is uniquely positioned to develop
2. **An agent lifecycle manager** — treating LLM agents as OS processes with sleep/wake/schedule semantics
3. **A hardware-software co-design project** — where CXL memory parameters directly inform software eviction policies

This sits at the intersection of imec's **memory hierarchy expertise**, the **booming AI inference market**, and an **unsolved infrastructure gap** that the industry will need to fill as agents go mainstream.

**The right framing: "What if LLM agents had an operating system — and that OS was co-designed with the memory hardware?"**

---

## Project Status

| Component | Status |
|---|---|
| Core memory hierarchy model | **Complete** — 4-tier architecture with physics-grounded constants |
| Semantic eviction policy | **Complete** — weighted scoring with 4 factors, tested against LRU |
| Agent process scheduler | **Complete** — OS-style lifecycle management |
| Discrete-event simulator | **Complete** — deterministic, 74 tests passing |
| FastAPI control plane | **Complete** — REST API for agent lifecycle |
| Benchmark suite | **Complete** — policy comparison, visualization |
| Hardware validation (CXL + H100) | **Planned** — requires lab access |
| vLLM integration | **Planned** — adapter for production inference engines |
| C++ performance core | **Planned** — for production throughput |

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. NVIDIA Dynamo — KV Block Manager documentation, ai-dynamo/dynamo GitHub, 2025-2026
3. Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," 2024
4. LMCache — vLLM KV-cache connector, lmcache.ai, 2024-2025
5. CXL Consortium — Compute Express Link Specification 3.1, 2024
6. vLLM CPU Offloading Connector — `v1/offloading_connector.py`, vLLM v0.20+

---

*Nemorix — giving AI agents a brain that doesn't disappear when you look away.*
