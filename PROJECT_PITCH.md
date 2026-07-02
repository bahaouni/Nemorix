# Nemorix — The Operating System for Persistent AI Agents

## The One-Liner

**"Virtual memory for AI cognition — Nemorix is the first OS-style runtime that keeps hundreds of persistent AI agents alive on a single GPU by treating their memory as a CXL-native, semantically-managed hierarchy."**

---

## The Problem (30 seconds)

Every AI inference system today assumes agents are **disposable**. When GPU memory fills up, agent state is thrown away and must be rebuilt from scratch — 1.2 seconds of wasted computation per wake-up. That works for chatbots. It does not work for the next wave:

- **Persistent coding assistants** that hold your entire repo context for hours
- **Legal review swarms** with 500 agents analyzing a 10,000-page case
- **Autonomous security analysts** monitoring systems 24/7, sleeping between alerts
- **Multi-day research agents** running experiments, pausing, resuming

These agents are idle 75–98% of the time. Their memory sits in $40/GB/month GPU VRAM doing nothing. When they wake up, they lose their brain and have to re-read everything.

**The hidden bottleneck in agentic AI is not compute. It is memory.**

---

## The Solution

Nemorix treats each AI agent like an **OS process** — with sleep states, virtual memory, and page migration across a 4-tier hardware hierarchy:

| Tier | Technology | Bandwidth | Cost | Role |
|------|-----------|-----------|------|------|
| GPU VRAM | H100 HBM3 (80 GB) | 3,000 GB/s | $40/GB/mo | Active agents |
| **CXL Memory** | Samsung CMM-D (512 GB) | 64 GB/s | $4/GB/mo | **Warm idle agents** — the key insight |
| CPU RAM | DDR5 (256 GB) | 50 GB/s | $2/GB/mo | Sleeping agents |
| NVMe SSD | Gen4 (4 TB) | 7 GB/s | $0.10/GB/mo | Cold archive |

When an agent goes idle, its state flows down. When it wakes up, Nemorix loads just the first 10% of layers and begins inference immediately — **9.9 ms instead of 1,200 ms**.

The key innovation: a **semantic eviction policy** that uses attention patterns, agent priority, and recompute cost to decide what stays close to the GPU. Not just "least recently used" — the agents that matter most stay warmest.

---

## Why This Is Different From Everything Else

### The Reframing

Everyone else optimizes for **requests**. We optimize for **residents**.

| Traditional OS | Nemorix |
|---|---|
| Process scheduling | Agent scheduling |
| Virtual memory | KV-cache hierarchy |
| Page migration | Tier migration (GPU → CXL → RAM → SSD) |
| Swap space | NVMe cold archive |
| NUMA locality | CXL locality |
| Sleep / Wake | Agent activation lifecycle |
| Page replacement (LRU) | Semantic eviction (attention + priority + cost) |

This is not an incremental improvement. It is a **new abstraction layer** for AI infrastructure.

### The CXL Insight Nobody Else Has

Most people think CXL is "memory expansion" or "hyperscaler plumbing."

We see it as an **AI runtime primitive**: idle agents are warm state, not cold storage. CXL's latency profile (5 µs access, 64 GB/s bandwidth) maps perfectly onto the "idle but ready" state of AI agents. This is a hardware capability searching for its killer application.

---

## The Numbers (Reproducible, Deterministic, Fair)

50 agents, Llama-3-70B, 64K tokens each, 24 hours, seed=42.

| Metric | No Offload (GPU-Only) | LRU + 4-Tier | **Nemorix** |
|---|---|---|---|
| Agents under 200ms SLA | 0 / 50 | 50 / 50 | **50 / 50** |
| Avg resume latency | 1,205 ms | 10.4 ms | **9.9 ms** |
| P99 latency | 1,638 ms | 15.6 ms | **15.6 ms** |
| Cost per agent-hour | $7.01 | $0.17 | **$0.17** |

**122× faster resume. 98% cost reduction. 8× agent density.**

### Where Semantic Eviction Wins

At moderate scale, CXL alone does most of the work — both LRU and Nemorix perform similarly. The semantic advantage emerges **under memory pressure**:

| 500 Agents | LRU | Nemorix |
|---|---|---|
| Agents under SLA | 65 / 500 | **273 / 500** |

**4.2× more agents served under SLA.** When CXL fills up and agents spill to SSD (230ms recall), the semantic policy protects the agents that matter most.

### Why You Can Trust These Numbers

- **Deterministic**: Every run produces identical results (seeded RNG)
- **Fair comparison**: LRU and Nemorix use the **same 4-tier hardware** — only the eviction algorithm differs
- **Hardware specs from datasheets**: H100 HBM3 (3,000 GB/s), Samsung CMM-D CXL (64 GB/s), MLPerf v4.0 (40K tok/s)
- **Physics-validated**: Transfer times computed from bytes/bandwidth + base latency
- **74 automated tests** verify every formula against spec sheets

---

## Audience-Specific Pitches

### For Investors (30 seconds)

"The biggest hidden bottleneck in agentic AI is not FLOPs — it's memory. Every idle agent wastes $40/GB/month in GPU VRAM doing nothing. Nemorix is the first runtime that treats AI agents like OS processes, migrating their state across GPU, CXL, RAM, and SSD based on lifecycle and semantic importance. We cut costs 98% and get 8× density on the same hardware. CXL is early enough that the architecture isn't locked — we define the software layer. This is the 'virtual memory moment' for AI infrastructure."

### For Researchers (1 minute)

"We apply OS virtual memory abstractions — process states, page migration, tiered storage — to LLM agent KV-cache management. The key architectural insight is that CXL pooled memory (Samsung CMM-D, 64 GB/s, $4/GB/mo) is the natural warm tier for idle agent state. Our semantic eviction policy uses a 4-factor scoring model (attention importance, recency, agent priority, recompute cost) and outperforms LRU by 4.2× in SLA compliance at 500 agents. Physics-based simulation calibrated against H100/CXL spec sheets, 74 automated tests, MIT license."

### For Enterprise Customers

"You're running 50 coding agents on 8 GPUs because each agent needs 20 GB of context. With Nemorix, one GPU serves all 50 — the other 7 are free. Drop-in layer for vLLM. No code changes to your agents. The idle ones sleep in CXL memory and wake up in 10 ms instead of 1.2 seconds."

### For CXL Hardware Partners (Samsung, SK Hynix, Intel)

"Your CXL DIMMs need killer apps. We are building the runtime that makes CXL essential for AI inference — not as bulk memory expansion, but as a first-class agent state tier. Nemorix is open source, MIT licensed, and designed to showcase CXL's unique latency/cost profile. Joint paper opportunity, hardware validation partnership."

---

## Market Timing

| Factor | Status |
|---|---|
| CXL hardware availability | Samsung CMM-D shipping 2024, Intel/AMD server support ramping |
| Agent frameworks | LangChain, AutoGen, CrewAI all moving to persistent agents |
| Context windows | Growing (128K → 1M+), making KV-cache even bigger |
| GPU shortage | Density gains (8× per GPU) directly save on fleet costs |
| Competition | Nobody combines CXL + semantic eviction + agent lifecycle |

---

## What The Company Could Become

**Path 1 — Infrastructure Runtime** (like Databricks, Redis, Modal): The standard memory orchestration layer for agent deployments. Every inference platform integrates Nemorix.

**Path 2 — Acquisition by CXL/GPU vendor** (NVIDIA, Samsung, SK Hynix, Intel, AMD): We define the software layer that makes CXL essential for AI. Hardware vendors acquire to own the stack.

**Path 3 — Open Source Standard** (like Kubernetes for containers): Nemorix becomes "the virtual memory for AI" — open source adoption drives ecosystem gravity.

---

## Defensibility

This is not easy to copy because it requires simultaneous expertise in:
- Operating systems (scheduling, virtual memory, process management)
- LLM inference internals (KV-cache structure, attention mechanics)
- CXL hardware (bandwidth/latency characteristics, NUMA topology)
- Memory hierarchy optimization (compression, tiered storage economics)
- Semantic scheduling theory (attention-aware eviction, recompute cost models)

That combination of systems, ML, and hardware knowledge is the moat.

---

## Biggest Risk (and why it's manageable)

**Risk**: Future models reduce KV-cache size (linear attention, state-space models, compressed context).

**Why it's manageable**: Even with smaller KV-caches, persistent agent state management still matters. The abstraction (OS-style lifecycle for AI workers) transcends any single architecture. And transformer KV-caches are getting bigger (128K → 1M+ tokens), not smaller, for the foreseeable horizon.

