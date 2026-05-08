# Changelog

All notable changes to Nemorix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-08

### Added

- **Core architecture**: 4-tier memory hierarchy (GPU VRAM / CXL / RAM / NVMe SSD)
  with physics-grounded transfer times and costs.
- **Agent Memory Object (AMO)**: dataclass representing an agent's KV-cache state,
  priority, activation probability, and lifecycle tracking.
- **KV-cache block model**: per-layer block abstraction with compression support
  (FP16 → FP8 → INT4) and attention score tracking.
- **Memory Tier Manager**: orchestrates block placement and migration across tiers
  with progressive compression on migration.
- **Agent Process Scheduler**: OS-style agent lifecycle management — running, ready,
  sleeping, suspended states with automatic idle suspension.
- **LRU eviction policy**: baseline Least Recently Used eviction for comparison.
- **Semantic eviction policy**: novel weighted scoring based on recency (0.25),
  attention importance (0.30), agent priority (0.20), and recompute cost (0.25).
- **Predictive prefetcher**: anticipates agent wake-ups based on activation probability.
- **Discrete-event simulator**: full simulation runner with configurable agent count,
  context length, tier sizes, and simulation duration.
- **Workload generator**: creates realistic agent workloads based on Llama-3-70B
  model configuration with varied context lengths and activation patterns.
- **FastAPI control plane**: REST API for agent create/resume/pause/list and
  system metrics (requires optional `api` dependencies).
- **Comprehensive test suite**: 74 tests covering physics accuracy, eviction policy
  correctness, scheduler lifecycle, simulation integrity, and scaling behavior.
- **Benchmark suite**: policy comparison across agent counts, simulation runner
  with CLI arguments, and visualization generation.
- **Documentation**: README, architecture documentation, project pitch,
  honest assessment, and roadmap.

### Hardware References

- GPU: NVIDIA H100 SXM5 HBM3 (3000 GB/s conservative, $40/GB/mo)
- CXL: Samsung CMM-D / SK Hynix Type-3 CXL 2.0 (64 GB/s, $4/GB/mo)
- RAM: Host DDR5 via PCIe 5.0 (50 GB/s, $2/GB/mo)
- SSD: NVMe PCIe Gen4 x4 (7 GB/s, $0.10/GB/mo)
- Prefill: 40,000 tokens/s on H100 FP16 (MLPerf Inference v4.0 lower-bound)
