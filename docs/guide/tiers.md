# Memory Tiers

Nemorix manages agent KV-cache across four hardware tiers, each with different latency, bandwidth, and cost:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — GPU VRAM       80 GB    1 µs   3,000 GB/s   $40/GB/mo (HBM3) │
│  TIER 2 — CXL Memory    512 GB    5 µs      64 GB/s    $4/GB/mo        │
│  TIER 3 — CPU RAM       256 GB   10 µs      50 GB/s    $2/GB/mo (DDR5) │
│  TIER 4 — NVMe SSD       4 TB   100 µs       7 GB/s   $0.10/GB/mo      │
└─────────────────────────────────────────────────────────────────────────┘
           ← fastest, smallest, most expensive
                                      cheapest, largest, slowest →
```

## Tier Details

### Tier 1 — GPU VRAM

- **Hardware:** H100 HBM3 (or equivalent)
- **Bandwidth:** 3,000 GB/s — data is essentially instant relative to inference compute time
- **Role:** Where inference actually runs. Only agents being actively served live here.
- **Capacity constraint:** At ~20 GB per agent (64K context, FP16), an 80 GB GPU holds **4–6 agents** simultaneously.

### Tier 2 — CXL Memory (the key innovation)

- **Hardware:** Samsung CMM-D, SK Hynix Type-3 (CXL 2.0, PCIe 5.0 x16)
- **Bandwidth:** 64 GB/s unidirectional read (source: Samsung CMM-D datasheet)
- **Latency:** ~5 µs base + transfer time
- **Role:** "Warm" tier for agents idle for 5–10 minutes. Fastest recall after GPU.
- **Resume latency:** ~25 ms average for a 32–64K agent (on-demand partial load)
- **Cost:** $4/GB/month — 10× cheaper than HBM, 40× more expensive than SSD

> **Why CXL matters:** No other KV-cache system uses CXL. It's the natural first tier below GPU VRAM — fast enough for sub-100ms recall, cheap enough to pool large amounts of agent state. This is Nemorix's primary hardware differentiator.

### Tier 3 — CPU RAM

- **Hardware:** DDR5 DRAM (server-grade)
- **Bandwidth:** 50 GB/s
- **Role:** "Sleeping" tier for agents idle 10–15 minutes. Larger pool than CXL.
- **Resume latency:** ~32 ms average

### Tier 4 — NVMe SSD

- **Hardware:** PCIe Gen4 NVMe (e.g., Samsung 990 Pro)
- **Bandwidth:** 7 GB/s sequential read
- **Role:** Cold storage for agents idle >15 minutes. Effectively unlimited capacity.
- **Resume latency:** ~230 ms average — noticeable, but far better than 1,200ms recompute.

## Progressive Compression

KV-cache blocks are quantized as they move to colder tiers, multiplying effective capacity:

| Migration | Quantization | Ratio | Quality Loss |
|---|---|---|---|
| GPU → CXL | FP16 → FP8 | 2× smaller | < 0.3% |
| CXL → RAM | FP8 → INT4 | 4× smaller vs FP16 | < 1.8% |
| RAM → SSD | INT4 (kept) | — | — |

> Quality loss estimates from: [KIVI (2024)](https://arxiv.org/abs/2402.02750), [KV-Quant (2024)](https://arxiv.org/abs/2401.18079)

## Configuring Tier Sizes

Tier sizes are set via CLI flags or by editing `SimulationConfig` directly:

```bash
# Custom GPU + CXL sizes via CLI
python benchmarks/run_simulation.py --gpu-gb 80 --cxl-gb 1024 --ram-gb 512 --ssd-gb 8000

# Inspect tier constants in code
# src/nemorix/core/tier_manager.py → TIER_CONFIGS dict
```

```python
from nemorix.core.tier_manager import MemoryTierManager, TierConfig

# Create a manager with custom tier sizes
manager = MemoryTierManager(
    tier_configs={
        "gpu":  TierConfig(capacity_gb=80,   bandwidth_gbps=3000, cost_per_gb_month=40.0),
        "cxl":  TierConfig(capacity_gb=1024,  bandwidth_gbps=64,  cost_per_gb_month=4.0),
        "ram":  TierConfig(capacity_gb=512,   bandwidth_gbps=50,  cost_per_gb_month=2.0),
        "ssd":  TierConfig(capacity_gb=8000,  bandwidth_gbps=7,   cost_per_gb_month=0.10),
    }
)
```

## Transfer Time Formula

Nemorix computes transfer latency as:

```
latency_ms = (bytes / bandwidth_bytes_per_s) × 1000 + base_latency_ms
```

For a 1 GiB block:
- **GPU ↔ GPU:** 0.3 ms (3,000 GB/s)
- **CXL read:** 15.6 ms (64 GB/s)
- **RAM read:** 20.0 ms (50 GB/s)
- **SSD read:** 143.0 ms (7 GB/s)

These values are validated by `tests/test_tier_manager.py::test_transfer_time`.
