# Memory Tiers

Nemorix manages agent KV-cache across four hardware tiers, each with different latency, bandwidth, and cost:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — GPU VRAM       80 GB    1 µs   3,000 GB/s   $40/GB/mo (HBM3) │
│  TIER 2 — CXL Memory    512 GB    5 µs      36 GB/s   ~$4/GB/mo        │
│  TIER 3 — CPU RAM       256 GB   10 µs      50 GB/s   ~$2/GB/mo (DDR5) │
│  TIER 4 — NVMe SSD       4 TB   100 µs       7 GB/s  ~$0.10/GB/mo      │
```

## Tier Details

### Tier 1 — GPU VRAM

- **Hardware:** H100 HBM3 (or equivalent)
- **Bandwidth:** 3,000 GB/s — data is essentially instant relative to inference compute time
- **Role:** Where inference actually runs. Only agents being actively served live here.
- **Capacity constraint:** At ~20 GB per agent (64K context, FP16), an 80 GB GPU holds **4–6 agents** simultaneously.

### Tier 2 — CXL Memory (the key innovation)

- **Hardware:** Samsung CMM-D (MD220), SK Hynix Type-3 (CXL 2.0, PCIe 5.0 x16)
- **Bandwidth:** 36 GB/s measured sequential read (Samsung CMM-D specification)
- **Latency:** ~5 µs base + transfer time
- **Role:** “Warm” tier for agents idle for 5–10 minutes. Fastest recall after GPU.
- **Resume latency:** ~16 ms average for a 32–64K agent (on-demand partial load, FP8)
- **Cost:** ~$4/GB/month — 10× cheaper than HBM, 40× more expensive than SSD

> **Why CXL matters:** CXL’s primary contribution is **capacity**, not raw bandwidth.
> Its 512 GB pool prevents agents from spilling to NVMe (230 ms recall), which is what
> keeps all 50 agents under the 200 ms SLA. The 2.2× latency improvement over a
> RAM-only hierarchy comes from avoiding that SSD overflow — not because 36 GB/s
> beats DDR5’s 50 GB/s for sequential reads (it doesn’t). This is Nemorix’s primary
> hardware differentiator: no other KV-cache system uses CXL as a managed tier.

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
# src/nemorix/core/tier_manager.py → MemoryTierManager.__init__
```

```python
from nemorix.core.tier_manager import MemoryTierManager

# Create a manager with custom tier capacities (GB)
manager = MemoryTierManager(gpu_gb=80, cxl_gb=1024, ram_gb=512, ssd_gb=8000)
```

## Transfer Time Formula

Nemorix computes transfer latency as:

```
latency_ms = (bytes / bandwidth_bytes_per_s) × 1000 + base_latency_ms
```

For a 1 GiB block:
- **GPU ↔ GPU:** 0.3 ms (3,000 GB/s)
- **CXL read:** 27.8 ms (36 GB/s, Samsung CMM-D measured)
- **RAM read:** 20.0 ms (50 GB/s)
- **SSD read:** 143.0 ms (7 GB/s)

These values are validated by `tests/test_accuracy.py::test_cxl_transfer_time_36gbps`.
