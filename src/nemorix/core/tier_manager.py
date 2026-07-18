"""Memory Tier Manager — orchestrates KV blocks across GPU/CXL/RAM/SSD."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol
from nemorix.core.kv_block import KVBlock


@dataclass
class MemoryTier:
    name: str
    capacity_bytes: int
    latency_us: float  # base access latency
    bandwidth_gbps: float  # GB/s
    cost_per_gb_month: float
    compress_dtype: str = "fp16"  # dtype to use when storing in this tier
    used_bytes: int = 0
    block_ids: List[str] = field(default_factory=list)

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    @property
    def utilization(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

    @property
    def cost_per_gb_hour(self) -> float:
        return self.cost_per_gb_month / 720.0

    def transfer_time_ms(self, size_bytes: int) -> float:
        size_gb = size_bytes / (1024**3)
        return (size_gb / self.bandwidth_gbps) * 1000.0 + self.latency_us / 1000.0

    def can_fit(self, size_bytes: int) -> bool:
        return self.free_bytes >= size_bytes

    def allocate(self, block_id: str, size_bytes: int) -> None:
        self.used_bytes += size_bytes
        self.block_ids.append(block_id)

    def release(self, block_id: str, size_bytes: int) -> None:
        self.used_bytes = max(0, self.used_bytes - size_bytes)
        if block_id in self.block_ids:
            self.block_ids.remove(block_id)


class EvictionPolicy(Protocol):
    def select_victims(
        self, blocks: List[KVBlock], required_bytes: int, current_time: float
    ) -> List[KVBlock]: ...


class MemoryTierManager:
    TIER_ORDER = ["gpu", "cxl", "ram", "ssd"]

    # Hardware reference specs (May 2026 pricing, H100 SXM5 + CXL 2.0 ecosystem)
    # GPU  : NVIDIA H100 SXM5 HBM3, 3350 GB/s (modeled at 3000 GB/s conservative)
    # CXL  : Samsung CMM-D (MD220) CXL 2.0 Type-3, measured ~36 GB/s sequential read
    #        (PCIe 5.0 x16 raw is ~63 GB/s; CXL protocol overhead + controller limits → 36)
    # RAM  : Host DDR5 via PCIe 5.0 bridge, ~50 GB/s from GPU perspective
    # SSD  : NVMe PCIe Gen4 x4, ~7 GB/s sequential read (Samsung 990 Pro class)
    # Costs: H100 spot ~$3/hr; CXL DRAM ~$4/GB/mo; DDR5 ~$2/GB/mo; NVMe ~$0.10/GB/mo
    def __init__(
        self,
        gpu_gb: float = 80,
        cxl_gb: float = 512,
        ram_gb: float = 256,
        ssd_gb: float = 4000,
    ):
        GB = 1024**3
        self.tiers: dict[str, MemoryTier] = {
            # latency_us, bandwidth_gbps, cost_per_gb_month
            "gpu": MemoryTier("gpu", int(gpu_gb * GB), 1.0, 3000.0, 40.0, "fp16"),
            "cxl": MemoryTier("cxl", int(cxl_gb * GB), 5.0, 36.0, 4.0, "fp8"),
            "ram": MemoryTier("ram", int(ram_gb * GB), 10.0, 50.0, 2.0, "int4"),
            "ssd": MemoryTier("ssd", int(ssd_gb * GB), 100.0, 7.0, 0.10, "int4"),
        }

    def get_tier(self, name: str) -> MemoryTier:
        return self.tiers[name]

    def next_colder_tier(self, tier_name: str) -> str | None:
        idx = self.TIER_ORDER.index(tier_name)
        if idx + 1 < len(self.TIER_ORDER):
            return self.TIER_ORDER[idx + 1]
        return None

    def migrate_block(self, block: KVBlock, target_tier: str) -> float:
        """Move block to target tier. Returns transfer time in ms."""
        src_tier = self.tiers[block.tier]
        dst_tier = self.tiers[target_tier]
        src_tier.release(block.block_id, block.size_bytes)
        # Apply compression for the destination tier
        old_size = block.size_bytes
        block.compress_to(dst_tier.compress_dtype)
        transfer_time = dst_tier.transfer_time_ms(old_size)
        dst_tier.allocate(block.block_id, block.size_bytes)
        block.tier = target_tier
        return transfer_time

    def migrate_agent_blocks(
        self, blocks: List[KVBlock], target_tier: str
    ) -> float:
        """Migrate all blocks for an agent. Returns total latency (on-demand)."""
        if not blocks:
            return 0.0
        # On-demand paging: only first 10% of layers needed immediately
        first_batch = max(1, len(blocks) // 10)
        latency = 0.0
        for i, block in enumerate(blocks):
            if block.tier == target_tier:
                continue
            t = self.migrate_block(block, target_tier)
            if i < first_batch:
                latency += t  # sequential for first batch
        return latency

    def ensure_space(
        self,
        tier_name: str,
        required_bytes: int,
        all_blocks: List[KVBlock],
        policy: EvictionPolicy,
        current_time: float,
    ) -> List[KVBlock]:
        """Evict blocks until tier has enough space. Returns evicted blocks."""
        tier = self.tiers[tier_name]
        if tier.can_fit(required_bytes):
            return []
        tier_blocks = [b for b in all_blocks if b.tier == tier_name]
        victims = policy.select_victims(
            tier_blocks, required_bytes - tier.free_bytes, current_time
        )
        colder = self.next_colder_tier(tier_name)
        evicted = []
        for v in victims:
            if colder:
                self.migrate_block(v, colder)
            evicted.append(v)
            if tier.can_fit(required_bytes):
                break
        return evicted

    def total_cost_per_hour(self) -> float:
        cost = 0.0
        for tier in self.tiers.values():
            used_gb = tier.used_bytes / (1024**3)
            cost += used_gb * tier.cost_per_gb_hour
        return cost
