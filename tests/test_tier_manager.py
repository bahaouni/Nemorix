"""Tests for MemoryTierManager."""

from __future__ import annotations

from nemorix.core.tier_manager import MemoryTierManager
from nemorix.core.kv_block import KVBlock
from nemorix.policies.lru import LRUEvictionPolicy


def test_tier_creation():
    """Test that tiers are created with correct capacities."""
    mgr = MemoryTierManager(gpu_gb=80, cxl_gb=512, ram_gb=256, ssd_gb=4000)
    assert mgr.get_tier("gpu").capacity_bytes == 80 * 1024**3
    assert mgr.get_tier("cxl").capacity_bytes == 512 * 1024**3
    assert mgr.get_tier("ssd").capacity_bytes == 4000 * 1024**3
    print("  [PASS] test_tier_creation")


def test_block_migration():
    """Test migrating a block between tiers."""
    mgr = MemoryTierManager(gpu_gb=1, cxl_gb=2, ram_gb=2, ssd_gb=10)
    block = KVBlock(
        block_id="test_001",
        agent_id="agent_1",
        layer_idx=0,
        num_tokens=1024,
        size_bytes=1024 * 1024,  # 1 MB
        dtype="fp16",
        tier="gpu",
    )
    mgr.get_tier("gpu").allocate(block.block_id, block.size_bytes)

    # Migrate GPU → CXL
    latency = mgr.migrate_block(block, "cxl")
    assert block.tier == "cxl"
    assert latency > 0
    assert block.dtype == "fp8"  # CXL tier compresses to FP8
    assert block.size_bytes < 1024 * 1024  # Should be smaller after compression

    # Migrate CXL → SSD
    latency2 = mgr.migrate_block(block, "ssd")
    assert block.tier == "ssd"
    assert block.dtype == "int4"
    assert latency2 > latency  # SSD is slower

    print("  [PASS] test_block_migration")


def test_transfer_time():
    """Test that transfer times are physics-based."""
    mgr = MemoryTierManager()
    gpu = mgr.get_tier("gpu")
    cxl = mgr.get_tier("cxl")
    ssd = mgr.get_tier("ssd")

    size = 1024**3  # 1 GB

    gpu_time = gpu.transfer_time_ms(size)
    cxl_time = cxl.transfer_time_ms(size)
    ssd_time = ssd.transfer_time_ms(size)

    # CXL should be faster than SSD
    assert cxl_time < ssd_time
    # GPU should be fastest
    assert gpu_time < cxl_time

    print(f"  [PASS] test_transfer_time (GPU: {gpu_time:.1f}ms, CXL: {cxl_time:.1f}ms, SSD: {ssd_time:.1f}ms)")


def test_eviction():
    """Test space reclamation via eviction."""
    mgr = MemoryTierManager(gpu_gb=0.001, cxl_gb=1, ram_gb=1, ssd_gb=10)
    policy = LRUEvictionPolicy()

    # Fill GPU with blocks
    blocks = []
    for i in range(10):
        b = KVBlock(
            block_id=f"b_{i}",
            agent_id="agent_1",
            size_bytes=100 * 1024,  # 100 KB each
            last_accessed=float(i),
            tier="gpu",
        )
        mgr.get_tier("gpu").allocate(b.block_id, b.size_bytes)
        blocks.append(b)

    # Try to make space for 500 KB
    evicted = mgr.ensure_space("gpu", 500 * 1024, blocks, policy, current_time=100.0)
    assert len(evicted) > 0
    # Oldest blocks should be evicted first (LRU)
    assert evicted[0].block_id == "b_0"

    print("  [PASS] test_eviction")


def test_cost_calculation():
    """Test cost per hour calculation."""
    mgr = MemoryTierManager(gpu_gb=80, cxl_gb=512, ram_gb=256, ssd_gb=4000)
    # Put 10 GB in GPU
    mgr.get_tier("gpu").used_bytes = 10 * 1024**3
    cost = mgr.total_cost_per_hour()
    assert cost > 0
    # GPU cost for 10 GB: 10 * (40/720) = $0.556/hr
    expected = 10 * (40.0 / 720.0)
    assert abs(cost - expected) < 0.01

    print(f"  [PASS] test_cost_calculation (10GB GPU = ${cost:.3f}/hr)")


if __name__ == "__main__":
    print("Running tier manager tests...")
    test_tier_creation()
    test_block_migration()
    test_transfer_time()
    test_eviction()
    test_cost_calculation()
    print("\nAll tier manager tests passed!")
