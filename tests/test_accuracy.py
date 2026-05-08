"""Physics and formula accuracy tests.

Each test has an exact expected value from a published hardware spec or
mathematical derivation. These tests catch regressions where a constant
is accidentally changed, making the simulation unrealistic.

Reference sources used:
  - NVIDIA H100 SXM5 datasheet (HBM3 bandwidth 3350 GB/s)
  - Samsung CMM-D / SK Hynix AiMX CXL 2.0 datasheet (~64 GB/s read)
  - NVMe PCIe Gen4 spec (7 GB/s sequential read; Samsung 990 Pro)
  - Llama-3-70B model card (80 layers, 8 KV heads, 128 head_dim, FP16)
  - vLLM/MLPerf Inference v4 offline benchmarks (40-60K tokens/s prefill on H100 FP16)
"""

from __future__ import annotations
import math

from nemorix.core.kv_block import KVBlock, COMPRESSION_RATIOS
from nemorix.core.tier_manager import MemoryTierManager
from nemorix.core.agent import AgentMemoryObject
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator, ModelConfig
from nemorix.simulation.runner import RECOMPUTE_TOKENS_PER_SEC, SLA_THRESHOLD_MS

TOLERANCE = 0.01  # 1% tolerance for floating-point comparisons


# ---------------------------------------------------------------------------
# 1. KV-CACHE SIZE FORMULA
# ---------------------------------------------------------------------------

def test_kv_cache_size_llama70b():
    """Llama-3-70B KV-cache for 64K tokens should be ~21 GB (FP16)."""
    tokens = 65536
    layers = 80
    kv_heads = 8
    head_dim = 128
    dtype_bytes = 2  # FP16

    expected_bytes = tokens * layers * 2 * kv_heads * head_dim * dtype_bytes
    expected_gb = expected_bytes / 1e9

    model = ModelConfig("Llama-3-70B", layers, kv_heads, head_dim, dtype_bytes)
    computed = model.total_kv_bytes(tokens)

    assert computed == expected_bytes, (
        f"KV-cache size mismatch: got {computed}, expected {expected_bytes}"
    )
    assert 20.0 <= expected_gb <= 22.0, (
        f"KV-cache for 64K tokens should be 20-22 GB, got {expected_gb:.2f} GB"
    )
    print(f"  [PASS] test_kv_cache_size_llama70b ({expected_gb:.2f} GB per agent)")


def test_kv_cache_fits_in_80gb_gpu():
    """Single max-size agent (64K tokens, 70B) should fit in 80 GB GPU."""
    model = ModelConfig("Llama-3-70B", 80, 8, 128, 2)
    agent_size_bytes = model.total_kv_bytes(65536)
    gpu_capacity_bytes = 80 * 1024**3  # 80 GB

    assert agent_size_bytes < gpu_capacity_bytes, (
        f"Agent ({agent_size_bytes/1e9:.1f} GB) doesn't fit in 80 GB GPU — "
        "simulation would loop forever trying to evict"
    )
    print(f"  [PASS] test_kv_cache_fits_in_80gb_gpu "
          f"(agent={agent_size_bytes/1e9:.1f} GB < GPU=80 GB)")


def test_kv_bytes_per_layer():
    """Layer-level KV-cache size should be consistent with total."""
    model = ModelConfig("Llama-3-70B", 80, 8, 128, 2)
    tokens = 32768
    per_layer = model.kv_bytes_per_layer(tokens)
    total = model.total_kv_bytes(tokens)
    assert per_layer * model.num_layers == total
    print(f"  [PASS] test_kv_bytes_per_layer ({per_layer/1e6:.0f} MB/layer × 80 = {total/1e9:.1f} GB)")


# ---------------------------------------------------------------------------
# 2. TRANSFER TIME PHYSICS — must match hardware spec
# ---------------------------------------------------------------------------

def test_transfer_time_ordering():
    """GPU must be fastest tier, then CXL, then RAM, then SSD."""
    mgr = MemoryTierManager()
    size = 1024**3  # 1 GB

    gpu_t = mgr.get_tier("gpu").transfer_time_ms(size)
    cxl_t = mgr.get_tier("cxl").transfer_time_ms(size)
    ram_t = mgr.get_tier("ram").transfer_time_ms(size)
    ssd_t = mgr.get_tier("ssd").transfer_time_ms(size)

    assert gpu_t < cxl_t < ram_t < ssd_t, (
        f"Tier ordering broken: GPU={gpu_t:.1f}ms CXL={cxl_t:.1f}ms "
        f"RAM={ram_t:.1f}ms SSD={ssd_t:.1f}ms"
    )
    print(f"  [PASS] test_transfer_time_ordering "
          f"({gpu_t:.1f} < {cxl_t:.1f} < {ram_t:.1f} < {ssd_t:.1f} ms for 1 GB)")


def test_gpu_transfer_time_h100():
    """GPU HBM3 at 3000 GB/s: 1 GB should take ~0.33 ms + 1 µs base = ~0.33 ms."""
    gpu = MemoryTierManager().get_tier("gpu")
    size = 1024**3  # 1 GB
    expected_ms = (1.0 / 3000.0) * 1000.0 + 1.0 / 1000.0  # ~0.334 ms
    computed_ms = gpu.transfer_time_ms(size)
    rel_error = abs(computed_ms - expected_ms) / expected_ms
    assert rel_error < TOLERANCE, (
        f"GPU transfer time: expected {expected_ms:.3f}ms, got {computed_ms:.3f}ms"
    )
    print(f"  [PASS] test_gpu_transfer_time_h100 ({computed_ms:.3f} ms for 1 GB)")


def test_cxl_transfer_time_64gbps():
    """CXL at 64 GB/s (Samsung CMM-D): 1 GB should take ~15.6 ms."""
    cxl = MemoryTierManager().get_tier("cxl")
    size = 1024**3  # 1 GB
    expected_ms = (1.0 / 64.0) * 1000.0 + 5.0 / 1000.0  # ~15.63 ms
    computed_ms = cxl.transfer_time_ms(size)
    rel_error = abs(computed_ms - expected_ms) / expected_ms
    assert rel_error < TOLERANCE, (
        f"CXL transfer time: expected {expected_ms:.3f}ms, got {computed_ms:.3f}ms"
    )
    # Also assert CXL is within realistic hardware bounds (10–30 ms/GB)
    assert 10.0 <= computed_ms <= 30.0, (
        f"CXL 1 GB transfer {computed_ms:.1f}ms is outside realistic hardware range [10,30]ms"
    )
    print(f"  [PASS] test_cxl_transfer_time_64gbps ({computed_ms:.2f} ms for 1 GB)")


def test_ssd_transfer_time_gen4():
    """NVMe Gen4 at 7 GB/s: 1 GB should take ~143 ms."""
    ssd = MemoryTierManager().get_tier("ssd")
    size = 1024**3  # 1 GB
    expected_ms = (1.0 / 7.0) * 1000.0 + 100.0 / 1000.0  # ~143.0 ms
    computed_ms = ssd.transfer_time_ms(size)
    rel_error = abs(computed_ms - expected_ms) / expected_ms
    assert rel_error < TOLERANCE, (
        f"SSD transfer time: expected {expected_ms:.1f}ms, got {computed_ms:.1f}ms"
    )
    # Must be within realistic NVMe Gen4 range (100–200 ms/GB)
    assert 100.0 <= computed_ms <= 200.0, (
        f"SSD 1 GB transfer {computed_ms:.1f}ms outside realistic range [100,200]ms"
    )
    print(f"  [PASS] test_ssd_transfer_time_gen4 ({computed_ms:.1f} ms for 1 GB)")


def test_transfer_time_scales_linearly_with_size():
    """Transfer time must be monotonically increasing with size."""
    cxl = MemoryTierManager().get_tier("cxl")
    sizes = [1024**2, 10 * 1024**2, 100 * 1024**2, 1024**3, 5 * 1024**3]  # 1MB to 5GB
    times = [cxl.transfer_time_ms(s) for s in sizes]
    for i in range(len(times) - 1):
        assert times[i] < times[i + 1], (
            f"Transfer time not monotone at index {i}: {times[i]:.3f} >= {times[i+1]:.3f}"
        )
    print(f"  [PASS] test_transfer_time_scales_linearly_with_size")


def test_transfer_time_nonnegative():
    """Transfer time must never be negative."""
    mgr = MemoryTierManager()
    for tier_name in ["gpu", "cxl", "ram", "ssd"]:
        t = mgr.get_tier(tier_name).transfer_time_ms(1)  # 1 byte minimum
        assert t >= 0, f"Negative transfer time for {tier_name}: {t}"
    print(f"  [PASS] test_transfer_time_nonnegative")


# ---------------------------------------------------------------------------
# 3. COMPRESSION RATIOS
# ---------------------------------------------------------------------------

def test_compression_fp16_to_fp8():
    """FP16 → FP8 must halve the block size."""
    block = KVBlock(block_id="b1", size_bytes=1024 * 1024, dtype="fp16")
    expected = 512 * 1024
    assert block.compressed_size("fp8") == expected, (
        f"FP16→FP8 should halve size: expected {expected}, got {block.compressed_size('fp8')}"
    )
    print(f"  [PASS] test_compression_fp16_to_fp8 (1 MB → 512 KB)")


def test_compression_fp16_to_int4():
    """FP16 → INT4 must quarter the block size."""
    block = KVBlock(block_id="b2", size_bytes=4 * 1024 * 1024, dtype="fp16")
    expected = 1024 * 1024
    assert block.compressed_size("int4") == expected, (
        f"FP16→INT4 should quarter size: expected {expected}, got {block.compressed_size('int4')}"
    )
    print(f"  [PASS] test_compression_fp16_to_int4 (4 MB → 1 MB)")


def test_compression_fp8_to_int4():
    """FP8 → INT4 must halve the block size again."""
    block = KVBlock(block_id="b3", size_bytes=512 * 1024, dtype="fp8")
    expected = 256 * 1024
    assert block.compressed_size("int4") == expected, (
        f"FP8→INT4 should halve: expected {expected}, got {block.compressed_size('int4')}"
    )
    print(f"  [PASS] test_compression_fp8_to_int4 (512 KB → 256 KB)")


def test_compression_ratios_consistent():
    """COMPRESSION_RATIOS dict must satisfy FP16 > FP8 > INT4."""
    assert COMPRESSION_RATIOS["fp16"] > COMPRESSION_RATIOS["fp8"] > COMPRESSION_RATIOS["int4"]
    assert COMPRESSION_RATIOS["fp16"] == 1.0, "fp16 should be the baseline (1.0)"
    assert COMPRESSION_RATIOS["fp8"] == 0.5, "fp8 should be 0.5 (half the bits)"
    assert COMPRESSION_RATIOS["int4"] == 0.25, "int4 should be 0.25 (quarter the bits)"
    print(f"  [PASS] test_compression_ratios_consistent (fp16=1.0, fp8=0.5, int4=0.25)")


def test_compression_monotone_across_tiers():
    """Blocks should get smaller as they move from GPU → CXL → SSD."""
    block = KVBlock(block_id="b4", size_bytes=1024 * 1024, dtype="fp16")
    size_gpu = block.size_bytes
    size_cxl = block.compressed_size("fp8")
    size_ssd = block.compressed_size("int4")
    assert size_gpu >= size_cxl >= size_ssd, (
        f"Size should decrease: GPU={size_gpu} CXL={size_cxl} SSD={size_ssd}"
    )
    print(f"  [PASS] test_compression_monotone_across_tiers")


# ---------------------------------------------------------------------------
# 4. COST FORMULA
# ---------------------------------------------------------------------------

def test_cost_ordering():
    """Cost per GB per hour must decrease from GPU → CXL → RAM → SSD."""
    mgr = MemoryTierManager()
    gpu_cost = mgr.get_tier("gpu").cost_per_gb_hour
    cxl_cost = mgr.get_tier("cxl").cost_per_gb_hour
    ram_cost = mgr.get_tier("ram").cost_per_gb_hour
    ssd_cost = mgr.get_tier("ssd").cost_per_gb_hour
    assert gpu_cost > cxl_cost > ram_cost > ssd_cost, (
        f"Cost ordering broken: GPU={gpu_cost:.4f} CXL={cxl_cost:.4f} "
        f"RAM={ram_cost:.4f} SSD={ssd_cost:.4f} $/GB/hr"
    )
    print(f"  [PASS] test_cost_ordering "
          f"(GPU>${gpu_cost:.4f} > CXL>${cxl_cost:.4f} > RAM>${ram_cost:.4f} > SSD>${ssd_cost:.5f})")


def test_gpu_cost_per_gb_hour():
    """GPU cost: $40/GB/month ÷ 720 hours/month = $0.0556/GB/hr."""
    gpu = MemoryTierManager().get_tier("gpu")
    expected = 40.0 / 720.0
    assert abs(gpu.cost_per_gb_hour - expected) < 0.0001, (
        f"GPU cost/GB/hr: expected {expected:.4f}, got {gpu.cost_per_gb_hour:.4f}"
    )
    # 10 GB should cost $0.556/hr (is in the test_tier_manager test too)
    cost_10gb = 10 * gpu.cost_per_gb_hour
    assert abs(cost_10gb - 0.5556) < 0.001, f"10 GB GPU cost: expected $0.556, got ${cost_10gb:.4f}"
    print(f"  [PASS] test_gpu_cost_per_gb_hour (${gpu.cost_per_gb_hour:.4f}/GB/hr)")


def test_cost_per_gb_positive():
    """All tier costs must be strictly positive."""
    mgr = MemoryTierManager()
    for name in ["gpu", "cxl", "ram", "ssd"]:
        cost = mgr.get_tier(name).cost_per_gb_hour
        assert cost > 0, f"Tier {name} has non-positive cost: {cost}"
    print(f"  [PASS] test_cost_per_gb_positive")


# ---------------------------------------------------------------------------
# 5. RECOMPUTE LATENCY FORMULA
# ---------------------------------------------------------------------------

def test_recompute_latency_64k():
    """64K tokens at 40K tokens/s should take exactly 1638.4 ms."""
    from nemorix.simulation.runner import SimulationRunner, SimulationConfig
    runner = SimulationRunner()
    latency = runner._recompute_latency_ms(65536)
    expected = (65536 / RECOMPUTE_TOKENS_PER_SEC) * 1000.0
    assert abs(latency - expected) < 0.001, (
        f"Recompute 64K: expected {expected:.1f}ms, got {latency:.1f}ms"
    )
    # Must be above SLA threshold (200ms) — this is the whole point
    assert latency > SLA_THRESHOLD_MS, (
        f"Recompute latency {latency:.0f}ms should exceed SLA ({SLA_THRESHOLD_MS}ms)"
    )
    print(f"  [PASS] test_recompute_latency_64k ({latency:.0f} ms)")


def test_recompute_latency_bounded_by_hardware():
    """Recompute throughput must be between 20K and 200K tokens/s (H100 bounds)."""
    assert 20_000 <= RECOMPUTE_TOKENS_PER_SEC <= 200_000, (
        f"RECOMPUTE_TOKENS_PER_SEC={RECOMPUTE_TOKENS_PER_SEC} outside plausible H100 range"
    )
    print(f"  [PASS] test_recompute_latency_bounded_by_hardware "
          f"({RECOMPUTE_TOKENS_PER_SEC:,.0f} tokens/s)")


def test_recompute_scales_with_tokens():
    """Recompute for 64K tokens must be 2x longer than for 32K tokens."""
    from nemorix.simulation.runner import SimulationRunner
    runner = SimulationRunner()
    lat_64k = runner._recompute_latency_ms(65536)
    lat_32k = runner._recompute_latency_ms(32768)
    ratio = lat_64k / lat_32k
    assert abs(ratio - 2.0) < 0.0001, (
        f"64K recompute should be 2x 32K: ratio={ratio:.4f}"
    )
    print(f"  [PASS] test_recompute_scales_with_tokens (64K={lat_64k:.0f}ms = 2×32K={lat_32k:.0f}ms)")


# ---------------------------------------------------------------------------
# 6. EVICTION POLICY CORRECTNESS
# ---------------------------------------------------------------------------

def _make_block(bid, agent_id="a1", layer=0, attention=0.5, accessed=0.0, tokens=1024):
    return KVBlock(
        block_id=bid, agent_id=agent_id, layer_idx=layer,
        num_tokens=tokens, size_bytes=tokens * 256, dtype="fp16",
        attention_score=attention, last_accessed=accessed,
    )


def test_eviction_score_in_unit_interval():
    """Semantic eviction score must be a probability-like value in [0, 1]."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("a1", 5)

    test_cases = [
        _make_block("b1", attention=0.0, accessed=0.0),
        _make_block("b2", attention=1.0, accessed=1000.0),
        _make_block("b3", attention=0.5, accessed=100.0, layer=79),
        _make_block("b4", attention=0.95, accessed=0.1, tokens=65536),
    ]
    for b in test_cases:
        score = policy.eviction_score(b, current_time=1000.0)
        assert 0.0 <= score <= 1.0, (
            f"Score out of [0,1] for block {b.block_id}: {score:.4f}"
        )
    print(f"  [PASS] test_eviction_score_in_unit_interval")


def test_lru_always_frees_enough_bytes():
    """LRU select_victims must always free >= required_bytes."""
    policy = LRUEvictionPolicy()
    blocks = [_make_block(f"b{i}", accessed=float(i), tokens=4096) for i in range(20)]
    total_available = sum(b.size_bytes for b in blocks)

    for required in [1, 100 * 1024, total_available // 2, total_available]:
        victims = policy.select_victims(blocks, required, current_time=999.0)
        freed = sum(v.size_bytes for v in victims)
        assert freed >= required, (
            f"LRU freed {freed} bytes but needed {required}"
        )
    print(f"  [PASS] test_lru_always_frees_enough_bytes")


def test_semantic_always_frees_enough_bytes():
    """Semantic select_victims must always free >= required_bytes."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("a1", 5)
    blocks = [_make_block(f"b{i}", attention=i / 20, accessed=float(i)) for i in range(20)]
    total_available = sum(b.size_bytes for b in blocks)

    for required in [1, 100 * 1024, total_available // 2, total_available]:
        victims = policy.select_victims(blocks, required, current_time=999.0)
        freed = sum(v.size_bytes for v in victims)
        assert freed >= required, (
            f"Semantic freed {freed} bytes but needed {required}"
        )
    print(f"  [PASS] test_semantic_always_frees_enough_bytes")


def test_semantic_score_weights_sum_to_one():
    """The four semantic weight components must sum to 1.0."""
    p = SemanticEvictionPolicy()
    total = p.w_recency + p.w_importance + p.w_priority + p.w_recompute
    assert abs(total - 1.0) < 1e-9, (
        f"Semantic weights sum to {total}, expected 1.0"
    )
    print(f"  [PASS] test_semantic_score_weights_sum_to_one ({total:.1f})")


def test_high_priority_agent_scores_higher():
    """Priority-1 block should have higher eviction score than priority-9 block (harder to evict)."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("critical", 1)
    policy.set_agent_priority("background", 9)

    # Identical blocks except agent_id
    b_crit = _make_block("c", agent_id="critical", attention=0.5, accessed=50.0)
    b_bg = _make_block("bg", agent_id="background", attention=0.5, accessed=50.0)

    score_crit = policy.eviction_score(b_crit, 100.0)
    score_bg = policy.eviction_score(b_bg, 100.0)

    assert score_crit > score_bg, (
        f"Critical agent (prio=1) should score higher than background (prio=9): "
        f"{score_crit:.4f} vs {score_bg:.4f}"
    )
    print(f"  [PASS] test_high_priority_agent_scores_higher "
          f"(critical={score_crit:.3f} > background={score_bg:.3f})")


def test_deep_layer_scores_higher_than_shallow():
    """Layer 79 should score higher than layer 0 (deeper = more expensive to recompute)."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("a1", 5)
    shallow = _make_block("shallow", layer=0, attention=0.5, accessed=50.0)
    deep = _make_block("deep", layer=79, attention=0.5, accessed=50.0)

    score_shallow = policy.eviction_score(shallow, 100.0)
    score_deep = policy.eviction_score(deep, 100.0)

    assert score_deep > score_shallow, (
        f"Deep layer should score higher: deep={score_deep:.4f} shallow={score_shallow:.4f}"
    )
    print(f"  [PASS] test_deep_layer_scores_higher_than_shallow "
          f"(layer79={score_deep:.3f} > layer0={score_shallow:.3f})")


# ---------------------------------------------------------------------------
# 7. WORKLOAD GENERATOR
# ---------------------------------------------------------------------------

def test_workload_agent_sizes_in_expected_range():
    """All agents in a 64K-token workload should be between 10 GB and 22 GB."""
    model = ModelConfig("Llama-3-70B", 80, 8, 128, 2)
    wg = WorkloadGenerator(model=model, seed=42)
    agents = wg.create_workload(50, 65536)

    min_size_gb = min(a.total_size_bytes for a in agents) / 1e9
    max_size_gb = max(a.total_size_bytes for a in agents) / 1e9

    assert min_size_gb >= 10.0, f"Smallest agent is {min_size_gb:.1f} GB (expected >= 10 GB)"
    assert max_size_gb <= 22.0, f"Largest agent is {max_size_gb:.1f} GB (expected <= 22 GB)"
    print(f"  [PASS] test_workload_agent_sizes_in_expected_range "
          f"({min_size_gb:.1f}–{max_size_gb:.1f} GB)")


def test_workload_activation_probs_in_range():
    """All activation probabilities must be in [0.02, 0.25] per step."""
    wg = WorkloadGenerator(seed=42)
    agents = wg.create_workload(100, 32768)
    for a in agents:
        assert 0.02 <= a.activation_probability <= 0.25, (
            f"Activation prob {a.activation_probability:.4f} out of [0.02, 0.25]"
        )
    print(f"  [PASS] test_workload_activation_probs_in_range")


def test_workload_priority_in_range():
    """All agent priorities must be integers in [1, 9]."""
    wg = WorkloadGenerator(seed=42)
    agents = wg.create_workload(100, 32768)
    for a in agents:
        assert 1 <= a.priority <= 9, f"Priority {a.priority} out of [1, 9]"
    print(f"  [PASS] test_workload_priority_in_range")


def test_workload_unique_agent_ids():
    """All agent IDs must be unique."""
    wg = WorkloadGenerator(seed=42)
    agents = wg.create_workload(100, 32768)
    ids = [a.agent_id for a in agents]
    assert len(ids) == len(set(ids)), "Duplicate agent IDs found in workload"
    print(f"  [PASS] test_workload_unique_agent_ids (100 unique IDs)")


def test_workload_blocks_match_agent_token_count():
    """Each block's num_tokens should match the agent's total_context_tokens."""
    wg = WorkloadGenerator(seed=42)
    agents = wg.create_workload(10, 65536)
    for agent in agents:
        for block in agent.blocks:
            assert block.num_tokens == agent.total_context_tokens, (
                f"Block {block.block_id} tokens ({block.num_tokens}) != "
                f"agent tokens ({agent.total_context_tokens})"
            )
    print(f"  [PASS] test_workload_blocks_match_agent_token_count")


def test_workload_correct_number_of_blocks():
    """Each agent should have exactly num_layers blocks."""
    model = ModelConfig("Llama-3-70B", 80, 8, 128, 2)
    wg = WorkloadGenerator(model=model, seed=42)
    agents = wg.create_workload(10, 65536)
    for agent in agents:
        assert len(agent.blocks) == model.num_layers, (
            f"Agent has {len(agent.blocks)} blocks, expected {model.num_layers}"
        )
    print(f"  [PASS] test_workload_correct_number_of_blocks (80 blocks/agent)")


# ---------------------------------------------------------------------------
# 8. TIER MANAGER CAPACITY
# ---------------------------------------------------------------------------

def test_tier_capacity_exact_bytes():
    """Tier capacities must be exactly N × 1024³ bytes (not rounded)."""
    mgr = MemoryTierManager(gpu_gb=80, cxl_gb=512, ram_gb=256, ssd_gb=4000)
    assert mgr.get_tier("gpu").capacity_bytes == 80 * 1024**3
    assert mgr.get_tier("cxl").capacity_bytes == 512 * 1024**3
    assert mgr.get_tier("ram").capacity_bytes == 256 * 1024**3
    assert mgr.get_tier("ssd").capacity_bytes == 4000 * 1024**3
    print(f"  [PASS] test_tier_capacity_exact_bytes")


def test_can_fit_returns_false_when_full():
    """Tier.can_fit() must return False when not enough free space."""
    mgr = MemoryTierManager(gpu_gb=0.001)  # tiny GPU
    gpu = mgr.get_tier("gpu")
    gpu.used_bytes = gpu.capacity_bytes  # fill it completely
    assert not gpu.can_fit(1), "Full tier should not accept any allocation"
    print(f"  [PASS] test_can_fit_returns_false_when_full")


def test_utilization_capped_at_one():
    """Tier.utilization must not exceed 1.0 under normal operation."""
    mgr = MemoryTierManager(gpu_gb=1)
    gpu = mgr.get_tier("gpu")
    gpu.used_bytes = gpu.capacity_bytes
    assert gpu.utilization <= 1.0, f"Utilization = {gpu.utilization:.4f} > 1.0"
    print(f"  [PASS] test_utilization_capped_at_one")


def test_allocate_and_release_symmetry():
    """Allocating then releasing a block must return used_bytes to its original value."""
    mgr = MemoryTierManager(gpu_gb=1)
    gpu = mgr.get_tier("gpu")
    before = gpu.used_bytes

    gpu.allocate("test_block", 1024 * 1024)
    assert gpu.used_bytes == before + 1024 * 1024

    gpu.release("test_block", 1024 * 1024)
    assert gpu.used_bytes == before, (
        f"After alloc+release, used_bytes={gpu.used_bytes} != original {before}"
    )
    print(f"  [PASS] test_allocate_and_release_symmetry")


# ---------------------------------------------------------------------------
# 9. METRICS CALCULATIONS
# ---------------------------------------------------------------------------

def test_avg_resume_latency_empty():
    """avg_resume_latency_ms should return 0.0 when there are no resumes."""
    from nemorix.simulation.runner import SimulationMetrics
    m = SimulationMetrics()
    assert m.avg_resume_latency_ms == 0.0
    print(f"  [PASS] test_avg_resume_latency_empty")


def test_sla_agents_counts_correctly():
    """sla_agents should count only agents whose avg latency < SLA_THRESHOLD_MS."""
    from nemorix.simulation.runner import SimulationMetrics
    m = SimulationMetrics()
    m.agent_latencies = {
        "fast_agent_1": [10.0, 20.0, 15.0],    # avg=15ms → under SLA
        "fast_agent_2": [50.0, 80.0],            # avg=65ms → under SLA
        "slow_agent_1": [500.0, 600.0],          # avg=550ms → over SLA
        "slow_agent_2": [3000.0],                # avg=3000ms → over SLA
    }
    assert m.sla_agents == 2, f"Expected 2 SLA agents, got {m.sla_agents}"
    print(f"  [PASS] test_sla_agents_counts_correctly (2/4 under {SLA_THRESHOLD_MS}ms SLA)")


def test_p50_and_p99_ordering():
    """P50 must be <= P99."""
    from nemorix.simulation.runner import SimulationMetrics
    m = SimulationMetrics()
    m.resume_latencies = [float(x) for x in range(1, 101)]  # 1..100 ms
    assert m.p50_resume_latency_ms <= m.p99_resume_latency_ms, (
        f"P50={m.p50_resume_latency_ms} > P99={m.p99_resume_latency_ms}"
    )
    # For 1..100, P50 should be around 50 and P99 should be >= 99
    assert 49 <= m.p50_resume_latency_ms <= 51
    assert m.p99_resume_latency_ms >= 99
    print(f"  [PASS] test_p50_and_p99_ordering (P50={m.p50_resume_latency_ms}, P99={m.p99_resume_latency_ms})")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # KV-cache size
        test_kv_cache_size_llama70b,
        test_kv_cache_fits_in_80gb_gpu,
        test_kv_bytes_per_layer,
        # Transfer time physics
        test_transfer_time_ordering,
        test_gpu_transfer_time_h100,
        test_cxl_transfer_time_64gbps,
        test_ssd_transfer_time_gen4,
        test_transfer_time_scales_linearly_with_size,
        test_transfer_time_nonnegative,
        # Compression
        test_compression_fp16_to_fp8,
        test_compression_fp16_to_int4,
        test_compression_fp8_to_int4,
        test_compression_ratios_consistent,
        test_compression_monotone_across_tiers,
        # Cost
        test_cost_ordering,
        test_gpu_cost_per_gb_hour,
        test_cost_per_gb_positive,
        # Recompute latency
        test_recompute_latency_64k,
        test_recompute_latency_bounded_by_hardware,
        test_recompute_scales_with_tokens,
        # Eviction
        test_eviction_score_in_unit_interval,
        test_lru_always_frees_enough_bytes,
        test_semantic_always_frees_enough_bytes,
        test_semantic_score_weights_sum_to_one,
        test_high_priority_agent_scores_higher,
        test_deep_layer_scores_higher_than_shallow,
        # Workload generator
        test_workload_agent_sizes_in_expected_range,
        test_workload_activation_probs_in_range,
        test_workload_priority_in_range,
        test_workload_unique_agent_ids,
        test_workload_blocks_match_agent_token_count,
        test_workload_correct_number_of_blocks,
        # Tier capacity
        test_tier_capacity_exact_bytes,
        test_can_fit_returns_false_when_full,
        test_utilization_capped_at_one,
        test_allocate_and_release_symmetry,
        # Metrics
        test_avg_resume_latency_empty,
        test_sla_agents_counts_correctly,
        test_p50_and_p99_ordering,
    ]

    print("Running accuracy tests...")
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
    else:
        print("All accuracy tests passed!")
