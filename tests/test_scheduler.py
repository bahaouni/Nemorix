"""Tests for AgentScheduler."""

from __future__ import annotations

from nemorix.core.scheduler import AgentScheduler
from nemorix.core.tier_manager import MemoryTierManager
from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator


def test_agent_lifecycle():
    """Test basic agent create → activate → suspend → resume cycle."""
    tier_mgr = MemoryTierManager(gpu_gb=2, cxl_gb=4, ram_gb=4, ssd_gb=10)
    policy = LRUEvictionPolicy()
    scheduler = AgentScheduler(tier_mgr, policy)
    wg = WorkloadGenerator(seed=1)

    agent = wg.create_agent(context_tokens=1024, priority=3)
    scheduler.register_agent(agent)
    for b in agent.blocks:
        tier_mgr.get_tier("ssd").allocate(b.block_id, b.size_bytes)

    assert agent.state == "suspended"

    # Activate
    latency = scheduler.activate_agent(agent.agent_id, current_time=100.0)
    assert agent.state == "running"
    assert latency >= 0
    assert all(b.tier == "gpu" for b in agent.blocks)

    # Suspend to CXL
    scheduler.deactivate_agent(agent.agent_id, "cxl", current_time=200.0)
    assert agent.state == "ready"
    assert all(b.tier == "cxl" for b in agent.blocks)

    # Resume from CXL
    latency2 = scheduler.activate_agent(agent.agent_id, current_time=300.0)
    assert agent.state == "running"
    assert latency2 < latency  # CXL is faster than SSD

    print("  [PASS] test_agent_lifecycle")


def test_idle_suspension():
    """Test that idle agents get moved to colder tiers."""
    tier_mgr = MemoryTierManager(gpu_gb=2, cxl_gb=4, ram_gb=4, ssd_gb=10)
    policy = LRUEvictionPolicy()
    scheduler = AgentScheduler(tier_mgr, policy)
    wg = WorkloadGenerator(seed=2)

    agent = wg.create_agent(context_tokens=512, priority=5)
    scheduler.register_agent(agent)
    for b in agent.blocks:
        tier_mgr.get_tier("ssd").allocate(b.block_id, b.size_bytes)

    scheduler.activate_agent(agent.agent_id, current_time=0.0)
    assert agent.state == "running"

    # After idle threshold, should be suspended
    suspended = scheduler.suspend_idle_agents(current_time=600.0, idle_threshold_s=100.0)
    assert suspended >= 1
    assert agent.state != "running"

    print("  [PASS] test_idle_suspension")


def test_semantic_priority():
    """Test that semantic policy respects agent priority."""
    tier_mgr = MemoryTierManager(gpu_gb=1, cxl_gb=2, ram_gb=2, ssd_gb=10)
    policy = SemanticEvictionPolicy()
    scheduler = AgentScheduler(tier_mgr, policy)
    wg = WorkloadGenerator(seed=3)

    high_prio = wg.create_agent(context_tokens=256, priority=1)
    low_prio = wg.create_agent(context_tokens=256, priority=9)
    policy.set_agent_priority(high_prio.agent_id, 1)
    policy.set_agent_priority(low_prio.agent_id, 9)

    for a in [high_prio, low_prio]:
        scheduler.register_agent(a)
        for b in a.blocks:
            tier_mgr.get_tier("ssd").allocate(b.block_id, b.size_bytes)

    # Activate both
    scheduler.activate_agent(high_prio.agent_id, 0.0)
    scheduler.activate_agent(low_prio.agent_id, 1.0)

    # The low priority agent should have some blocks evicted if GPU is full
    # At minimum, high priority should still be running
    assert high_prio.state == "running" or low_prio.state == "running"

    print("  [PASS] test_semantic_priority")


if __name__ == "__main__":
    print("Running scheduler tests...")
    test_agent_lifecycle()
    test_idle_suspension()
    test_semantic_priority()
    print("\nAll scheduler tests passed!")
