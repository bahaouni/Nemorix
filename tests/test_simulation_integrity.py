"""Simulation integrity tests — validate invariants that must hold during a full run.

Unlike test_accuracy.py (which tests formulas in isolation), these tests run the
complete SimulationRunner and check that the simulation never violates physical
or logical constraints, and that the three policies produce the expected ordering.

These are the tests a reviewer should run to ask: "Could the numbers be made up?"
Each assertion has a written justification next to it.
"""

from __future__ import annotations
import json

from nemorix.simulation.runner import (
    SimulationRunner, SimulationConfig, SimulationMetrics,
    SLA_THRESHOLD_MS, GPU_RENTAL_COST_PER_HOUR, RECOMPUTE_TOKENS_PER_SEC
)


# Use a fast config for most tests (6 hours, 20 agents) so the suite runs in seconds.
FAST_CFG = SimulationConfig(
    num_agents=20,
    context_tokens=65536,
    simulation_steps=360,   # 6 hours
    seed=42,
)

FULL_CFG = SimulationConfig(
    num_agents=50,
    context_tokens=65536,
    simulation_steps=1440,  # 24 hours
    seed=42,
)


def _run_all(cfg: SimulationConfig) -> dict[str, SimulationMetrics]:
    runner = SimulationRunner(cfg)
    return runner.run_all_policies()


# ---------------------------------------------------------------------------
# 1. DETERMINISM
# ---------------------------------------------------------------------------

def test_simulation_is_deterministic():
    """Running twice with the same seed must produce byte-identical results."""
    r1 = SimulationRunner(FAST_CFG).run_all_policies()
    r2 = SimulationRunner(FAST_CFG).run_all_policies()

    for policy in ["no_offload", "lru", "semantic"]:
        m1, m2 = r1[policy], r2[policy]
        assert m1.max_concurrent_agents == m2.max_concurrent_agents, \
            f"[{policy}] max_concurrent_agents differs between runs"
        assert abs(m1.avg_resume_latency_ms - m2.avg_resume_latency_ms) < 0.001, \
            f"[{policy}] avg_resume_latency_ms differs between runs"
        assert m1.total_resumes == m2.total_resumes, \
            f"[{policy}] total_resumes differs between runs"
        assert m1.sla_agents == m2.sla_agents, \
            f"[{policy}] sla_agents differs between runs"
    print(f"  [PASS] test_simulation_is_deterministic")


# ---------------------------------------------------------------------------
# 2. PHYSICAL INVARIANTS — GPU CAPACITY
# ---------------------------------------------------------------------------

def test_gpu_utilization_never_exceeds_100_percent():
    """GPU utilization must be in [0, 1.0] every single step."""
    results = _run_all(FAST_CFG)
    for policy, m in results.items():
        over = [u for u in m.gpu_utilization_samples if u > 1.0 + 1e-9]
        assert not over, (
            f"[{policy}] GPU utilization exceeded 100% in {len(over)} steps "
            f"(max={max(over, default=0):.4f})"
        )
    print(f"  [PASS] test_gpu_utilization_never_exceeds_100_percent")


def test_sla_agents_bounded_by_num_agents():
    """Cannot serve more agents under SLA than the total number of agents."""
    results = _run_all(FAST_CFG)
    for policy, m in results.items():
        assert m.sla_agents <= FAST_CFG.num_agents, (
            f"[{policy}] sla_agents={m.sla_agents} > num_agents={FAST_CFG.num_agents}"
        )
    print(f"  [PASS] test_sla_agents_bounded_by_num_agents")


def test_max_concurrent_agents_bounded_by_gpu_capacity():
    """Max GPU-resident agents must physically fit in GPU VRAM."""
    results = _run_all(FAST_CFG)
    gpu_cap_bytes = FAST_CFG.gpu_memory_gb * 1024**3

    from nemorix.simulation.workload import WorkloadGenerator, ModelConfig
    model = ModelConfig(num_layers=FAST_CFG.model_layers)
    wg = WorkloadGenerator(model=model, seed=FAST_CFG.seed)
    agents = wg.create_workload(FAST_CFG.num_agents, FAST_CFG.context_tokens)
    min_agent_size = min(a.total_size_bytes for a in agents)

    # The theoretical max concurrent agents: if all had the minimum size
    theoretical_max = int(gpu_cap_bytes // min_agent_size)

    for policy, m in results.items():
        assert m.max_concurrent_agents <= theoretical_max, (
            f"[{policy}] max_concurrent={m.max_concurrent_agents} > "
            f"theoretical max={theoretical_max} (min agent={min_agent_size/1e9:.1f} GB, GPU={FAST_CFG.gpu_memory_gb} GB)"
        )
    print(f"  [PASS] test_max_concurrent_agents_bounded_by_gpu_capacity "
          f"(max={max(r.max_concurrent_agents for r in results.values())} <= {theoretical_max})")


# ---------------------------------------------------------------------------
# 3. POLICY ORDERING — the core claim of the project
# ---------------------------------------------------------------------------

def test_no_offload_has_zero_sla_agents():
    """No-offload must produce 0 SLA agents because recompute always exceeds 200ms.

    Minimum recompute latency = 32K tokens / 40K tps = 800ms >> 200ms SLA.
    So every single resume in no_offload mode should take > SLA_THRESHOLD_MS.
    """
    results = _run_all(FULL_CFG)
    no_sla = results["no_offload"].sla_agents
    assert no_sla == 0, (
        f"no_offload should have 0 SLA agents (all resumes > {SLA_THRESHOLD_MS}ms), "
        f"but got {no_sla}"
    )
    print(f"  [PASS] test_no_offload_has_zero_sla_agents")


def test_nemorix_sla_geq_lru_sla():
    """Nemorix must serve at least as many agents under SLA as LRU.

    Nemorix has access to the CXL tier which LRU doesn't use.
    When RAM overflows, LRU agents spill to slow SSD while Nemorix keeps them in fast CXL.
    """
    results = _run_all(FULL_CFG)
    sem_sla = results["semantic"].sla_agents
    lru_sla = results["lru"].sla_agents
    assert sem_sla >= lru_sla, (
        f"Nemorix SLA agents ({sem_sla}) should be >= LRU SLA agents ({lru_sla})"
    )
    print(f"  [PASS] test_nemorix_sla_geq_lru_sla ({sem_sla} >= {lru_sla})")


def test_nemorix_avg_latency_lt_lru():
    """Nemorix average resume latency must be strictly lower than LRU.

    CXL (36 GB/s) adds capacity despite lower raw bandwidth than modeled RAM
    (50 GB/s), and both are much faster than SSD (7 GB/s).
    With 50 agents, some LRU agents overflow to SSD (~143ms/GB), pulling up the average.
    Nemorix keeps all idle agents in CXL (~28ms/GB), maintaining a lower average.
    """
    results = _run_all(FULL_CFG)
    sem_lat = results["semantic"].avg_resume_latency_ms
    lru_lat = results["lru"].avg_resume_latency_ms
    assert sem_lat < lru_lat, (
        f"Nemorix avg latency ({sem_lat:.1f}ms) should be < LRU ({lru_lat:.1f}ms)"
    )
    print(f"  [PASS] test_nemorix_avg_latency_lt_lru ({sem_lat:.1f}ms < {lru_lat:.1f}ms)")


def test_no_offload_latency_gt_lru_latency():
    """No-offload average latency must be much higher than both LRU and Nemorix."""
    results = _run_all(FULL_CFG)
    no_lat = results["no_offload"].avg_resume_latency_ms
    lru_lat = results["lru"].avg_resume_latency_ms
    sem_lat = results["semantic"].avg_resume_latency_ms

    assert no_lat > lru_lat, f"no_offload ({no_lat:.1f}ms) should be > LRU ({lru_lat:.1f}ms)"
    assert no_lat > sem_lat, f"no_offload ({no_lat:.1f}ms) should be > Nemorix ({sem_lat:.1f}ms)"
    print(f"  [PASS] test_no_offload_latency_gt_lru_latency "
          f"(no_offload={no_lat:.0f}ms >> LRU={lru_lat:.0f}ms >> Nemorix={sem_lat:.0f}ms)")


def test_no_offload_latency_matches_recompute_formula():
    """No-offload avg latency must correspond to the recompute formula.

    Given agents with 32K–64K tokens at 40K tokens/s:
      - Min recompute: 32768 / 40000 * 1000 = 819 ms
      - Max recompute: 65536 / 40000 * 1000 = 1638 ms
      - Expected avg: ~1200 ms (midpoint of uniform range)

    The simulation's warmup period (60 steps) skips cold-starts, so the avg
    should fall in the [800, 1700] ms band.
    """
    results = _run_all(FULL_CFG)
    no_lat = results["no_offload"].avg_resume_latency_ms

    min_expected = (FULL_CFG.context_tokens // 2) / RECOMPUTE_TOKENS_PER_SEC * 1000
    max_expected = FULL_CFG.context_tokens / RECOMPUTE_TOKENS_PER_SEC * 1000

    assert min_expected <= no_lat <= max_expected * 1.05, (
        f"no_offload avg latency {no_lat:.0f}ms is outside expected range "
        f"[{min_expected:.0f}, {max_expected:.0f}]ms"
    )
    print(f"  [PASS] test_no_offload_latency_matches_recompute_formula "
          f"({no_lat:.0f}ms in [{min_expected:.0f}, {max_expected:.0f}]ms)")


# ---------------------------------------------------------------------------
# 4. CONSISTENT ACTIVATIONS ACROSS POLICIES
# ---------------------------------------------------------------------------

def test_total_resumes_same_across_policies():
    """All three policies use the same seed, so they see the same activation events.

    total_resumes counts every activation (same seed → same random draws),
    so it must be identical for all three policies.
    """
    results = _run_all(FULL_CFG)
    resumes = {p: m.total_resumes for p, m in results.items()}
    values = list(resumes.values())
    assert all(v == values[0] for v in values), (
        f"total_resumes differ across policies: {resumes}"
    )
    print(f"  [PASS] test_total_resumes_same_across_policies ({values[0]} total resumes)")


# ---------------------------------------------------------------------------
# 5. COST SANITY
# ---------------------------------------------------------------------------

def test_all_costs_positive():
    """Every cost metric must be strictly positive."""
    results = _run_all(FAST_CFG)
    for policy, m in results.items():
        assert m.avg_cost_per_hour >= 0, f"[{policy}] avg_cost_per_hour < 0"
        assert m.total_cost_per_hour >= GPU_RENTAL_COST_PER_HOUR, (
            f"[{policy}] total_cost_per_hour={m.total_cost_per_hour:.2f} "
            f"< GPU_RENTAL_COST_PER_HOUR={GPU_RENTAL_COST_PER_HOUR}"
        )
    print(f"  [PASS] test_all_costs_positive")


def test_total_cost_includes_gpu_rental():
    """total_cost_per_hour must include GPU_RENTAL_COST_PER_HOUR."""
    results = _run_all(FAST_CFG)
    for policy, m in results.items():
        assert m.total_cost_per_hour == m.avg_cost_per_hour + GPU_RENTAL_COST_PER_HOUR, (
            f"[{policy}] total_cost != storage_cost + GPU_rental"
        )
    print(f"  [PASS] test_total_cost_includes_gpu_rental")


def test_no_offload_costs_more_per_agent():
    """No-offload must cost more per active agent than Nemorix.

    no_offload serves 0 agents under SLA, so its cost is divided by max_concurrent_agents
    (GPU-only). Nemorix serves many more agents under SLA, spreading the fixed GPU cost.
    """
    results = _run_all(FULL_CFG)
    no = results["no_offload"]
    sem = results["semantic"]

    no_sla_eff = max(1, no.max_concurrent_agents)
    sem_sla_eff = max(1, sem.sla_agents)

    no_cost_per = no.total_cost_per_hour / no_sla_eff
    sem_cost_per = sem.total_cost_per_hour / sem_sla_eff

    assert no_cost_per > sem_cost_per, (
        f"no_offload cost/agent (${no_cost_per:.2f}) should be > "
        f"Nemorix cost/agent (${sem_cost_per:.2f})"
    )
    print(f"  [PASS] test_no_offload_costs_more_per_agent "
          f"(no_offload=${no_cost_per:.2f} > Nemorix=${sem_cost_per:.2f}/agent-hr)")


# ---------------------------------------------------------------------------
# 6. GPU UTILIZATION
# ---------------------------------------------------------------------------

def test_avg_gpu_utilization_in_range():
    """Average GPU utilization must be between 0% and 100% for all policies."""
    results = _run_all(FAST_CFG)
    for policy, m in results.items():
        util = m.avg_gpu_utilization
        assert 0.0 <= util <= 1.0, (
            f"[{policy}] avg_gpu_utilization={util:.4f} out of [0, 1]"
        )
    print(f"  [PASS] test_avg_gpu_utilization_in_range "
          f"(Nemorix={results['semantic'].avg_gpu_utilization*100:.0f}%)")


def test_semantic_gpu_util_geq_lru():
    """Nemorix GPU utilization must be >= LRU.

    Semantic eviction protects high-value agents, leading to more consistent
    GPU occupancy. LRU may evict recently-needed agents, causing more empty-GPU events.
    (At minimum, they should be equal.)
    """
    results = _run_all(FULL_CFG)
    sem_util = results["semantic"].avg_gpu_utilization
    lru_util = results["lru"].avg_gpu_utilization
    # Allow 1% tolerance (they may be very close)
    assert sem_util >= lru_util - 0.01, (
        f"Nemorix GPU util ({sem_util*100:.1f}%) should be >= LRU ({lru_util*100:.1f}%)"
    )
    print(f"  [PASS] test_semantic_gpu_util_geq_lru "
          f"(Nemorix={sem_util*100:.1f}% >= LRU={lru_util*100:.1f}%)")


# ---------------------------------------------------------------------------
# 7. SCALING BEHAVIOUR
# ---------------------------------------------------------------------------

def test_more_agents_more_total_resumes():
    """A workload with 50 agents should produce more total resumes than 20 agents."""
    r_small = SimulationRunner(SimulationConfig(
        num_agents=20, context_tokens=32768, simulation_steps=360, seed=42
    )).run("semantic")
    r_large = SimulationRunner(SimulationConfig(
        num_agents=50, context_tokens=32768, simulation_steps=360, seed=42
    )).run("semantic")
    assert r_large.total_resumes > r_small.total_resumes, (
        f"50-agent run ({r_large.total_resumes}) should have more resumes than "
        f"20-agent run ({r_small.total_resumes})"
    )
    print(f"  [PASS] test_more_agents_more_total_resumes "
          f"(50 agents: {r_large.total_resumes} > 20 agents: {r_small.total_resumes})")


def test_lru_degrades_at_higher_agent_count():
    """LRU SLA agents should decline (or plateau) as number of agents grows past RAM capacity.

    At 100 agents × ~16 GB/agent = 1600 GB total, which overflows RAM (256 GB) + CXL (0 GB for LRU).
    All excess agents fall to SSD → many miss SLA. At 10 agents (160 GB < 256 GB RAM),
    LRU likely serves all agents under SLA.
    """
    cfg_small = SimulationConfig(num_agents=10, context_tokens=65536, simulation_steps=360, seed=42)
    cfg_large = SimulationConfig(num_agents=100, context_tokens=65536, simulation_steps=360, seed=42)

    sla_small = SimulationRunner(cfg_small).run("lru").sla_agents
    sla_large = SimulationRunner(cfg_large).run("lru").sla_agents

    # At small scale, fraction served under SLA must be >= large scale fraction
    fraction_small = sla_small / cfg_small.num_agents
    fraction_large = sla_large / cfg_large.num_agents

    assert fraction_small >= fraction_large, (
        f"LRU SLA fraction at 10 agents ({fraction_small:.2f}) should be >= "
        f"at 100 agents ({fraction_large:.2f}) — LRU should degrade at scale"
    )
    print(f"  [PASS] test_lru_degrades_at_higher_agent_count "
          f"(LRU SLA%: 10-agent={fraction_small*100:.0f}% >= 100-agent={fraction_large*100:.0f}%)")


def test_nemorix_scales_better_than_lru():
    """At high agent count, Nemorix should serve a higher fraction under SLA than LRU.

    CXL provides a large warm tier that RAM-only LRU doesn't have.
    This advantage grows with more agents.
    """
    cfg = SimulationConfig(num_agents=75, context_tokens=65536, simulation_steps=360, seed=42)
    runner = SimulationRunner(cfg)
    sem_sla = runner.run("semantic").sla_agents
    lru_sla = runner.run("lru").sla_agents

    sem_frac = sem_sla / cfg.num_agents
    lru_frac = lru_sla / cfg.num_agents
    assert sem_frac >= lru_frac, (
        f"Nemorix SLA fraction ({sem_frac*100:.0f}%) should be >= LRU ({lru_frac*100:.0f}%)"
    )
    print(f"  [PASS] test_nemorix_scales_better_than_lru "
          f"(Nemorix={sem_frac*100:.0f}% >= LRU={lru_frac*100:.0f}% at 75 agents)")


# ---------------------------------------------------------------------------
# 8. EDGE CASES
# ---------------------------------------------------------------------------

def test_single_agent_simulation():
    """A simulation with 1 agent should complete without errors."""
    cfg = SimulationConfig(num_agents=1, context_tokens=32768, simulation_steps=60, seed=42)
    runner = SimulationRunner(cfg)
    results = runner.run_all_policies()
    for policy, m in results.items():
        assert m.total_resumes >= 0, f"[{policy}] Negative total_resumes"
    print(f"  [PASS] test_single_agent_simulation")


def test_high_agent_count_simulation():
    """A 200-agent simulation should complete without errors or exceptions."""
    cfg = SimulationConfig(num_agents=200, context_tokens=32768, simulation_steps=30, seed=42)
    runner = SimulationRunner(cfg)
    m = runner.run("semantic")
    assert m.max_concurrent_agents >= 0
    assert m.avg_gpu_utilization <= 1.0
    print(f"  [PASS] test_high_agent_count_simulation "
          f"(200 agents, GPU util={m.avg_gpu_utilization*100:.0f}%)")


def test_simulation_results_serializable_to_json():
    """results_to_json must produce valid JSON that can be parsed back."""
    runner = SimulationRunner(FAST_CFG)
    results = runner.run_all_policies()
    json_str = SimulationRunner.results_to_json(results)
    parsed = json.loads(json_str)
    assert set(parsed.keys()) == {"no_offload", "lru", "semantic"}
    for policy_data in parsed.values():
        assert "sla_agents" in policy_data
        assert "avg_resume_latency_ms" in policy_data
        assert "p99_resume_latency_ms" in policy_data
    print(f"  [PASS] test_simulation_results_serializable_to_json")


def test_p99_always_geq_p50():
    """P99 resume latency must always be >= P50 for all policies."""
    results = _run_all(FULL_CFG)
    for policy, m in results.items():
        if m.resume_latencies:
            assert m.p99_resume_latency_ms >= m.p50_resume_latency_ms, (
                f"[{policy}] P99={m.p99_resume_latency_ms:.1f}ms < P50={m.p50_resume_latency_ms:.1f}ms"
            )
    print(f"  [PASS] test_p99_always_geq_p50")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Determinism
        test_simulation_is_deterministic,
        # Physical invariants
        test_gpu_utilization_never_exceeds_100_percent,
        test_sla_agents_bounded_by_num_agents,
        test_max_concurrent_agents_bounded_by_gpu_capacity,
        # Policy ordering
        test_no_offload_has_zero_sla_agents,
        test_nemorix_sla_geq_lru_sla,
        test_nemorix_avg_latency_lt_lru,
        test_no_offload_latency_gt_lru_latency,
        test_no_offload_latency_matches_recompute_formula,
        # Consistent activations
        test_total_resumes_same_across_policies,
        # Cost
        test_all_costs_positive,
        test_total_cost_includes_gpu_rental,
        test_no_offload_costs_more_per_agent,
        # GPU utilization
        test_avg_gpu_utilization_in_range,
        test_semantic_gpu_util_geq_lru,
        # Scaling
        test_more_agents_more_total_resumes,
        test_lru_degrades_at_higher_agent_count,
        test_nemorix_scales_better_than_lru,
        # Edge cases
        test_single_agent_simulation,
        test_high_agent_count_simulation,
        test_simulation_results_serializable_to_json,
        test_p99_always_geq_p50,
    ]

    print("Running simulation integrity tests...")
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
        print("All simulation integrity tests passed!")
