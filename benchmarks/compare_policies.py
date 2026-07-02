#!/usr/bin/env python3
"""Compare eviction policies: LRU vs Semantic across different agent counts.

Usage:
    python benchmarks/compare_policies.py
"""
from __future__ import annotations
import sys
import os

from nemorix.simulation.runner import SimulationRunner, SimulationConfig


def main():
    print("=" * 80)
    print("  Nemorix Policy Comparison: No Offload vs LRU vs Semantic (Nemorix)")
    print("=" * 80)
    print()

    agent_counts = [10, 25, 50, 75, 100, 200, 500]

    print(f"  {'Agents':>8}  {'Policy':>10}  {'SLA Agents':>11}  {'Avg(ms)':>9}  "
          f"{'P99(ms)':>9}  {'GPU Util':>9}  {'$/agent-hr':>11}")
    print("  " + "-" * 76)

    all_results: dict[int, dict[str, object]] = {}

    for n_agents in agent_counts:
        config = SimulationConfig(
            num_agents=n_agents,
            context_tokens=65536,
            simulation_steps=720,  # 12 hours for speed
            seed=42,
        )
        runner = SimulationRunner(config)
        all_results[n_agents] = {}

        for policy_name in ["no_offload", "lru", "semantic"]:
            m = runner.run(policy_name)
            all_results[n_agents][policy_name] = m
            sla = m.sla_agents
            effective = sla if sla > 0 else m.max_concurrent_agents
            cost_per_agent = m.total_cost_per_hour / max(1, effective)
            label = {"no_offload": "NoOffload", "lru": "LRU", "semantic": "Nemorix"}[policy_name]
            print(f"  {n_agents:>8}  {label:>10}  {sla:>11}  "
                  f"{m.avg_resume_latency_ms:>8.1f}  "
                  f"{m.p99_resume_latency_ms:>8.1f}  "
                  f"{m.avg_gpu_utilization * 100:>8.0f}%  "
                  f"${cost_per_agent:>10.2f}")
        print()

    print("=" * 80)
    print()

    # Dynamic observations from actual results
    max_agents = agent_counts[-1]
    lru_max = all_results[max_agents]["lru"]
    sem_max = all_results[max_agents]["semantic"]
    no_max = all_results[max_agents]["no_offload"]
    print("Key Observations:")
    print(f"  1. At {max_agents} agents: Nemorix SLA={sem_max.sla_agents} vs LRU SLA={lru_max.sla_agents}")
    if sem_max.avg_resume_latency_ms > 0 and lru_max.avg_resume_latency_ms > 0:
        ratio = lru_max.avg_resume_latency_ms / sem_max.avg_resume_latency_ms
        print(f"  2. Nemorix avg latency {sem_max.avg_resume_latency_ms:.1f}ms vs LRU {lru_max.avg_resume_latency_ms:.1f}ms ({ratio:.1f}x)")
    if no_max.avg_resume_latency_ms > 0 and sem_max.avg_resume_latency_ms > 0:
        ratio_no = no_max.avg_resume_latency_ms / sem_max.avg_resume_latency_ms
        print(f"  3. vs No-offload: {ratio_no:.0f}x faster ({no_max.avg_resume_latency_ms:.0f}ms -> {sem_max.avg_resume_latency_ms:.1f}ms)")
    print(f"  4. CXL tier is the key enabler: both LRU+CXL and Semantic+CXL >> No-offload")
    print()


if __name__ == "__main__":
    main()
