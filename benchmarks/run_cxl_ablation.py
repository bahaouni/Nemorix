#!/usr/bin/env python3
"""CXL ablation benchmark — isolates the contribution of the CXL tier.

Runs LRU and Semantic policies with and without CXL (CXL=0 means agents fall
directly from GPU to CPU RAM, then SSD) to quantify how much of the latency
improvement is due to CXL specifically vs just having any offload tier.

This produces the numbers for Paper §4.4 ("The CXL Effect").

Usage:
    python benchmarks/run_cxl_ablation.py [--agents 50] [--hours 24]
Output:
    cxl_ablation.json
"""
from __future__ import annotations
import argparse
import json
import os

from nemorix.simulation.runner import SimulationRunner, SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="Nemorix CXL Ablation")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("  Nemorix CXL Ablation Study  —  Paper §4.4 (The CXL Effect)")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Three configurations: with CXL (512 GB), without CXL (0), no-offload
    # ------------------------------------------------------------------
    configs = [
        ("No offload (baseline)", SimulationConfig(
            num_agents=args.agents, context_tokens=args.tokens,
            gpu_memory_gb=80, cxl_memory_gb=0.001, ram_gb=0.001, ssd_gb=0.001,
            simulation_steps=args.hours * 60, seed=args.seed)),
        ("LRU, no CXL  (GPU + RAM + SSD)", SimulationConfig(
            num_agents=args.agents, context_tokens=args.tokens,
            gpu_memory_gb=80, cxl_memory_gb=0.001, ram_gb=256, ssd_gb=4000,
            simulation_steps=args.hours * 60, seed=args.seed)),
        ("LRU + CXL   (GPU + CXL + RAM + SSD)", SimulationConfig(
            num_agents=args.agents, context_tokens=args.tokens,
            gpu_memory_gb=80, cxl_memory_gb=512, ram_gb=256, ssd_gb=4000,
            simulation_steps=args.hours * 60, seed=args.seed)),
        ("Nemorix + CXL (semantic, full hierarchy)", SimulationConfig(
            num_agents=args.agents, context_tokens=args.tokens,
            gpu_memory_gb=80, cxl_memory_gb=512, ram_gb=256, ssd_gb=4000,
            simulation_steps=args.hours * 60, seed=args.seed)),
    ]

    header = f"  {'Config':<38} {'SLA':>5} {'Avg(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'$/ag-hr':>9}"
    print(header)
    print("  " + "-" * 82)

    results = {}
    for label, cfg in configs:
        runner = SimulationRunner(cfg)
        # For no-offload and LRU-no-CXL, use LRU policy;
        # for full configs use their respective policies.
        if "Nemorix" in label:
            m = runner.run("semantic")
        else:
            m = runner.run("lru")
        sla = m.sla_agents
        effective = sla if sla > 0 else m.max_concurrent_agents
        cost = m.total_cost_per_hour / max(1, effective)
        print(f"  {label:<38} {sla:>5} {m.avg_resume_latency_ms:>8.1f} "
              f"{m.p95_resume_latency_ms:>8.1f} {m.p99_resume_latency_ms:>8.1f} "
              f"${cost:>8.2f}")
        results[label] = {
            "sla_agents": sla,
            "avg_resume_latency_ms": round(m.avg_resume_latency_ms, 2),
            "p95_resume_latency_ms": round(m.p95_resume_latency_ms, 2),
            "p99_resume_latency_ms": round(m.p99_resume_latency_ms, 2),
            "cost_per_agent_hour": round(cost, 4),
        }

    print()
    no_cxl_lru = results["LRU, no CXL  (GPU + RAM + SSD)"]
    with_cxl_lru = results["LRU + CXL   (GPU + CXL + RAM + SSD)"]
    with_cxl_sem = results["Nemorix + CXL (semantic, full hierarchy)"]
    if with_cxl_lru["avg_resume_latency_ms"] > 0:
        cxl_gain = no_cxl_lru["avg_resume_latency_ms"] / with_cxl_lru["avg_resume_latency_ms"]
        print(f"  CXL tier speedup (LRU): {cxl_gain:.1f}x  "
              f"({no_cxl_lru['avg_resume_latency_ms']:.1f} ms -> "
              f"{with_cxl_lru['avg_resume_latency_ms']:.1f} ms)")
    if with_cxl_lru["avg_resume_latency_ms"] > 0:
        sem_gain = with_cxl_lru["avg_resume_latency_ms"] / with_cxl_sem["avg_resume_latency_ms"]
        print(f"  Semantic over LRU (both + CXL): {sem_gain:.2f}x")
    print()

    out_path = os.path.join(os.path.dirname(__file__), "cxl_ablation.json")
    with open(out_path, "w") as f:
        json.dump({"agents": args.agents, "hours": args.hours, "seed": args.seed,
                   "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")
    print()
    print("  Use these numbers for Paper §4.4 (The CXL Effect).")


if __name__ == "__main__":
    main()
