#!/usr/bin/env python3
"""Main benchmark: run Nemorix simulation and print results table.

Usage:
    python benchmarks/run_simulation.py [--agents 50] [--tokens 65536] [--hours 24]
"""
from __future__ import annotations
import argparse
import sys
import os

from nemorix.simulation.runner import SimulationRunner, SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="Nemorix Benchmark Suite")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--gpu-gb", type=float, default=80)
    parser.add_argument("--cxl-gb", type=float, default=512)
    parser.add_argument("--ram-gb", type=float, default=256)
    parser.add_argument("--ssd-gb", type=float, default=4000)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    config = SimulationConfig(
        num_agents=args.agents,
        context_tokens=args.tokens,
        gpu_memory_gb=args.gpu_gb,
        cxl_memory_gb=args.cxl_gb,
        ram_gb=args.ram_gb,
        ssd_gb=args.ssd_gb,
        simulation_steps=args.hours * 60,
        seed=args.seed,
    )

    runner = SimulationRunner(config)

    print("Running Nemorix simulation...")
    print(f"  {config.num_agents} agents, {config.context_tokens // 1024}K tokens each")
    print(f"  {config.simulation_steps // 60}h simulation, seed={config.seed}")
    print()

    results = runner.run_all_policies()

    if args.json:
        print(SimulationRunner.results_to_json(results))
    else:
        print(SimulationRunner.format_results(results, config))

    # Save results to file
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        f.write(SimulationRunner.results_to_json(results))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
