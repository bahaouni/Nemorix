#!/usr/bin/env python3
"""Statistical robustness benchmark.

Runs every policy across MANY random seeds and reports mean +/- standard
deviation for each headline metric. This answers the single most common
reviewer / investor question: "Did you cherry-pick a lucky seed?"

The answer, with this script, is: "No -- here are the numbers across N seeds,
with their spread."

Usage:
    python benchmarks/run_robustness.py [--agents 50] [--seeds 10] [--hours 24]
"""
from __future__ import annotations
import argparse
import os

from nemorix.simulation.runner import SimulationRunner, SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="Nemorix Robustness Benchmark")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (uses 42, 43, ... 42+N-1)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]

    config = SimulationConfig(
        num_agents=args.agents,
        context_tokens=args.tokens,
        simulation_steps=args.hours * 60,
    )

    runner = SimulationRunner(config)

    print(f"Running robustness sweep over {len(seeds)} seeds "
          f"({args.agents} agents, {args.hours}h each)...")
    print("This runs the full simulation 3 x N times; please wait.\n")

    aggregated = runner.run_multiseed(seeds)

    if args.json:
        out = SimulationRunner.multiseed_to_json(aggregated, seeds)
        print(out)
    else:
        print(SimulationRunner.format_multiseed(aggregated, seeds, config))

    # Always save the JSON artifact for the pitch deck / paper.
    out_path = os.path.join(os.path.dirname(__file__), "robustness.json")
    with open(out_path, "w") as f:
        f.write(SimulationRunner.multiseed_to_json(aggregated, seeds))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
