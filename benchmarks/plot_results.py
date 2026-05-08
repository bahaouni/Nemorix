#!/usr/bin/env python3
"""Generate visualisation charts from benchmark results.

Usage:
    python benchmarks/plot_results.py
    (Requires: pip install matplotlib)
"""
from __future__ import annotations
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

from nemorix.simulation.runner import SimulationRunner, SimulationConfig


def run_and_collect():
    """Run simulation for multiple agent counts and collect data."""
    agent_counts = [10, 20, 30, 50, 75, 100]
    data = {"agent_counts": agent_counts, "lru": [], "semantic": []}

    for n in agent_counts:
        config = SimulationConfig(num_agents=n, simulation_steps=720, seed=42)
        runner = SimulationRunner(config)
        for policy in ["lru", "semantic"]:
            m = runner.run(policy)
            data[policy].append({
                "max_agents": m.max_concurrent_agents,
                "resume_ms": m.avg_resume_latency_ms,
                "gpu_util": m.avg_gpu_utilization,
                "cost": m.avg_cost_per_hour / max(1, m.max_concurrent_agents),
            })
    return data


def plot_comparison(data: dict, output_dir: str):
    counts = data["agent_counts"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Nemorix: LRU vs Semantic Eviction Policy Comparison", fontsize=14, fontweight="bold")

    # 1. Max concurrent agents
    ax = axes[0][0]
    ax.plot(counts, [d["max_agents"] for d in data["lru"]], "o-", label="LRU", color="#e74c3c")
    ax.plot(counts, [d["max_agents"] for d in data["semantic"]], "s-", label="Nemorix (Semantic + CXL)", color="#2ecc71")
    ax.set_xlabel("Total Agents")
    ax.set_ylabel("Max Concurrent Active")
    ax.set_title("Agent Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Resume latency
    ax = axes[0][1]
    ax.plot(counts, [d["resume_ms"] for d in data["lru"]], "o-", label="LRU", color="#e74c3c")
    ax.plot(counts, [d["resume_ms"] for d in data["semantic"]], "s-", label="Nemorix", color="#2ecc71")
    ax.set_xlabel("Total Agents")
    ax.set_ylabel("Avg Resume Latency (ms)")
    ax.set_title("Agent Resume Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. GPU utilization
    ax = axes[1][0]
    ax.bar([c - 1.5 for c in counts], [d["gpu_util"] * 100 for d in data["lru"]], 3, label="LRU", color="#e74c3c", alpha=0.7)
    ax.bar([c + 1.5 for c in counts], [d["gpu_util"] * 100 for d in data["semantic"]], 3, label="Nemorix", color="#2ecc71", alpha=0.7)
    ax.set_xlabel("Total Agents")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU VRAM Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cost per agent-hour
    ax = axes[1][1]
    ax.plot(counts, [d["cost"] for d in data["lru"]], "o-", label="LRU", color="#e74c3c")
    ax.plot(counts, [d["cost"] for d in data["semantic"]], "s-", label="Nemorix", color="#2ecc71")
    ax.set_xlabel("Total Agents")
    ax.set_ylabel("Cost per Agent-Hour ($)")
    ax.set_title("Infrastructure Cost Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "nemorix_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.close()


def main():
    output_dir = os.path.dirname(__file__)
    print("Running simulations for chart generation...")
    data = run_and_collect()
    plot_comparison(data, output_dir)

    # Save raw data
    json_path = os.path.join(output_dir, "chart_data.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
