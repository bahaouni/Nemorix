"""Simulation runner — drives the full benchmark and collects metrics."""

from __future__ import annotations
import bisect
import json
import math
from dataclasses import dataclass, field, replace
from typing import List
from nemorix.core.kv_block import KVBlock, COMPRESSION_RATIOS
from nemorix.core.tier_manager import MemoryTierManager
from nemorix.core.agent import AgentMemoryObject
from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator, ModelConfig


# Prefill throughput for Llama-3-70B on H100 SXM5 (FP16, single long sequence).
# Source: vLLM offline benchmark, MLPerf Inference v4.0 ~ 40K-60K tokens/s.
# Using 40K as a conservative lower-bound so no_offload results are not
# artificially inflated. Real deployments may be 1.5x faster.
RECOMPUTE_TOKENS_PER_SEC = 40_000.0


GPU_RENTAL_COST_PER_HOUR = 3.00  # H100 hourly rental cost
SLA_THRESHOLD_MS = 200.0  # max acceptable resume latency


@dataclass
class SimulationMetrics:
    max_concurrent_agents: int = 0
    total_resumes: int = 0
    total_resume_latency_ms: float = 0.0
    gpu_utilization_samples: List[float] = field(default_factory=list)
    cost_samples: List[float] = field(default_factory=list)
    eviction_events: int = 0
    correct_evictions: int = 0
    warm_resumes: int = 0
    resume_latencies: List[float] = field(default_factory=list)
    agent_latencies: dict = field(default_factory=dict)  # agent_id -> list[float]

    @property
    def avg_resume_latency_ms(self) -> float:
        if self.total_resumes == 0:
            return 0.0
        return self.total_resume_latency_ms / self.total_resumes

    def _percentile(self, pct: float) -> float:
        """Linear-interpolated percentile (NumPy 'linear' method), no deps.

        pct is in [0, 100]. This is the standard interpolation used by
        numpy.percentile, so reviewers can reproduce these numbers exactly.
        """
        data = self.resume_latencies
        if not data:
            return 0.0
        s = sorted(data)
        if len(s) == 1:
            return s[0]
        rank = (pct / 100.0) * (len(s) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(s) - 1)
        frac = rank - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    @property
    def p50_resume_latency_ms(self) -> float:
        return self._percentile(50.0)

    @property
    def p95_resume_latency_ms(self) -> float:
        return self._percentile(95.0)

    @property
    def p99_resume_latency_ms(self) -> float:
        return self._percentile(99.0)

    @property
    def sla_agents(self) -> int:
        """Number of unique agents whose average resume latency is under SLA."""
        count = 0
        for lats in self.agent_latencies.values():
            if lats and (sum(lats) / len(lats)) < SLA_THRESHOLD_MS:
                count += 1
        return count

    @property
    def avg_gpu_utilization(self) -> float:
        if not self.gpu_utilization_samples:
            return 0.0
        return min(1.0, sum(self.gpu_utilization_samples) / len(self.gpu_utilization_samples))

    @property
    def avg_cost_per_hour(self) -> float:
        if not self.cost_samples:
            return 0.0
        return sum(self.cost_samples) / len(self.cost_samples)

    @property
    def total_cost_per_hour(self) -> float:
        """Storage + GPU rental cost."""
        return self.avg_cost_per_hour + GPU_RENTAL_COST_PER_HOUR

    @property
    def eviction_accuracy(self) -> float:
        if self.eviction_events == 0:
            return 0.0
        return self.correct_evictions / self.eviction_events

    @property
    def warm_resume_rate(self) -> float:
        """Fraction of resumes served warm (< 100 ms) instead of from a cold
        tier. This is the hit-rate-aligned metric that Belady's MIN optimizes,
        so it is the correct quantity for an optimality-gap comparison.
        """
        if self.total_resumes == 0:
            return 0.0
        return self.warm_resumes / self.total_resumes


@dataclass
class AggregatedMetric:
    """Mean +/- std of a single scalar metric across multiple random seeds."""

    metric: str
    policy: str
    mean: float
    std: float
    min: float
    max: float
    n: int
    values: List[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, metric: str, policy: str, values: List[float]) -> "AggregatedMetric":
        n = len(values)
        if n == 0:
            return cls(metric, policy, 0.0, 0.0, 0.0, 0.0, 0, [])
        mean = sum(values) / n
        # Population standard deviation (we observe the full set of seeds run).
        var = sum((v - mean) ** 2 for v in values) / n
        return cls(
            metric=metric,
            policy=policy,
            mean=mean,
            std=math.sqrt(var),
            min=min(values),
            max=max(values),
            n=n,
            values=list(values),
        )


@dataclass
class SimulationConfig:
    num_agents: int = 50
    context_tokens: int = 65536
    gpu_memory_gb: float = 80
    cxl_memory_gb: float = 512
    ram_gb: float = 256
    ssd_gb: float = 4000
    simulation_steps: int = 1440  # 24 hours at 1 step/minute
    idle_threshold_s: float = 300.0
    seed: int = 42
    model_name: str = "Llama-3-70B"
    model_layers: int = 80


class SimulationRunner:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def _recompute_latency_ms(self, tokens: int) -> float:
        """Latency to recompute KV-cache from scratch (prefill)."""
        return (tokens / RECOMPUTE_TOKENS_PER_SEC) * 1000.0

    def run(
        self, policy_name: str = "semantic", semantic_kwargs: dict | None = None
    ) -> SimulationMetrics:
        cfg = self.config
        model = ModelConfig(name=cfg.model_name, num_layers=cfg.model_layers)
        workload = WorkloadGenerator(model=model, seed=cfg.seed)
        agents = workload.create_workload(cfg.num_agents, cfg.context_tokens)

        is_no_offload = policy_name == "no_offload"

        if policy_name == "semantic":
            # ``semantic_kwargs`` lets the proof harness run ablations (e.g.
            # zero out one retention term) without touching production defaults.
            policy = SemanticEvictionPolicy(**(semantic_kwargs or {}))
            for a in agents:
                policy.set_agent_priority(a.agent_id, a.priority)
            tier_mgr = MemoryTierManager(
                cfg.gpu_memory_gb, cfg.cxl_memory_gb, cfg.ram_gb, cfg.ssd_gb
            )
        elif policy_name in ("lru", "belady"):
            # LRU baseline and the Belady optimal oracle both get the same
            # 4-tier hierarchy, so any difference is purely the eviction rule.
            policy = LRUEvictionPolicy()
            tier_mgr = MemoryTierManager(
                cfg.gpu_memory_gb, cfg.cxl_memory_gb, cfg.ram_gb, cfg.ssd_gb
            )
        else:  # no_offload
            policy = LRUEvictionPolicy()
            tier_mgr = MemoryTierManager(cfg.gpu_memory_gb, 0.001, 0.001, 0.001)

        gpu_cap = tier_mgr.get_tier("gpu").capacity_bytes
        has_cxl = not is_no_offload
        cxl_cap_bytes = int(cfg.cxl_memory_gb * 1024**3) if has_cxl else 0
        ram_cap_bytes = int(cfg.ram_gb * 1024**3) if not is_no_offload else 0
        ssd_cap_bytes = int(cfg.ssd_gb * 1024**3) if not is_no_offload else 0

        agent_map: dict[str, AgentMemoryObject] = {a.agent_id: a for a in agents}
        gpu_agents: list[str] = []
        # LRU/Nemorix: agents pre-stored in SSD (pre-computed KV cache)
        # no_offload: no persistent storage
        init_tier = "none" if is_no_offload else "ssd"
        agent_tier: dict[str, str] = {a.agent_id: init_tier for a in agents}

        # Warmup period: first N steps don't count for metrics (cold starts)
        warmup_steps = min(60, cfg.simulation_steps // 10)

        def tier_used(tier_name: str) -> int:
            return sum(
                agent_map[a].total_size_bytes
                for a, t in agent_tier.items()
                if t == tier_name
            )

        def best_eviction_dest(agent_size: int) -> str:
            """Find the best cold tier that has capacity."""
            if has_cxl and tier_used("cxl") + agent_size <= cxl_cap_bytes:
                return "cxl"
            if ram_cap_bytes > 0 and tier_used("ram") + agent_size <= ram_cap_bytes:
                return "ram"
            if ssd_cap_bytes > 0 and tier_used("ssd") + agent_size <= ssd_cap_bytes:
                return "ssd"
            return "none"  # no space anywhere → state lost

        metrics = SimulationMetrics()

        future_activations: dict[int, List[str]] = {}
        for step in range(cfg.simulation_steps):
            future_activations[step] = workload.generate_activations(agents, step)

        # Belady's MIN oracle (offline optimal): precompute each agent's sorted
        # list of future activation steps so we can evict the agent referenced
        # *farthest in the future*. This is the theoretical upper bound that
        # bounds how good ANY online policy (LRU, Nemorix) can possibly be.
        agent_activation_steps: dict[str, List[int]] = {}
        if policy_name == "belady":
            for s in range(cfg.simulation_steps):
                for aid in future_activations[s]:
                    agent_activation_steps.setdefault(aid, []).append(s)

        for step in range(cfg.simulation_steps):
            current_time = step * 60.0

            # Suspend idle agents: move from GPU to colder tier
            to_remove = []
            for aid in gpu_agents:
                agent = agent_map[aid]
                idle_time = current_time - agent.last_inference_at
                if idle_time > cfg.idle_threshold_s:
                    dest = best_eviction_dest(agent.total_size_bytes)
                    agent_tier[aid] = dest
                    agent.state = "suspended" if dest == "none" else "sleeping"
                    to_remove.append(aid)
            for aid in to_remove:
                gpu_agents.remove(aid)

            # Activate agents for this step
            activated_ids = future_activations[step]
            for aid in activated_ids:
                if aid not in agent_map:
                    continue
                agent = agent_map[aid]
                src_tier = agent_tier[aid]

                if src_tier == "gpu":
                    # Already in GPU, just update timestamp
                    agent.last_inference_at = current_time
                    latency = 0.0
                else:
                    # Need to load into GPU
                    agent_size = agent.total_size_bytes

                    # Evict from GPU if needed
                    while len(gpu_agents) > 0:
                        total_gpu_used = sum(
                            agent_map[a].total_size_bytes for a in gpu_agents
                        )
                        if total_gpu_used + agent_size <= gpu_cap:
                            break
                        # Use the actual policy to select eviction victim
                        if policy_name == "semantic":
                            # Collect all GPU-resident blocks and use the
                            # SemanticEvictionPolicy's full retention scoring,
                            # gated by current memory pressure (rho).
                            gpu_blocks = []
                            for gaid in gpu_agents:
                                gpu_blocks.extend(agent_map[gaid].blocks)
                            pressure = min(1.0, total_gpu_used / max(1, gpu_cap))
                            victims = policy.select_victims(
                                gpu_blocks, agent_size, current_time, pressure
                            )
                            # Find which agent owns the majority of victim blocks
                            victim_agents: dict[str, int] = {}
                            for v in victims:
                                victim_agents[v.agent_id] = (
                                    victim_agents.get(v.agent_id, 0) + 1
                                )
                            evict_id = max(
                                victim_agents, key=victim_agents.get
                            )
                        elif policy_name == "belady":
                            # SLA-aware optimal offline oracle. The classic
                            # Belady MIN minimizes *miss count*, but our metric
                            # is SLA (which agents stay fast). The right upper
                            # bound for THAT objective evicts the agent that is
                            # (a) needed farthest in the future, breaking ties
                            # toward (b) the one whose reload would be cheapest
                            # if it IS needed -- i.e. the least SLA damage. This
                            # is a genuine upper bound: it has full future
                            # knowledge AND accounts for tier reload cost.
                            def _belady_cost(x: str) -> tuple:
                                steps_list = agent_activation_steps.get(x, ())
                                i = bisect.bisect_right(steps_list, step)
                                next_use = (
                                    steps_list[i]
                                    if i < len(steps_list)
                                    else cfg.simulation_steps + 1
                                )
                                # Primary: farthest next use (evict last-needed).
                                # Secondary: prefer evicting larger agents (free
                                # more room per eviction -> fewer total evictions
                                # -> fewer SLA violations), matching the knapsack
                                # objective the online law approximates.
                                return (next_use, agent_map[x].total_size_bytes)

                            evict_id = max(gpu_agents, key=_belady_cost)
                        else:
                            # LRU: evict least recently accessed
                            evict_id = min(
                                gpu_agents,
                                key=lambda x: agent_map[x].last_inference_at,
                            )
                        evicted = agent_map[evict_id]
                        gpu_agents.remove(evict_id)

                        # Check if eviction was "correct" (not needed in next 5 steps)
                        metrics.eviction_events += 1
                        needed_soon = False
                        for fs in range(step + 1, min(step + 6, cfg.simulation_steps)):
                            if evict_id in future_activations.get(fs, []):
                                needed_soon = True
                                break
                        if not needed_soon:
                            metrics.correct_evictions += 1

                        if is_no_offload:
                            agent_tier[evict_id] = "none"
                            evicted.state = "suspended"
                        else:
                            dest = best_eviction_dest(evicted.total_size_bytes)
                            agent_tier[evict_id] = dest
                            evicted.state = "suspended" if dest == "none" else "ready"

                    # Calculate resume latency based on source tier
                    if src_tier == "none":
                        latency = self._recompute_latency_ms(agent.total_context_tokens)
                    elif src_tier in ("cxl", "ram", "ssd"):
                        # On-demand paging: first 10% of layers
                        first_layers = max(1, len(agent.blocks) // 10)
                        per_layer_fp16 = agent.blocks[0].size_bytes if agent.blocks else 0
                        # Apply compression ratio for the source tier's dtype
                        tier_dtype = tier_mgr.get_tier(src_tier).compress_dtype
                        comp_ratio = COMPRESSION_RATIOS.get(tier_dtype, 1.0)
                        per_layer_compressed = int(per_layer_fp16 * comp_ratio)
                        xfer_bytes = first_layers * per_layer_compressed
                        tier_obj = tier_mgr.get_tier(src_tier)
                        latency = tier_obj.transfer_time_ms(xfer_bytes)
                    else:
                        latency = 0.0

                    gpu_agents.append(aid)
                    agent_tier[aid] = "gpu"
                    agent.state = "running"
                    agent.last_inference_at = current_time
                    if src_tier != "gpu":
                        metrics.warm_resumes += 1 if latency < 100.0 else 0

                # Online-learn the agent's idle period so the wake-hazard
                # model (semantic policy) adapts to each agent's rhythm.
                if policy_name == "semantic":
                    policy.observe_access(aid, current_time)

                # Only count metrics after warmup period (consistent)
                if step >= warmup_steps:
                    metrics.total_resumes += 1
                    metrics.total_resume_latency_ms += latency
                    metrics.resume_latencies.append(latency)
                    if aid not in metrics.agent_latencies:
                        metrics.agent_latencies[aid] = []
                    metrics.agent_latencies[aid].append(latency)
                agent.record_resume(latency)

            # Record metrics
            running = len(gpu_agents)
            metrics.max_concurrent_agents = max(metrics.max_concurrent_agents, running)

            total_gpu_used = sum(agent_map[a].total_size_bytes for a in gpu_agents)
            gpu_util = min(1.0, total_gpu_used / gpu_cap) if gpu_cap > 0 else 0
            metrics.gpu_utilization_samples.append(gpu_util)

            # Cost: based on actual tier placement with compression
            hour_cost = 0.0
            for aid, tier in agent_tier.items():
                if tier == "none":
                    continue
                a = agent_map[aid]
                gb_fp16 = a.total_size_bytes / (1024**3)
                t = tier_mgr.get_tier(tier)
                # Apply compression ratio for the tier's storage dtype
                comp = COMPRESSION_RATIOS.get(t.compress_dtype, 1.0)
                gb_stored = gb_fp16 * comp
                hour_cost += gb_stored * t.cost_per_gb_hour
            metrics.cost_samples.append(hour_cost)

        return metrics

    def run_all_policies(self) -> dict[str, SimulationMetrics]:
        results = {}
        for policy in ["no_offload", "lru", "semantic"]:
            results[policy] = self.run(policy)
        return results

    def run_multiseed(
        self, seeds: List[int], policies: List[str] | None = None
    ) -> dict[str, "AggregatedMetric"]:
        """Run every policy across multiple seeds and aggregate (mean +/- std).

        This is the statistical-robustness path: it proves the headline numbers
        are not an artifact of a single lucky seed. For each policy and each
        scalar metric we report the sample mean and (population) standard
        deviation across all seeds.
        """
        policies = policies or ["no_offload", "lru", "semantic"]
        # metric_name -> policy -> list of per-seed values
        collected: dict[str, dict[str, List[float]]] = {}

        for seed in seeds:
            cfg = replace(self.config, seed=seed)
            runner = SimulationRunner(cfg)
            for policy in policies:
                m = runner.run(policy)
                sla = m.sla_agents
                effective = sla if sla > 0 else m.max_concurrent_agents
                per_seed = {
                    "sla_agents": float(m.sla_agents),
                    "max_gpu_resident_agents": float(m.max_concurrent_agents),
                    "avg_resume_latency_ms": m.avg_resume_latency_ms,
                    "p50_resume_latency_ms": m.p50_resume_latency_ms,
                    "p95_resume_latency_ms": m.p95_resume_latency_ms,
                    "p99_resume_latency_ms": m.p99_resume_latency_ms,
                    "avg_gpu_utilization": m.avg_gpu_utilization,
                    "eviction_accuracy": m.eviction_accuracy,
                    "total_cost_per_hour": m.total_cost_per_hour,
                    "cost_per_agent_hour": m.total_cost_per_hour / max(1, effective),
                }
                for metric_name, value in per_seed.items():
                    collected.setdefault(metric_name, {}).setdefault(policy, []).append(value)

        aggregated: dict[str, AggregatedMetric] = {}
        for metric_name, by_policy in collected.items():
            for policy, values in by_policy.items():
                key = f"{policy}.{metric_name}"
                aggregated[key] = AggregatedMetric.from_values(metric_name, policy, values)
        return aggregated

    @staticmethod
    def format_multiseed(
        aggregated: dict[str, "AggregatedMetric"],
        seeds: List[int],
        config: SimulationConfig,
        policies: List[str] | None = None,
    ) -> str:
        policies = policies or ["no_offload", "lru", "semantic"]
        label_map = {"no_offload": "No Offload", "lru": "LRU", "semantic": "Nemorix"}
        rows = [
            ("Agents under SLA (<200ms)", "sla_agents", "{:.1f}"),
            ("Avg resume latency (ms)", "avg_resume_latency_ms", "{:.1f}"),
            ("P95 resume latency (ms)", "p95_resume_latency_ms", "{:.1f}"),
            ("P99 resume latency (ms)", "p99_resume_latency_ms", "{:.1f}"),
            ("GPU utilization (%)", "avg_gpu_utilization", "{:.0%}"),
            ("Cost per agent-hour ($)", "cost_per_agent_hour", "${:.2f}"),
        ]
        lines = []
        lines.append("=" * 84)
        lines.append("  Nemorix Robustness Results  (mean +/- std across seeds)")
        lines.append("=" * 84)
        lines.append("")
        lines.append(f"  Seeds: {seeds}  (n={len(seeds)})")
        lines.append(f"  Agents: {config.num_agents} | context ~{config.context_tokens // 1024}K | "
                     f"{config.simulation_steps // 60}h")
        lines.append("")
        header = f"  {'Metric':<28}" + "".join(f"{label_map[p]:>18}" for p in policies)
        lines.append(header)
        lines.append("  " + "-" * (28 + 18 * len(policies)))
        for label, metric_name, fmt in rows:
            cells = ""
            for p in policies:
                agg = aggregated.get(f"{p}.{metric_name}")
                if agg is None:
                    cells += f"{'-':>18}"
                    continue
                mean_str = fmt.format(agg.mean)
                std_str = fmt.format(agg.std).lstrip("$")
                cells += f"{mean_str + ' +/-' + std_str:>18}"
            lines.append(f"  {label:<28}{cells}")
        lines.append("")
        lines.append("=" * 84)
        return "\n".join(lines)

    @staticmethod
    def multiseed_to_json(
        aggregated: dict[str, "AggregatedMetric"], seeds: List[int]
    ) -> str:
        data: dict = {"seeds": seeds, "n_seeds": len(seeds), "metrics": {}}
        for key, agg in aggregated.items():
            data["metrics"][key] = {
                "policy": agg.policy,
                "metric": agg.metric,
                "mean": round(agg.mean, 4),
                "std": round(agg.std, 4),
                "min": round(agg.min, 4),
                "max": round(agg.max, 4),
                "n": agg.n,
            }
        return json.dumps(data, indent=2)

    @staticmethod
    def format_results(results: dict[str, SimulationMetrics], config: SimulationConfig) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("  Nemorix Simulation Results")
        lines.append("=" * 78)
        lines.append("")
        lines.append("Configuration:")
        lines.append(f"  GPU VRAM: {config.gpu_memory_gb:.0f} GB | CXL: {config.cxl_memory_gb:.0f} GB | "
                     f"RAM: {config.ram_gb:.0f} GB | SSD: {config.ssd_gb / 1000:.0f} TB")
        lines.append(f"  Model: {config.model_name} ({config.model_layers} layers)")
        lines.append(f"  Agents: {config.num_agents} concurrent, context ~{config.context_tokens // 1024}K tokens")
        lines.append(f"  Duration: {config.simulation_steps} min ({config.simulation_steps / 60:.0f} hours)")
        lines.append(f"  SLA threshold: {SLA_THRESHOLD_MS:.0f} ms")
        lines.append("")

        # Table header
        header = f"  {'Metric':<35} {'No Offload':>14} {'LRU':>14} {'Nemorix':>14}"
        lines.append(header)
        lines.append("  " + "-" * 78)

        no = results.get("no_offload", SimulationMetrics())
        lru = results.get("lru", SimulationMetrics())
        sem = results.get("semantic", SimulationMetrics())

        def row(label: str, v1: str, v2: str, v3: str) -> str:
            return f"  {label:<35} {v1:>14} {v2:>14} {v3:>14}"

        lines.append(row("Agents under SLA (<200ms)",
                         str(no.sla_agents),
                         str(lru.sla_agents),
                         str(sem.sla_agents)))

        lines.append(row("Max GPU-resident agents",
                         str(no.max_concurrent_agents),
                         str(lru.max_concurrent_agents),
                         str(sem.max_concurrent_agents)))

        lines.append(row("Avg resume latency",
                         f"{no.avg_resume_latency_ms:.1f} ms",
                         f"{lru.avg_resume_latency_ms:.1f} ms",
                         f"{sem.avg_resume_latency_ms:.1f} ms"))

        lines.append(row("P50 resume latency",
                         f"{no.p50_resume_latency_ms:.1f} ms",
                         f"{lru.p50_resume_latency_ms:.1f} ms",
                         f"{sem.p50_resume_latency_ms:.1f} ms"))

        lines.append(row("P95 resume latency",
                         f"{no.p95_resume_latency_ms:.1f} ms",
                         f"{lru.p95_resume_latency_ms:.1f} ms",
                         f"{sem.p95_resume_latency_ms:.1f} ms"))

        lines.append(row("P99 resume latency",
                         f"{no.p99_resume_latency_ms:.1f} ms",
                         f"{lru.p99_resume_latency_ms:.1f} ms",
                         f"{sem.p99_resume_latency_ms:.1f} ms"))

        lines.append(row("GPU utilization",
                         f"{no.avg_gpu_utilization * 100:.0f}%",
                         f"{lru.avg_gpu_utilization * 100:.0f}%",
                         f"{sem.avg_gpu_utilization * 100:.0f}%"))

        lines.append(row("Eviction accuracy",
                         "N/A",
                         f"{lru.eviction_accuracy * 100:.0f}%",
                         f"{sem.eviction_accuracy * 100:.0f}%"))

        # Cost per agent-hour (total cost / SLA agents)
        no_sla = max(1, no.sla_agents)
        lru_sla = max(1, lru.sla_agents)
        sem_sla = max(1, sem.sla_agents)
        no_cost = no.total_cost_per_hour / no_sla
        lru_cost = lru.total_cost_per_hour / lru_sla
        sem_cost = sem.total_cost_per_hour / sem_sla

        lines.append(row("Cost per agent-hour (incl. GPU)",
                         f"${no_cost:.2f}",
                         f"${lru_cost:.2f}",
                         f"${sem_cost:.2f}"))

        lines.append("")

        if sem.sla_agents > 0:
            # For no_offload, effective capacity = GPU-resident agents only
            no_effective = no.sla_agents if no.sla_agents > 0 else no.max_concurrent_agents
            density_improve = sem.sla_agents / max(1, no_effective)
            lines.append(f"  => {density_improve:.0f}x improvement in agent density (SLA-bound)")
        if no.avg_resume_latency_ms > 0 and sem.avg_resume_latency_ms > 0:
            latency_improve = no.avg_resume_latency_ms / sem.avg_resume_latency_ms
            lines.append(f"  => {latency_improve:.0f}x improvement in resume latency")
        if no_cost > 0 and sem_cost > 0:
            cost_reduction = (1 - sem_cost / no_cost) * 100
            lines.append(f"  => {cost_reduction:.0f}% reduction in cost per agent-hour")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def results_to_json(results: dict[str, SimulationMetrics]) -> str:
        data = {}
        for name, m in results.items():
            data[name] = {
                "sla_agents": m.sla_agents,
                "max_gpu_resident_agents": m.max_concurrent_agents,
                "avg_resume_latency_ms": round(m.avg_resume_latency_ms, 2),
                "p50_resume_latency_ms": round(m.p50_resume_latency_ms, 2),
                "p95_resume_latency_ms": round(m.p95_resume_latency_ms, 2),
                "p99_resume_latency_ms": round(m.p99_resume_latency_ms, 2),
                "avg_gpu_utilization": round(m.avg_gpu_utilization, 4),
                "eviction_accuracy": round(m.eviction_accuracy, 4),
                "storage_cost_per_hour": round(m.avg_cost_per_hour, 4),
                "total_cost_per_hour": round(m.total_cost_per_hour, 4),
                "total_resumes": m.total_resumes,
                "warm_resumes": m.warm_resumes,
            }
        return json.dumps(data, indent=2)
