"""Agent Process Scheduler — OS-style scheduling for LLM agents."""

from __future__ import annotations
from typing import List
from nemorix.core.agent import AgentMemoryObject
from nemorix.core.tier_manager import MemoryTierManager, EvictionPolicy


class AgentScheduler:
    def __init__(self, tier_manager: MemoryTierManager, policy: EvictionPolicy):
        self.tier_manager = tier_manager
        self.policy = policy
        self.agents: dict[str, AgentMemoryObject] = {}

    def register_agent(self, agent: AgentMemoryObject) -> None:
        self.agents[agent.agent_id] = agent

    def all_blocks(self) -> list:
        blocks = []
        for a in self.agents.values():
            blocks.extend(a.blocks)
        return blocks

    def activate_agent(self, agent_id: str, current_time: float) -> float:
        """Bring agent to RUNNING state. Returns resume latency in ms."""
        agent = self.agents[agent_id]
        if agent.state == "running":
            return 0.0

        agent_size = agent.total_size_bytes
        # Ensure GPU has space
        self.tier_manager.ensure_space(
            "gpu", agent_size, self.all_blocks(), self.policy, current_time
        )
        # Migrate agent blocks to GPU
        latency = self.tier_manager.migrate_agent_blocks(agent.blocks, "gpu")
        agent.state = "running"
        agent.last_inference_at = current_time
        for b in agent.blocks:
            b.last_accessed = current_time
        agent.record_resume(latency)
        return latency

    def deactivate_agent(
        self, agent_id: str, target_tier: str, current_time: float
    ) -> None:
        """Move agent to a colder tier."""
        agent = self.agents[agent_id]
        state_map = {"cxl": "ready", "ram": "sleeping", "ssd": "suspended"}
        self.tier_manager.migrate_agent_blocks(agent.blocks, target_tier)
        agent.state = state_map.get(target_tier, "suspended")

    def suspend_idle_agents(
        self, current_time: float, idle_threshold_s: float = 300.0
    ) -> int:
        """Move agents idle longer than threshold to colder tiers. Returns count."""
        suspended = 0
        for agent in list(self.agents.values()):
            if agent.state != "running":
                continue
            idle_time = current_time - agent.last_inference_at
            if idle_time > idle_threshold_s * 3:
                self.deactivate_agent(agent.agent_id, "ssd", current_time)
                suspended += 1
            elif idle_time > idle_threshold_s * 2:
                self.deactivate_agent(agent.agent_id, "ram", current_time)
                suspended += 1
            elif idle_time > idle_threshold_s:
                self.deactivate_agent(agent.agent_id, "cxl", current_time)
                suspended += 1
        return suspended

    def get_running_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.state == "running")

    def get_state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.agents.values():
            counts[a.state] = counts.get(a.state, 0) + 1
        return counts
