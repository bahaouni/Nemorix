"""API request/response schemas."""

from __future__ import annotations


class AgentCreateRequest:
    def __init__(self, context_tokens: int = 65536, priority: int = 5):
        self.context_tokens = context_tokens
        self.priority = priority


class AgentResponse:
    def __init__(
        self,
        agent_id: str,
        state: str,
        tier: str,
        size_mb: float,
        resume_count: int,
        avg_resume_latency_ms: float,
    ):
        self.agent_id = agent_id
        self.state = state
        self.tier = tier
        self.size_mb = size_mb
        self.resume_count = resume_count
        self.avg_resume_latency_ms = avg_resume_latency_ms

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "tier": self.tier,
            "size_mb": round(self.size_mb, 2),
            "resume_count": self.resume_count,
            "avg_resume_latency_ms": round(self.avg_resume_latency_ms, 2),
        }


class MetricsResponse:
    def __init__(self, tiers: dict, agents: int, running: int):
        self.tiers = tiers
        self.agents = agents
        self.running = running

    def to_dict(self) -> dict:
        return {
            "tiers": self.tiers,
            "total_agents": self.agents,
            "running_agents": self.running,
        }
