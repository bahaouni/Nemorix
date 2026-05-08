"""Agent Memory Object — an agent's persistent brain state."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from nemorix.core.kv_block import KVBlock


@dataclass
class AgentMemoryObject:
    agent_id: str = ""
    blocks: List[KVBlock] = field(default_factory=list)
    state: str = "suspended"  # running, ready, sleeping, suspended
    priority: int = 5  # 0 = highest, 10 = lowest
    total_context_tokens: int = 0
    created_at: float = 0.0
    last_inference_at: float = 0.0
    activation_probability: float = 0.1
    resume_count: int = 0
    total_resume_latency_ms: float = 0.0

    @property
    def total_size_bytes(self) -> int:
        return sum(b.size_bytes for b in self.blocks)

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    def blocks_in_tier(self, tier: str) -> List[KVBlock]:
        return [b for b in self.blocks if b.tier == tier]

    @property
    def primary_tier(self) -> str:
        if not self.blocks:
            return "none"
        tier_counts: dict[str, int] = {}
        for b in self.blocks:
            tier_counts[b.tier] = tier_counts.get(b.tier, 0) + 1
        return max(tier_counts, key=tier_counts.get)  # type: ignore[arg-type]

    @property
    def avg_resume_latency_ms(self) -> float:
        if self.resume_count == 0:
            return 0.0
        return self.total_resume_latency_ms / self.resume_count

    def record_resume(self, latency_ms: float) -> None:
        self.resume_count += 1
        self.total_resume_latency_ms += latency_ms
