"""Semantic-aware eviction policy — Nemorix's core innovation."""

from __future__ import annotations
import math
from typing import List
from nemorix.core.kv_block import KVBlock


class SemanticEvictionPolicy:
    """Eviction policy that considers attention importance, reconstruction
    cost, and agent priority — not just access recency."""

    def __init__(
        self,
        w_recency: float = 0.25,
        w_importance: float = 0.30,
        w_priority: float = 0.20,
        w_recompute: float = 0.25,
    ):
        self.w_recency = w_recency
        self.w_importance = w_importance
        self.w_priority = w_priority
        self.w_recompute = w_recompute
        # Maps agent_id -> priority; populated by the scheduler
        self.agent_priorities: dict[str, int] = {}

    def set_agent_priority(self, agent_id: str, priority: int) -> None:
        self.agent_priorities[agent_id] = priority

    def _recency_score(self, block: KVBlock, current_time: float) -> float:
        """Higher = more recently accessed = keep."""
        age = max(1.0, current_time - block.last_accessed)
        return 1.0 / (1.0 + math.log(age))

    def _importance_score(self, block: KVBlock) -> float:
        """Higher = more important (receives more attention) = keep."""
        return block.attention_score

    def _priority_score(self, block: KVBlock) -> float:
        """Higher = higher priority agent = keep."""
        prio = self.agent_priorities.get(block.agent_id, 5)
        return 1.0 - (prio / 10.0)

    def _recompute_cost(self, block: KVBlock) -> float:
        """Higher = more expensive to recompute = keep.
        Deeper layers and longer prefixes cost more to recompute."""
        layer_cost = block.layer_idx / 80.0  # normalize to ~80 layers
        token_cost = min(1.0, block.num_tokens / 4096.0)
        return (layer_cost + token_cost) / 2.0

    def eviction_score(self, block: KVBlock, current_time: float) -> float:
        """Lower score = evict first."""
        return (
            self.w_recency * self._recency_score(block, current_time)
            + self.w_importance * self._importance_score(block)
            + self.w_priority * self._priority_score(block)
            + self.w_recompute * self._recompute_cost(block)
        )

    def select_victims(
        self, blocks: List[KVBlock], required_bytes: int, current_time: float
    ) -> List[KVBlock]:
        scored = [(b, self.eviction_score(b, current_time)) for b in blocks]
        scored.sort(key=lambda x: x[1])  # lowest score evicted first
        victims: list[KVBlock] = []
        freed = 0
        for block, _ in scored:
            victims.append(block)
            freed += block.size_bytes
            if freed >= required_bytes:
                break
        return victims
