"""Baseline LRU (Least Recently Used) eviction policy."""

from __future__ import annotations
from typing import List
from nemorix.core.kv_block import KVBlock


class LRUEvictionPolicy:
    def select_victims(
        self, blocks: List[KVBlock], required_bytes: int, current_time: float
    ) -> List[KVBlock]:
        sorted_blocks = sorted(blocks, key=lambda b: b.last_accessed)
        victims: list[KVBlock] = []
        freed = 0
        for block in sorted_blocks:
            victims.append(block)
            freed += block.size_bytes
            if freed >= required_bytes:
                break
        return victims
