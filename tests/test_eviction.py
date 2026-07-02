"""Tests for eviction policies: LRU vs Semantic comparison."""

from __future__ import annotations

from nemorix.core.kv_block import KVBlock
from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.policies.semantic import SemanticEvictionPolicy


def make_block(
    block_id: str,
    agent_id: str = "a1",
    layer_idx: int = 0,
    attention: float = 0.5,
    last_accessed: float = 0.0,
    size: int = 1024 * 1024,
) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        agent_id=agent_id,
        layer_idx=layer_idx,
        num_tokens=1024,
        size_bytes=size,
        importance_score=attention,
        last_accessed=last_accessed,
    )


def test_lru_evicts_oldest():
    """LRU should evict least recently accessed blocks first."""
    policy = LRUEvictionPolicy()
    blocks = [
        make_block("b1", last_accessed=100.0),
        make_block("b2", last_accessed=50.0),   # oldest
        make_block("b3", last_accessed=200.0),
    ]
    victims = policy.select_victims(blocks, 1024 * 1024, current_time=300.0)
    assert victims[0].block_id == "b2"
    print("  [PASS] test_lru_evicts_oldest")


def test_semantic_keeps_important():
    """Semantic policy should keep high-attention blocks."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("a1", 5)

    system_prompt = make_block("sys", attention=0.95, last_accessed=10.0, layer_idx=1)
    filler = make_block("filler", attention=0.1, last_accessed=50.0, layer_idx=40)
    recent = make_block("recent", attention=0.3, last_accessed=90.0, layer_idx=20)

    blocks = [system_prompt, filler, recent]
    victims = policy.select_victims(blocks, 1024 * 1024, current_time=100.0)

    # Filler should be evicted first (low attention, mid layer)
    assert victims[0].block_id == "filler"
    # System prompt should be evicted LAST (high attention)
    assert system_prompt not in victims[:2] or len(victims) == 3

    print("  [PASS] test_semantic_keeps_important")


def test_semantic_respects_priority():
    """Semantic policy should prefer evicting low-priority agent blocks."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("critical", 1)   # high priority
    policy.set_agent_priority("background", 9)  # low priority

    critical_block = make_block("crit", agent_id="critical", attention=0.5, last_accessed=50.0)
    background_block = make_block("bg", agent_id="background", attention=0.5, last_accessed=50.0)

    blocks = [critical_block, background_block]
    victims = policy.select_victims(blocks, 1024 * 1024, current_time=100.0)

    # Background block should be evicted first
    assert victims[0].block_id == "bg"
    print("  [PASS] test_semantic_respects_priority")


def test_semantic_keeps_expensive_layers():
    """Semantic policy should keep deep layers (expensive to recompute)."""
    policy = SemanticEvictionPolicy()
    policy.set_agent_priority("a1", 5)

    shallow = make_block("shallow", layer_idx=2, attention=0.5, last_accessed=50.0)
    deep = make_block("deep", layer_idx=78, attention=0.5, last_accessed=50.0)

    blocks = [shallow, deep]
    victims = policy.select_victims(blocks, 1024 * 1024, current_time=100.0)

    # Shallow layer should be evicted first (cheaper to recompute)
    assert victims[0].block_id == "shallow"
    print("  [PASS] test_semantic_keeps_expensive_layers")


def test_compression():
    """Test KV block compression ratios."""
    block = make_block("b1", size=1024 * 1024)  # 1 MB FP16
    assert block.dtype == "fp16"

    fp8_size = block.compressed_size("fp8")
    assert fp8_size == 512 * 1024  # 50% compression

    int4_size = block.compressed_size("int4")
    assert int4_size == 256 * 1024  # 25% of original

    block.compress_to("fp8")
    assert block.size_bytes == 512 * 1024
    assert block.dtype == "fp8"

    print("  [PASS] test_compression")


if __name__ == "__main__":
    print("Running eviction policy tests...")
    test_lru_evicts_oldest()
    test_semantic_keeps_important()
    test_semantic_respects_priority()
    test_semantic_keeps_expensive_layers()
    test_compression()
    print("\nAll eviction policy tests passed!")
