"""Dependency-free tests for the measured runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.runtime.codec import TensorCodec
from nemorix.runtime.manager import RuntimeKVManager
from nemorix.runtime.models import EncodedTensor, TransferEvent
from nemorix.runtime.store import TieredLayerStore


@dataclass
class FakeTensor:
    data: bytes
    dtype: str = "float16"

    @property
    def nbytes(self) -> int:
        return len(self.data)


class FakeFP8Codec(TensorCodec):
    """Predictable 2:1 codec used where PyTorch/CUDA is unavailable."""

    def encode(self, tensor: FakeTensor) -> EncodedTensor:
        payload = tensor.data[::2]
        return EncodedTensor(payload, (len(tensor.data),), tensor.dtype, "float8_e4m3fn", len(payload))

    def decode(self, encoded: EncodedTensor, device: str) -> FakeTensor:
        return FakeTensor(encoded.payload * 2, encoded.source_dtype)

    def synchronize(self, device: str | None = None) -> None:
        return None


def make_manager(tmp_path, *, capacity: int = 1024, critical_fraction: float = 0.1):
    policy = LRUEvictionPolicy()
    store = TieredLayerStore(capacity, tmp_path, policy)
    manager = RuntimeKVManager(
        FakeFP8Codec(), store, policy, device="cpu", critical_fraction=critical_fraction
    )
    return manager, store


def make_layers(count: int, size: int = 16):
    return {
        index: (FakeTensor(bytes([index]) * size), FakeTensor(bytes([index + 1]) * size))
        for index in range(count)
    }


def test_offload_and_partial_resume(tmp_path):
    manager, store = make_manager(tmp_path)
    manager.register_agent("agent", make_layers(20), num_tokens=128)

    events = manager.offload_agent("agent")
    assert len(events) == 20
    assert not manager.resident["agent"]
    assert store.ram_used_bytes == 20 * 16  # K/V each compress from 16 to 8 bytes.

    handle = manager.resume_agent("agent")
    assert sorted(handle.critical) == [0, 1]
    assert handle.metrics.critical_layers == 2
    all_layers = handle.wait()

    assert len(all_layers) == 20
    assert handle.metrics.bytes_critical == 32
    assert handle.metrics.bytes_total == 320
    assert handle.metrics.critical_ms <= handle.metrics.total_ms
    assert handle.metrics.source_tiers == {"ram"}
    manager.close()


def test_ram_capacity_spills_lru_records_to_nvme(tmp_path):
    manager, store = make_manager(tmp_path, capacity=16)
    manager.register_agent("old", make_layers(2), num_tokens=16, current_time=1.0)
    manager.offload_agent("old", current_time=1.0)

    assert store.ram_used_bytes == 16
    assert len(store.nvme) == 1
    assert any(event.operation == "write" for event in store.events)

    handle = manager.resume_agent("old", critical_layers=2)
    assert len(handle.wait()) == 2
    assert "nvme" in handle.metrics.source_tiers
    assert any(event.operation == "read" for event in store.events)
    manager.close()


def test_gpu_capacity_uses_lru_selection(tmp_path):
    manager, _ = make_manager(tmp_path)
    manager.gpu_capacity_bytes = 128
    manager.register_agent("old", make_layers(2), num_tokens=16, current_time=1.0)
    manager.register_agent("new", make_layers(2), num_tokens=16, current_time=2.0)

    victims = manager.ensure_gpu_capacity(32, current_time=3.0)

    assert victims == ["old:0"]
    assert 0 not in manager.resident["old"]
    assert 0 in manager.resident["new"]
    manager.close()


def test_gpu_capacity_uses_retention_law_importance(tmp_path):
    policy = SemanticEvictionPolicy()
    store = TieredLayerStore(1024, tmp_path, policy)
    manager = RuntimeKVManager(
        FakeFP8Codec(), store, policy, device="cpu", gpu_capacity_bytes=64
    )
    manager.register_agent(
        "agent",
        make_layers(2),
        num_tokens=16,
        importance={0: 0.0, 1: 1.0},
        current_time=1.0,
    )

    victims = manager.ensure_gpu_capacity(32, current_time=2.0)

    assert victims == ["agent:0"]
    assert 0 not in manager.resident["agent"]
    assert 1 in manager.resident["agent"]
    manager.close()


def test_missing_agent_and_validation(tmp_path):
    manager, _ = make_manager(tmp_path)
    with pytest.raises(KeyError):
        manager.resume_agent("missing")
    with pytest.raises(ValueError):
        RuntimeKVManager(FakeFP8Codec(), manager.store, manager.policy, critical_fraction=0)
    manager.close()


def test_transfer_event_bandwidth():
    event = TransferEvent("copy", "a", 0, "gpu", "ram", 1024**3, 1000.0)
    assert event.bandwidth_gbps == pytest.approx(1.0)
