"""Measured RAM and NVMe storage for encoded KV layers."""

from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter

from nemorix.core.kv_block import KVBlock
from nemorix.runtime.models import StoredLayer, TransferEvent


class TieredLayerStore:
    """Keep encoded layers in RAM and spill selected records to NVMe.

    The store owns encoded host objects. NVMe records use pickle because native
    FP8 tensors are not safely reconstructible from a raw byte stream without
    shape/dtype metadata. Files are trusted local artifacts and must never be
    loaded from an untrusted directory.
    """

    def __init__(
        self,
        ram_capacity_bytes: int,
        nvme_directory: str | Path,
        policy: object,
    ) -> None:
        if ram_capacity_bytes <= 0:
            raise ValueError("ram_capacity_bytes must be positive")
        self.ram_capacity_bytes = ram_capacity_bytes
        self.nvme_directory = Path(nvme_directory)
        self.nvme_directory.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.ram: dict[str, StoredLayer] = {}
        self.nvme: dict[str, Path] = {}
        self.events: list[TransferEvent] = []

    @property
    def ram_used_bytes(self) -> int:
        return sum(record.nbytes for record in self.ram.values())

    def put(self, record: StoredLayer, current_time: float) -> None:
        old = self.ram.pop(record.record_id, None)
        if old is None:
            path = self.nvme.pop(record.record_id, None)
            if path is not None:
                path.unlink(missing_ok=True)
        required = max(0, self.ram_used_bytes + record.nbytes - self.ram_capacity_bytes)
        if required:
            self._spill(required, current_time)
        if record.nbytes > self.ram_capacity_bytes:
            self._write_nvme(record)
        else:
            record.tier = "ram"
            self.ram[record.record_id] = record

    def get(self, agent_id: str, layer_idx: int) -> StoredLayer:
        record_id = f"{agent_id}:{layer_idx}"
        record = self.ram.get(record_id)
        if record is not None:
            return record
        path = self.nvme.get(record_id)
        if path is None:
            raise KeyError(record_id)
        start = perf_counter()
        with path.open("rb") as stream:
            record = pickle.load(stream)
        elapsed = (perf_counter() - start) * 1000.0
        self.events.append(
            TransferEvent("read", agent_id, layer_idx, "nvme", "ram", path.stat().st_size, elapsed)
        )
        return record

    def layers_for(self, agent_id: str) -> list[int]:
        prefix = f"{agent_id}:"
        ids = set(self.ram) | set(self.nvme)
        return sorted(int(record_id.removeprefix(prefix)) for record_id in ids if record_id.startswith(prefix))

    def remove(self, agent_id: str, layer_idx: int) -> None:
        record_id = f"{agent_id}:{layer_idx}"
        self.ram.pop(record_id, None)
        path = self.nvme.pop(record_id, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def _spill(self, required_bytes: int, current_time: float) -> None:
        blocks = [self._as_block(record) for record in self.ram.values()]
        victims = self.policy.select_victims(blocks, required_bytes, current_time)
        for victim in victims:
            record = self.ram.pop(victim.block_id)
            self._write_nvme(record)

    def _write_nvme(self, record: StoredLayer) -> None:
        path = self.nvme_directory / f"{record.agent_id}-{record.layer_idx}.nkv"
        record.tier = "nvme"
        start = perf_counter()
        with path.open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = (perf_counter() - start) * 1000.0
        self.nvme[record.record_id] = path
        self.events.append(
            TransferEvent("write", record.agent_id, record.layer_idx, "ram", "nvme", path.stat().st_size, elapsed)
        )

    @staticmethod
    def _as_block(record: StoredLayer) -> KVBlock:
        return KVBlock(
            block_id=record.record_id,
            agent_id=record.agent_id,
            layer_idx=record.layer_idx,
            num_tokens=record.num_tokens,
            size_bytes=record.nbytes,
            dtype="fp8",
            importance_score=record.importance_score,
            last_accessed=record.last_accessed,
            tier="ram",
        )
