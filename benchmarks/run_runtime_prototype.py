"""Measure the Nemorix FP8 GPU↔RAM/NVMe prototype with synthetic KV data."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path
from statistics import mean

from nemorix.policies.lru import LRUEvictionPolicy
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.runtime import RuntimeKVManager, TieredLayerStore, TorchFP8Codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="PyTorch destination/source device")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--critical-fraction", type=float, default=0.10)
    parser.add_argument("--ram-gib", type=float, default=8.0)
    parser.add_argument("--policy", choices=("lru", "retention"), default="retention")
    parser.add_argument("--nvme-directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu for software-path validation")
    if args.layers <= 0 or args.iterations <= 0:
        raise SystemExit("--layers and --iterations must be positive")

    policy = LRUEvictionPolicy() if args.policy == "lru" else SemanticEvictionPolicy(max_layers=args.layers)
    temporary = None
    nvme_directory = args.nvme_directory
    if nvme_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="nemorix-runtime-")
        nvme_directory = Path(temporary.name)
    store = TieredLayerStore(int(args.ram_gib * 1024**3), nvme_directory, policy)
    manager = RuntimeKVManager(
        TorchFP8Codec(),
        store,
        policy,
        device=args.device,
        critical_fraction=args.critical_fraction,
    )

    shape = (args.blocks, args.block_tokens, args.kv_heads, args.head_dim)
    reports: list[dict[str, object]] = []
    try:
        for iteration in range(args.iterations):
            agent_id = f"benchmark-{iteration}"
            layers = {
                layer: (
                    torch.randn(shape, dtype=torch.float16, device=args.device),
                    torch.randn(shape, dtype=torch.float16, device=args.device),
                )
                for layer in range(args.layers)
            }
            manager.register_agent(agent_id, layers, num_tokens=args.blocks * args.block_tokens)
            offload_events = manager.offload_agent(agent_id)
            handle = manager.resume_agent(agent_id)
            restored = handle.wait()
            squared_error = 0.0
            squared_signal = 0.0
            for layer_idx, restored_pair in restored.items():
                original_pair = layers[layer_idx]
                for original, recovered in zip(original_pair, restored_pair):
                    difference = recovered.float() - original.float()
                    squared_error += difference.square().sum().item()
                    squared_signal += original.float().square().sum().item()
            nrmse = (squared_error / max(squared_signal, 1e-30)) ** 0.5
            offload_ms = sum(event.elapsed_ms for event in offload_events)
            source_bytes = sum(event.bytes_moved for event in offload_events)
            offload_gib_s = (source_bytes / 1024**3) / max(offload_ms / 1000.0, 1e-30)
            page_in_gib_s = (handle.metrics.bytes_total / 1024**3) / max(
                handle.metrics.total_ms / 1000.0, 1e-30
            )
            reports.append(
                {
                    "phase": "cold" if iteration == 0 else "warm",
                    "offload_ms": offload_ms,
                    "critical_resume_ms": handle.metrics.critical_ms,
                    "complete_resume_ms": handle.metrics.total_ms,
                    "fp8_bytes": handle.metrics.bytes_total,
                    "critical_layers": handle.metrics.critical_layers,
                    "total_layers": handle.metrics.total_layers,
                    "source_tiers": sorted(handle.metrics.source_tiers),
                    "effective_offload_source_gib_s": offload_gib_s,
                    "effective_page_in_fp8_gib_s": page_in_gib_s,
                    "fp8_relative_rmse": nrmse,
                }
            )
    finally:
        manager.close()
        if temporary is not None:
            temporary.cleanup()

    result = {
        "mode": "gpu-transfer" if args.device.startswith("cuda") else "cpu-software-path",
        "device": args.device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "policy": args.policy,
        "shape_per_tensor": shape,
        "iterations": reports,
        "mean_offload_ms": mean(float(row["offload_ms"]) for row in reports),
        "mean_critical_resume_ms": mean(float(row["critical_resume_ms"]) for row in reports),
        "mean_complete_resume_ms": mean(float(row["complete_resume_ms"]) for row in reports),
        "mean_effective_offload_source_gib_s": mean(
            float(row["effective_offload_source_gib_s"]) for row in reports
        ),
        "mean_effective_page_in_fp8_gib_s": mean(
            float(row["effective_page_in_fp8_gib_s"]) for row in reports
        ),
        "mean_fp8_relative_rmse": mean(float(row["fp8_relative_rmse"]) for row in reports),
        "note": (
            "Offload throughput uses FP16 source bytes and includes FP8 conversion, allocation, "
            "copy, and synchronization; page-in throughput uses stored FP8 bytes and includes "
            "decode. Neither is raw bus bandwidth. CPU results validate only the software path."
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
