"""Tensor codec interfaces and the optional PyTorch FP8 implementation."""

from __future__ import annotations

from typing import Any, Protocol

from nemorix.runtime.models import EncodedTensor


class TensorCodec(Protocol):
    """Backend contract used by :class:`RuntimeKVManager`."""

    def encode(self, tensor: Any) -> EncodedTensor: ...

    def decode(self, encoded: EncodedTensor, device: str) -> Any: ...

    def synchronize(self, device: str | None = None) -> None: ...


class TorchFP8Codec:
    """Encode CUDA/CPU tensors as native E4M3 FP8 in host memory.

    This backend deliberately fails if native PyTorch FP8 is unavailable; it
    does not silently substitute INT8 and call it FP8. CUDA sources are copied
    through pinned host memory when possible. Restores return the original
    source dtype on the requested device.
    """

    def __init__(self, *, pin_memory: bool = True) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - hardware extra
            raise RuntimeError("Install Nemorix with the 'runtime' extra") from exc
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build has no native float8_e4m3fn dtype")
        self.torch = torch
        self.pin_memory = pin_memory

    def encode(self, tensor: Any) -> EncodedTensor:
        torch = self.torch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("TorchFP8Codec accepts torch.Tensor values only")
        source_dtype = str(tensor.dtype).removeprefix("torch.")
        try:
            quantized = tensor.detach().contiguous().to(torch.float8_e4m3fn)
        except RuntimeError as exc:
            device_name = (
                torch.cuda.get_device_name(tensor.device)
                if tensor.device.type == "cuda" and torch.cuda.is_available()
                else str(tensor.device)
            )
            raise RuntimeError(
                "Native E4M3 conversion failed on "
                f"{device_name}. A100 can validate GPU↔host transfers but has no FP8 "
                "Tensor Cores, and FP8 cast support depends on the PyTorch/CUDA build. "
                "Use a current supported build or H100/H200/B100 for native FP8 work."
            ) from exc
        use_pinned = self.pin_memory and quantized.device.type == "cuda"
        try:
            host = torch.empty(
                quantized.shape,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=use_pinned,
            )
        except RuntimeError:
            host = torch.empty(quantized.shape, dtype=torch.float8_e4m3fn, device="cpu")
        host.copy_(quantized, non_blocking=use_pinned)
        if use_pinned:
            torch.cuda.synchronize(quantized.device)
        return EncodedTensor(
            payload=host,
            shape=tuple(tensor.shape),
            source_dtype=source_dtype,
            storage_dtype="float8_e4m3fn",
            nbytes=host.numel() * host.element_size(),
        )

    def decode(self, encoded: EncodedTensor, device: str) -> Any:
        torch = self.torch
        if encoded.storage_dtype != "float8_e4m3fn":
            raise ValueError(f"Unsupported storage dtype: {encoded.storage_dtype}")
        dtype = getattr(torch, encoded.source_dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown source dtype: {encoded.source_dtype}")
        fp8 = encoded.payload.to(device=device, non_blocking=device.startswith("cuda"))
        restored = fp8.to(dtype=dtype)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        return restored

    def synchronize(self, device: str | None = None) -> None:
        torch = self.torch
        if torch.cuda.is_available() and (device is None or device.startswith("cuda")):
            torch.cuda.synchronize(device)
