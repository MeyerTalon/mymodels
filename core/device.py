from enum import StrEnum

import torch


class Precision(StrEnum):
    FP32 = 'fp32'
    FP16 = 'fp16'
    BF16 = 'bf16'


AUTOCAST_DTYPES: dict[Precision, torch.dtype] = {
    Precision.FP16: torch.float16,
    Precision.BF16: torch.bfloat16,
}


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def autocast_dtype(device: torch.device, precision: Precision) -> torch.dtype | None:
    """`None` on CPU whatever the precision: CPU autocast gains little and float16 is poorly supported there."""
    if device.type == 'cpu':
        return None
    return AUTOCAST_DTYPES.get(precision)


def autocast(device: torch.device, precision: Precision) -> torch.autocast:
    dtype = autocast_dtype(device, precision)
    return torch.autocast(
        device_type=device.type, dtype=dtype, enabled=dtype is not None
    )
