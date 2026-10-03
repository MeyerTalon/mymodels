import pytest
import torch

from core.device import Precision, autocast_dtype


def test_cpu_never_autocasts() -> None:
    for precision in Precision:
        assert autocast_dtype(torch.device('cpu'), precision) is None


def test_accelerator_dtypes() -> None:
    device = torch.device('cuda')
    assert autocast_dtype(device, Precision.FP16) == torch.float16
    assert autocast_dtype(device, Precision.BF16) == torch.bfloat16
    assert autocast_dtype(device, Precision.FP32) is None


def test_unknown_precision_raises() -> None:
    with pytest.raises(ValueError):
        Precision('fp8')
