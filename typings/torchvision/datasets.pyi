from collections.abc import Callable

from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset

class MNIST(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        root: str,
        *,
        train: bool,
        download: bool,
        transform: Callable[[Image], Tensor],
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[Tensor, int]: ...
