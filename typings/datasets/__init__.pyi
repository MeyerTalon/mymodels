from collections.abc import Callable, Iterator
from typing import Literal, TypeAlias

Row: TypeAlias = dict[str, object]

class IterableDataset:
    def filter(self, function: Callable[[Row], bool]) -> IterableDataset: ...
    def shuffle(self, *, seed: int, buffer_size: int) -> IterableDataset: ...
    def take(self, n: int) -> IterableDataset: ...
    def __iter__(self) -> Iterator[Row]: ...

def load_dataset(
    path: str,
    name: str,
    *,
    split: str,
    revision: str | None,
    streaming: Literal[True],
) -> IterableDataset: ...
