import json
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from core.config import Config, optional_int, require_str
from core.data import make_loader, train_split_index
from core.paths import resolve_repo_path
from core.snapshot import Record, load_or_build_snapshot
from core.training import TrainingConfig
from translation.tokenizer import TranslationTokenizer

SNAPSHOT_NAME = 'translation_pairs'
PAIR_FIELDS = ('src', 'tgt', 'src_lang', 'tgt_lang')
LOCAL_CORPUS_SOURCE = 'local_corpus'
TSV_SUFFIX = '.tsv'
JSONL_SUFFIX = '.jsonl'
TSV_HEADER_NAMES = frozenset({'src', 'source'})
TSV_COMMENT_PREFIX = '#'
MIN_TARGET_TOKENS = 1

EncodedPair = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PairBatch(NamedTuple):
    """Padding masks are `True` at padded positions."""

    src: torch.Tensor
    tgt_input: torch.Tensor
    tgt_labels: torch.Tensor
    src_pad_mask: torch.Tensor
    tgt_pad_mask: torch.Tensor


class TranslationDataset(Dataset[EncodedPair]):
    """Items are (source with language prefix, bos + target, target + eos), each at most `max_seq_len` long."""

    def __init__(
        self,
        pairs: Sequence[Mapping[str, str]],
        tokenizer: TranslationTokenizer,
        max_seq_len: int,
    ) -> None:
        if not pairs:
            raise ValueError('at least one translation pair is required')
        self.examples = [_encode_pair(pair, tokenizer, max_seq_len) for pair in pairs]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedPair:
        return self.examples[index]


def _encode_pair(
    pair: Mapping[str, str], tokenizer: TranslationTokenizer, max_seq_len: int
) -> EncodedPair:
    src_ids = tokenizer.encode_source(pair['src'], pair['tgt_lang'])[:max_seq_len]
    tgt_ids = tokenizer.encode(pair['tgt'])[: max(max_seq_len - 1, MIN_TARGET_TOKENS)]
    return (
        torch.tensor(src_ids, dtype=torch.long),
        torch.tensor([tokenizer.bos_id, *tgt_ids], dtype=torch.long),
        torch.tensor([*tgt_ids, tokenizer.eos_id], dtype=torch.long),
    )


def load_pairs(config: Config, settings: TrainingConfig) -> list[Record]:
    corpus_dir = resolve_repo_path(require_str(config, 'corpus_dir'))
    return load_or_build_snapshot(
        settings.data_dir,
        SNAPSHOT_NAME,
        description='translation',
        metadata={'source': LOCAL_CORPUS_SOURCE, 'corpus_dir': str(corpus_dir)},
        fields=PAIR_FIELDS,
        record_limit=optional_int(config, 'max_pairs'),
        cache_only=settings.dataset_cache_only,
        build=lambda: read_corpus(corpus_dir),
    )


def read_corpus(corpus_dir: Path) -> list[Record]:
    """Reads every `*.tsv` (src, tgt, src_lang, tgt_lang columns) and `*.jsonl` file in filename order."""
    paths = sorted(
        [*corpus_dir.glob(f'*{TSV_SUFFIX}'), *corpus_dir.glob(f'*{JSONL_SUFFIX}')]
    )
    pairs = [
        pair
        for path in paths
        for pair in (read_tsv(path) if path.suffix == TSV_SUFFIX else read_jsonl(path))
    ]
    if not pairs:
        raise ValueError(
            f'no translation corpus found. place parallel files in {corpus_dir} as '
            f'*{TSV_SUFFIX} (src<TAB>tgt<TAB>src_lang<TAB>tgt_lang) or '
            f'*{JSONL_SUFFIX} ({{"src", "tgt", "src_lang", "tgt_lang"}}) and '
            're-run training'
        )
    return pairs


def read_tsv(path: Path) -> Iterator[Record]:
    lines = path.read_text(encoding='utf-8').splitlines()
    for line_number, raw in enumerate(lines):
        line = raw.strip()
        columns = line.split('\t')
        if not line or line.startswith(TSV_COMMENT_PREFIX):
            continue
        if len(columns) < len(PAIR_FIELDS):
            continue
        if line_number == 0 and columns[0].lower() in TSV_HEADER_NAMES:
            continue
        pair = make_pair(dict(zip(PAIR_FIELDS, columns, strict=False)))
        if pair is not None:
            yield pair


def read_jsonl(path: Path) -> Iterator[Record]:
    with path.open(encoding='utf-8') as file:
        for raw in file:
            if not raw.strip():
                continue
            row: object = json.loads(raw)
            if not isinstance(row, dict):
                continue
            pair = make_pair({str(key): str(value) for key, value in row.items()})
            if pair is not None:
                yield pair


def make_pair(row: Mapping[str, str]) -> Record | None:
    """`None` when the source or target text is empty."""
    src = row.get('src', '').strip()
    tgt = row.get('tgt', '').strip()
    if not src or not tgt:
        return None
    return {
        'src': src,
        'tgt': tgt,
        'src_lang': row.get('src_lang', '').strip().lower(),
        'tgt_lang': row.get('tgt_lang', '').strip().lower(),
    }


def collate_pairs(batch: list[EncodedPair], pad_id: int) -> PairBatch:
    src, tgt_input, tgt_labels = (
        pad_sequence(list(column), batch_first=True, padding_value=pad_id)
        for column in zip(*batch, strict=True)
    )
    return PairBatch(src, tgt_input, tgt_labels, src == pad_id, tgt_input == pad_id)


def create_dataloaders(
    pairs: Sequence[Mapping[str, str]],
    tokenizer: TranslationTokenizer,
    *,
    max_seq_len: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
) -> tuple[DataLoader[EncodedPair], DataLoader[EncodedPair] | None]:
    """Splits at the pair level so train and validation never share a sentence; the train split keeps at least one pair."""
    split = max(train_split_index(len(pairs), val_fraction), 1)
    collate = partial(collate_pairs, pad_id=tokenizer.pad_id)
    train_loader = make_loader(
        TranslationDataset(pairs[:split], tokenizer, max_seq_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate,
    )
    val_pairs = pairs[split:]
    if not val_pairs:
        return train_loader, None
    val_loader = make_loader(
        TranslationDataset(val_pairs, tokenizer, max_seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate,
    )
    return train_loader, val_loader
