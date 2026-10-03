import json
from pathlib import Path

import pytest

from core.snapshot import (
    Record,
    load_or_build_snapshot,
    load_snapshot,
    manifest_path,
    snapshot_path,
    write_snapshot,
)

NAME = 'works'
METADATA = {'corpus_url': 'https://example.com/corpus.txt'}
FIELDS = ('id', 'text')
RECORDS = [{'id': str(index), 'text': f'text {index}'} for index in range(3)]


def _load_or_build(
    directory: Path,
    *,
    record_limit: int | None = None,
    cache_only: bool = False,
    records: list[Record] = RECORDS,
) -> list[Record]:
    return load_or_build_snapshot(
        directory,
        NAME,
        description='test',
        metadata=METADATA,
        fields=FIELDS,
        record_limit=record_limit,
        cache_only=cache_only,
        build=lambda: records,
    )


def test_round_trip_and_limit(tmp_path: Path) -> None:
    write_snapshot(tmp_path, NAME, RECORDS, METADATA)
    assert (
        load_snapshot(tmp_path, NAME, METADATA, fields=FIELDS, record_limit=None)
        == RECORDS
    )
    limited = load_snapshot(tmp_path, NAME, METADATA, fields=FIELDS, record_limit=2)
    assert limited == RECORDS[:2]
    assert (
        load_snapshot(tmp_path, NAME, METADATA, fields=FIELDS, record_limit=4) is None
    )


def test_metadata_mismatch_and_tampering_invalidate(tmp_path: Path) -> None:
    write_snapshot(tmp_path, NAME, RECORDS, METADATA)
    other = {'corpus_url': 'https://example.com/other.txt'}
    assert (
        load_snapshot(tmp_path, NAME, other, fields=FIELDS, record_limit=None) is None
    )
    snapshot_path(tmp_path, NAME).write_text(
        '{"id": "9", "text": "x"}\n', encoding='utf-8'
    )
    assert (
        load_snapshot(tmp_path, NAME, METADATA, fields=FIELDS, record_limit=None)
        is None
    )


def test_manifest_records_count_and_checksum(tmp_path: Path) -> None:
    write_snapshot(tmp_path, NAME, RECORDS, METADATA)
    manifest = json.loads(manifest_path(tmp_path, NAME).read_text(encoding='utf-8'))
    assert manifest['record_count'] == len(RECORDS)
    assert manifest['corpus_url'] == METADATA['corpus_url']
    assert len(manifest['sha256']) == 64


def test_load_or_build_caches_full_build_and_reuses_it(tmp_path: Path) -> None:
    assert _load_or_build(tmp_path, record_limit=1) == RECORDS[:1]
    assert _load_or_build(tmp_path, cache_only=True) == RECORDS


def test_load_or_build_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no compatible test snapshot'):
        _load_or_build(tmp_path, cache_only=True)
    with pytest.raises(ValueError, match='only found'):
        _load_or_build(tmp_path, record_limit=5)
    with pytest.raises(ValueError, match='positive'):
        _load_or_build(tmp_path, record_limit=0)
    assert not snapshot_path(tmp_path, NAME).exists()
