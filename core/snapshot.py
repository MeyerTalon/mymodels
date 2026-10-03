import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

HASH_CHUNK_BYTES = 1024 * 1024
SNAPSHOT_SUFFIX = '.jsonl'
MANIFEST_SUFFIX = '.manifest.json'
SHA256_KEY = 'sha256'
RECORD_COUNT_KEY = 'record_count'

Record = dict[str, str]
SnapshotMetadata = Mapping[str, str | int | None]


def snapshot_path(directory: Path, name: str) -> Path:
    return directory / f'{name}{SNAPSHOT_SUFFIX}'


def manifest_path(directory: Path, name: str) -> Path:
    return directory / f'{name}{MANIFEST_SUFFIX}'


def load_or_build_snapshot(
    directory: Path,
    name: str,
    *,
    description: str,
    metadata: SnapshotMetadata,
    fields: Sequence[str],
    record_limit: int | None,
    cache_only: bool,
    build: Callable[[], list[Record]],
) -> list[Record]:
    """Persists every built record, so a later run asking for more than `record_limit` can reuse the snapshot."""
    if record_limit is not None and record_limit <= 0:
        raise ValueError(f'{description} record limit must be positive')
    cached = load_snapshot(
        directory, name, metadata, fields=fields, record_limit=record_limit
    )
    if cached is not None:
        print(f'Loaded {len(cached)} records from {snapshot_path(directory, name)}')
        return cached
    if cache_only:
        raise ValueError(
            f'dataset_cache_only is set but no compatible {description} snapshot '
            f'exists in {directory}'
        )
    records = build()
    if record_limit is not None and len(records) < record_limit:
        raise ValueError(
            f'requested {record_limit} {description} records but only found '
            f'{len(records)}'
        )
    write_snapshot(directory, name, records, metadata)
    print(f'Cached {len(records)} records in {snapshot_path(directory, name)}')
    return records if record_limit is None else records[:record_limit]


def load_snapshot(
    directory: Path,
    name: str,
    metadata: SnapshotMetadata,
    *,
    fields: Sequence[str],
    record_limit: int | None,
) -> list[Record] | None:
    """`None` unless the manifest matches `metadata`, the checksum matches, and enough records exist."""
    snapshot = snapshot_path(directory, name)
    manifest = manifest_path(directory, name)
    if not snapshot.is_file() or not manifest.is_file():
        return None
    try:
        stored: object = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(stored, dict):
            return None
        if any(stored.get(key) != value for key, value in metadata.items()):
            return None
        if stored.get(SHA256_KEY) != file_sha256(snapshot):
            return None
        records = _read_records(snapshot, fields, record_limit)
    except (OSError, TypeError, ValueError):
        return None
    if record_limit is not None and len(records) < record_limit:
        return None
    return records


def _read_records(
    path: Path, fields: Sequence[str], record_limit: int | None
) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding='utf-8') as file:
        for line in file:
            if record_limit is not None and len(records) == record_limit:
                break
            records.append(_parse_record(line, fields))
    return records


def _parse_record(line: str, fields: Sequence[str]) -> Record:
    row: object = json.loads(line)
    if not isinstance(row, dict):
        raise TypeError('snapshot rows must be JSON objects')
    record = {str(key): value for key, value in row.items()}
    if not all(isinstance(record.get(field), str) for field in fields):
        raise ValueError(f'snapshot row is missing one of {list(fields)}')
    return {str(key): str(value) for key, value in record.items()}


def write_snapshot(
    directory: Path,
    name: str,
    records: Sequence[Mapping[str, str]],
    metadata: SnapshotMetadata,
) -> None:
    """Writes temporary files and renames them, so a crash never leaves a manifest vouching for a partial snapshot."""
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_path(directory, name)
    manifest = manifest_path(directory, name)
    snapshot_tmp = _temporary_path(directory, snapshot.name)
    manifest_tmp = _temporary_path(directory, manifest.name)
    try:
        with snapshot_tmp.open('w', encoding='utf-8') as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
        manifest_body = {
            **metadata,
            RECORD_COUNT_KEY: len(records),
            SHA256_KEY: file_sha256(snapshot_tmp),
        }
        manifest_tmp.write_text(
            json.dumps(manifest_body, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        snapshot_tmp.replace(snapshot)
        manifest_tmp.replace(manifest)
    finally:
        snapshot_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _temporary_path(directory: Path, filename: str) -> Path:
    descriptor, path = tempfile.mkstemp(prefix=f'.{filename}.', dir=directory)
    os.close(descriptor)
    return Path(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()
