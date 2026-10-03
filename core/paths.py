from pathlib import Path

PROJECT_MARKER = 'pyproject.toml'
TOKENIZER_DIRNAME = 'tokenizer_files'

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / PROJECT_MARKER).is_file()
)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def tokenizer_dir(package: str) -> Path:
    return REPO_ROOT / package / TOKENIZER_DIRNAME
