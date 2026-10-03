---
name: python-coding
description: >
  Rules every piece of Python in this repo must follow: no comments or
  suppressions, names and types as the documentation (units in names,
  docstrings only for what the signature can't show), no magic numbers,
  shared code in core/ and gpt/, strict types with no Any, single quotes, and
  fast offline tests. Use this skill whenever writing, editing, refactoring,
  or reviewing any Python in this repo (core/, gpt/, any model package,
  tests, typings/ stubs), even for a small fix, and even if the user doesn't
  mention style or conventions.
---

# Python coding rules

These rules apply to every `.py` and `.pyi` file. The goal is code that the
next model package imports and builds on instead of copy-pasting.

Repo context: Python 3.11 (`TypeVar` and `Generic`, not PEP 695 syntax), `uv`
for deps, ruff for format and lint (single quotes), mypy `--strict` with stubs
in `typings/`, pytest. `mise run check` enforces type hints, no `Any`, import
order, keyword-only booleans, and no comments. The rules below cover what the
tools can't judge.

## 0. No comments, no suppressions

- No `#` comments anywhere. If code needs one, rename or restructure it.
- No `# noqa`, `# type: ignore`, `cast`, or config changes that loosen ruff or
  mypy. Fix the code.
- An untyped library call gets a typed local (`token_ids: list[int] = ...`)
  or a stub in `typings/`, declaring only what the code uses.

## 1. Layers

- `core/` is model-agnostic; `gpt/` is everything shared by decoder-only
  language models; a model package holds only what is unique to it, usually
  its data source.
- Entry points are thin: `training.py` and `inference.py` parse args, load the
  config, and call shared code. Copy `wikipedia/training.py` for a language
  model, `mnist/training.py` for anything else.
- Extract on the second copy, into the lowest layer both callers share.
- Read config once at the edge (`TrainingConfig.from_config`,
  `DecoderConfig.from_config`) and pass typed values down. Library functions
  take paths, settings, and devices as parameters, never globals.
- Return data; let callers decide whether to print, save, or plot it.

## 2. Types

- Config sections are frozen dataclasses with a `from_config` classmethod that
  reads every key through `core.config.require_*` and validates ranges.
- Closed sets are `StrEnum`s (`Precision`), not `str`.
- Fixed-shape batches are `NamedTuple`s (`PairBatch`) or tuple aliases
  (`TokenBatch`); generic code takes a `TypeVar`.
- Structural interfaces are `Protocol`s (`TextTokenizer`, `BatchLoader`), so
  tests can pass fakes.
- Boolean parameters are keyword-only.
- Bad input types raise `TypeError`; bad values raise `ValueError`. CLIs catch
  `FileNotFoundError`, `TypeError`, and `ValueError` and exit with the message.

## 3. Names and types are the documentation

- A name says what a thing is, and carries its unit: `timeout_s`,
  `IMAGE_SIDE_PX`, `HASH_CHUNK_BYTES`, `max_new_tokens`.
- A test name states the behavior it checks
  (`test_future_tokens_do_not_affect_past_logits`).
- Docstrings are the exception, not the default. Write one only for what the
  signature can't show, in one or two plain sentences:
  - a reason (`autocast_dtype`: why CPU never autocasts);
  - an invariant (`PackedDataset`: targets are inputs shifted by one);
  - a precondition (`EncoderDecoderTransformer.generate`: the source carries
    the language prefix);
  - an ordering (`find_checkpoint`: best, then latest, then bare);
  - a safety constraint (`write_snapshot`: why writes are atomic);
  - a tensor shape (`DecoderOnlyTransformer.forward`).
- Never restate the name or signature, and no `Args:` / `Returns:` sections.

## 4. No magic numbers

- A literal whose meaning isn't stated where it's used becomes a module-level
  constant whose name carries the meaning and unit: thresholds
  (`MIN_WORK_CHARS`), sizes (`HASH_CHUNK_BYTES`), geometry (`CONV_KERNEL_PX`),
  scales (`INIT_STD`, `MIN_TEMPERATURE`), and defaults (`DEFAULT_TOP_K`).
- Derive related constants from each other
  (`FEATURE_SIDE_PX = IMAGE_SIDE_PX // POOL_KERNEL_PX**POOL_COUNT`).
- Never reach into a structure by position to mean something: name it or look
  it up, as `REPO_ROOT` finds the directory holding `pyproject.toml`.
- A literal is fine where a name at the same spot already says what it is (a
  keyword argument, a config key, a dataclass field), as `0`/`1` identities,
  as format widths, and as test fixtures and expected values.
- Ruff `PLR2004` enforces this for comparisons (relaxed in `tests/` only).
  Everything else is on you and Bugbot.

## 5. Tests

- Tests live in `<pkg>/tests/test_<module>.py` and run in seconds: tiny dims,
  `tmp_path` for every written file, CPU, no network, no real weights.
- Use the shared helpers: `core.tests.factories.training_config` for configs,
  `gpt.tests.fakes.CharTokenizer` for language models. Fake network
  boundaries with `monkeypatch` (as `wikipedia/tests/test_data.py` does).
- Every non-trivial branch in parsing, splitting, masking, or snapshot logic
  gets a test.

## Before finishing

Run `mise run fmt` and `mise run check`. Then re-read the diff: nothing
duplicated that could have been imported, units in names, no magic numbers,
no docstring that restates its item, new modules and tasks added to the
Feature map and Commands in `AGENTS.md`.
