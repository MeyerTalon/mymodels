# AGENTS.md

Guidance for AI coding agents working in this repo.

## Overview

Personal PyTorch models, trained from scratch on consumer hardware (Apple silicon first, then CUDA, then CPU). Shared machinery lives in two packages; each model is a thin data and config layer on top:

- `core/`: everything model-agnostic: config validation, devices and precision, snapshots, tokenizer, sampling, checkpoints, reporting, and the `Trainer` base class.
- `gpt/`: the decoder-only transformer, packed token data, and the train and infer CLIs shared by every language model.
- `wikipedia/`, `shakespeare/`, `western/`: GPT language models that differ only in where their text comes from. `shakespeare/` can open a live activation view with `--show_activations`.
- `translation/`: an encoder-decoder transformer for multilingual translation (`<2xx>` target-language prefix).
- `mnist/`: a small CNN digit classifier.

`mise.toml` at the root pins the toolchain (Python 3.11, uv, ripgrep) and defines every task. Always go through mise rather than calling uv or python directly: tasks run from the repo root, where every relative path in a config resolves.

## Dune and the Golden Path

**The codebase is a pristine snapshot of the state agents should extend.**

This codebase is Dune. Agents working in it must follow the Golden Path: the strict paved paths set out below. The code is the living memory of how code here is written, and agents extend whatever patterns they find. Clean patterns produce clean code; anti-patterns and slop spread like weeds. Every change must leave the repo in a state you would want copied.

The user is the chef. Agents are the kitchen staff: they follow the paved path, they don't invent new ones. When the path doesn't cover a case, stop and ask instead of improvising.

### Paved-path rules

- **Copy the existing pattern.** Before writing anything, find the closest existing example (module, function, test, config, task) and match it. A new language model copies `wikipedia/`: a `data.py` with `load_texts`, two three-line entry points, configs, and tests. If no pattern exists, propose one to the user before inventing it.
- **Shared code lives in `core/` or `gpt/`.** If two packages need the same logic, it moves down a layer. Never copy a function between model packages.
- **No comments.** Not in code, configs, or stubs. Code explains itself through names, types, and structure. If something needs a comment to be understood, rewrite it.
- **Names and types are the documentation.** Units go in names (`timeout_s`, `side_px`, `grad_accum_steps`). Docstrings are allowed only for what the signature can't show: a reason, an invariant, a precondition, an ordering, a safety constraint, or a tensor shape. Never restate the name or signature.
- **No magic numbers.** A literal whose meaning isn't stated where it's used gets a named constant that carries its meaning and unit: thresholds (`MIN_WORK_CHARS`), sizes (`HASH_CHUNK_BYTES`), image geometry (`IMAGE_SIDE_PX`), init scales (`INIT_STD`), and sentinels. Never reach into a structure by position to mean something: name it or look it up. A literal is fine only where a name at the same spot already says what it is (a keyword argument, a config key, a dataclass field), as `0`/`1` identities, as format widths, or as test fixtures and expected values.
- **Hyperparameters come from configs.** Every knob is a required key in every config of its family, read through `core.config.require_*`. No defaults in code, no `config.get(key, fallback)`.
- **No workarounds.** No `# noqa`, `# type: ignore`, `cast`, `Any`, skipped or weakened tests, broad `except Exception`, or special cases that paper over a wrong design. Untyped third-party APIs get a stub in `typings/`. Unused code is deleted, not suppressed. Fix the root cause or stop and report it.
- **Never loosen enforcement.** Don't relax ruff, mypy, pytest, or the golden-path scan to make code pass. Tightening is welcome. Loosening needs explicit user approval.
- **Don't extend anti-patterns.** If the code you're touching contains one, fix it in the same change or flag it to the user. Never copy it.
- **`mise run check` is the gate.** It must pass before any change is done, and CI runs it on every push and pull request.

### Feature map and CLI

- The [Feature map](#feature-map) section is the index of every module and what it owns. Update it in the same change whenever you add, move, rename, or remove a module.
- `mise` is the CLI. Every runnable action is a mise task. A new model gets `train:<pkg>` and `infer:<pkg>` tasks, and they go in [Commands](#commands).

### When an agent gets corrected

Every correction means the paved path has a gap. Fix the path, not just the instance. Change one or more of these, preferring the earliest layer that can catch the mistake:

1. **Codebase**: fix the code so the right pattern is the one sitting there to be copied.
2. **Static analysis**: a lint, type, test, or scan that makes the mistake fail `mise run check`. Lints live in `pyproject.toml`; the comment and suppression scan is the `lint:golden-path` task in `mise.toml`.
3. **Rules / Bugbot**: `.cursor/BUGBOT.md`, review rules for what static analysis can't catch.
4. **Skills**: `.agents/skills/` for workflows and judgment calls.
5. **Style guide**: this file, as a last resort.

Machines beat prose. When the user corrects you, fix the instance and propose the layer change that prevents it next time.

### Skills

Skills live in `.agents/skills/<name>/SKILL.md`. `.claude/skills` and `.cursor/skills` are symlinks to that tree (checked by `lint:golden-path`), so never create a real copy elsewhere. To add one, copy `_template`.

- `ponytail`: the laziest solution that works. Applies to every coding task.
- `python-coding`: rules for every `.py` file.
- `ml-coding`: model, data, training, checkpoint, and inference rules. Composes with `python-coding`.
- `concise`: maximally brief replies on request.
- `push`: add, commit, and push. Explicit invocation only.

## Commands

```bash
mise run setup                     # uv sync into .venv
mise run check                     # lint + test: run before finishing any change
mise run fmt                       # ruff format
mise run train:<pkg> <config>      # e.g. mise run train:wikipedia wikipedia/configs/wikipedia_small.yaml
mise run infer:<pkg> <args>        # see below
mise tasks                         # list all tasks
```

`<pkg>` is one of `mnist`, `shakespeare`, `translation`, `western`, `wikipedia`. Inference examples:

```bash
mise run infer:wikipedia --model_name wikipedia_small --prompt "The history of"
mise run infer:shakespeare --model_name shakespeare_small --prompt "To be, or not to be" --show_activations
mise run infer:mnist --model_name mnist_small --index 0 --show_probs
mise run infer:translation --model_name translation_small --source_lang en --target_lang es --prompt "hello world"
```

Language-model inference also takes `--max_length`, `--temperature`, `--top_k` (1 is greedy), and `--weights_dir`. `--model_name` is the checkpoint prefix in `<pkg>/weights/`; inference loads `_best.pt`, then `_latest.pt`, then the bare name.

`check` runs `lint` and `test`:

- `lint`: `ruff format --check`, `ruff check`, `mypy` (strict, stubs from `typings/`), and `lint:golden-path`.
- `lint:golden-path`: fails on any `#` comment in `.py`, `.pyi`, and `.yaml` files or in `mise.toml`, `pyproject.toml`, and `.github/`, and on skill dirs that aren't symlinks to `.agents/skills`.
- `test`: `pytest` over every `<pkg>/tests/`. Tests use tiny models, temp dirs, and fakes: no network, no real weights, a few seconds total.

Add dependencies with `uv add` (or `uv add --dev`), never by hand-editing `uv.lock`.

## Configuration

Each model is driven by one YAML file in `<pkg>/configs/`. Every key is required. All configs share the `TrainingConfig` keys in `core/training.py` (optimizer, schedule, precision, loader, and output dirs). Each family adds its own:

| Family | Extra keys |
|---|---|
| GPT (`wikipedia`, `shakespeare`, `western`) | `vocab_size`, `min_frequency`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `max_seq_len`, `dropout` |
| `wikipedia` | `number_of_articles`, `dataset_*`, `shuffle_buffer_size` |
| `shakespeare` | `corpus_url`, `max_works` |
| `western` | `corpus_dir`, `max_works` |
| `translation` | `vocab_size`, `min_frequency`, `languages`, encoder-decoder dims, `corpus_dir`, `max_pairs` |
| `mnist` | `in_channels`, `conv1_channels`, `conv2_channels`, `hidden_dim`, `num_classes`, `dropout`, `dataset_seed` |

`expected_parameters` is each config's trainable-parameter count, checked by a test that builds the model on the `meta` device. It assumes the tokenizer fills `vocab_size`; a small corpus can train a smaller vocabulary and a smaller model. Change it whenever you change dims, depth, or `vocab_size`.

### Hardware notes

- Configs are sized for Apple silicon with about 24GB of unified memory.
- `precision: bf16` everywhere: bf16 has float32's exponent range, so it stays stable on MPS without a gradient scaler, where fp16 can diverge. CPU always runs float32.
- Effective batch is `batch_size × grad_accum_steps`. The `medium` and `large` configs trade a small `batch_size` for accumulation to fit memory (64 sequences per step).
- `large` is GPT-2-base scale and is heavy for that memory budget.
- `max_works` / `max_pairs` / `number_of_articles` bound smoke runs; `dataset_cache_only: true` proves a run works offline from the snapshot.

## Feature map

```
mise.toml                  toolchain and tasks (the CLI)
pyproject.toml             deps, ruff, mypy, pytest
typings/                   stubs for datasets and torchvision
.agents/skills/            agent skills (symlinked from .claude/ and .cursor/)
.cursor/BUGBOT.md          review rules
.github/workflows/check.yml  CI: `mise run check`
core/
  paths.py                 REPO_ROOT, repo-relative paths, tokenizer dirs
  config.py                YAML loading and typed required-key accessors
  device.py                device selection, Precision, autocast
  snapshot.py              checksummed JSONL corpus snapshots with manifests
  tokenizer.py             byte-level BPE, special tokens, TextTokenizer protocol
  sampling.py              temperature and top-k sampling
  weights.py               GPT-style weight init
  checkpoint.py            checkpoint naming, lookup, and loading
  reporting.py             per-run loss and accuracy chart
  data.py                  DataLoader factory, train/val split
  training.py              TrainingConfig, Trainer base, optimizer and schedule
  tests/factories.py       minimal training config for tests
gpt/
  architecture.py          DecoderOnlyTransformer
  data.py                  packed token blocks and loaders
  training.py              LanguageModelTrainer, train CLI
  inference.py             model loading, text generation, infer CLI
  tests/fakes.py           CharTokenizer
wikipedia/data.py          Hugging Face streaming sample
shakespeare/data.py        Gutenberg download and play/poem splitting
shakespeare/visualization.py  optional live activation view for inference
western/data.py            local *.txt novels
translation/
  tokenizer.py             TranslationTokenizer with <2xx> tokens
  architecture.py          EncoderDecoderTransformer
  data.py                  TSV/JSONL pairs, padded batches
  training.py              TranslationTrainer, train CLI
  inference.py             translate, infer CLI
mnist/
  architecture.py          MnistCNN
  data.py                  torchvision MNIST, seeded val split
  training.py              ClassifierTrainer, train CLI
  inference.py             predict from an image or test index
docs/<pkg>.md              per-model usage notes
```

Each model package's `training.py` and `inference.py` stay thin: parse args, call the shared code. Per-package `data/`, `weights/`, and `tokenizer_files/` are gitignored and regenerated by training; `reports/` is tracked.

## Safety rules

- **Keep checkpoints loadable.** Module attribute names, `nn.Sequential` order, and checkpoint keys (`epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `loss`, `accuracy`, `config`, `tokenizer_vocab_size`) are a compatibility surface for existing weights. Changing them needs explicit user approval.
- **Keep snapshots valid.** Snapshot metadata keys (for example the `WikipediaSource` fields) are compared against existing manifests; renaming one silently forces a re-download.
- **Never commit gitignored artifacts** (data, weights, tokenizer files). Reports are tracked on purpose.
- **Leave changes local by default.** Don't `git add`, commit, push, or open pull requests unless the user explicitly asks for that action. Finishing a task, passing `mise run check`, or being told to "make" or "fix" something is not an instruction to do any of them.
- Never add yourself as a contributor, and never mention yourself in a commit message or pull request. No `Co-authored-by` trailers for an agent, model, or tool.
- Don't expand scope beyond the asked change.

## Conventions

- Follow the `python-coding` and `ml-coding` skills for every `.py` file; `ponytail` applies to every coding task. Those skills are the detailed rules; don't duplicate them here.
- Strings use single quotes (enforced by `ruff format`).
- The repo-root `README.md` is always written entirely in lowercase.
- Docs can lag code; when they disagree, trust the code and fix the doc in the same change.
