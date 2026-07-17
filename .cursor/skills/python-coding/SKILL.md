---
name: python-coding
description: >-
  Applies Python and PyTorch coding best practices when writing or editing
  Python. Trigger whenever the user asks to write, edit, refactor, or review
  Python/PyTorch code, invokes "/python-coding", or otherwise works in `.py`
  files in this repo. Use even when the request is implicit — if the task
  involves Python, apply it. Compose with other skills (e.g. concise) as needed.
---

# Python Coding

Write and edit Python that is clear, typed, documented, and consistent with this repo. Prefer existing patterns over inventing new ones. When PyTorch applies, follow PyTorch best practices.

## Rules

- **PEP 8 formatting.** Match PEP 8 (and existing repo style): naming, imports, spacing, line length. Prefer the project's established formatting over personal preference.
- **Docstrings on functions.** Every function and method gets a docstring that states its purpose. Document Args/Returns/Raises when that information is not obvious from the signature and types. This repo uses Google-style docstrings (`Args:` / `Returns:` / `Raises:` sections) — match that unless the surrounding file differs.
- **Lowercase comments and docstrings.** Write all comment and docstring prose in lowercase — inline comments, module/file headers, and function/class docstrings alike (e.g. `# load configuration`, `"""selects the best available device."""`). Keep casing only where it is meaningful: identifiers and code references (`DataLoader`, `nn.TransformerEncoder`, `FileNotFoundError`), proper nouns and acronyms (PyTorch, Wikipedia, BPE, CUDA), code literals (`True`, `None`), and Google docstring section headers (`Args:` / `Returns:` / `Raises:`). Applies to code files only — `.py` and comments in `.yaml` configs — never to Markdown or other documentation files, with one exception: the repo-root `README.md` is always written entirely in lowercase.
- **Type hints everywhere.** Annotate parameters, return types, and important locals/attributes. Prefer precise types over bare `Any` unless truly necessary. This repo uses `typing`-module generics (`Dict[str, Any]`, `Optional[torch.Tensor]`, `List[str]`) — match that rather than switching to lowercase built-in generics.
- **No redundant code.** Don't restate the obvious, keep dead code, wrap no-ops, or add comments that only repeat the code. Prefer the simplest correct form.
- **No unnecessary duplication.** Extract shared logic when the same behavior appears more than once (or clearly will). Don't abstract prematurely for a one-off.
- **Unit tests for critical functions.** When writing or changing a function whose failure would break training, inference, or data handling, add a lightweight unit test that confirms its core behavior. Tests must run in seconds: tiny inputs, tiny model dims, no network, no downloads, no training loops, no real weights. Use `pytest` (plain test functions with asserts — no test classes unless grouping helps). Put tests in `<pkg>/tests/test_<module>.py`. Skip tests for trivial glue or one-off scripts.
- **Follow existing patterns.** Mirror nearby modules for structure, naming, error handling, config loading, training loops, and package layout before introducing a new approach.
- **PyTorch when applicable.** Use `nn.Module` idioms, `batch_first` consistently with the rest of the model, correct device/dtype handling, `torch.no_grad()` for inference, shape-safe ops, and existing project conventions (e.g. native `nn.Transformer*` usage) rather than ad-hoc rewrites.

## Anti-goals

Don't skip types or docstrings to save time, don't copy-paste with slight variations, and don't "improve" working code into a different style than the surrounding package. Clean ≠ novel. Don't write heavy tests — anything that trains, downloads, or takes more than a few seconds defeats the purpose.
