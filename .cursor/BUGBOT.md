# Bugbot rules

This codebase is Dune: agents must follow the Golden Path in `AGENTS.md`. `mise run check` enforces what static analysis can catch. Review for what it can't, and flag every violation below as a blocking issue.

## Compatibility

- Flag renamed module attributes, reordered `nn.Sequential` layers, or changed checkpoint keys (`epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `loss`, `accuracy`, `config`, `tokenizer_vocab_size`) unless the PR description calls out that existing weights stop loading.
- Flag changes to snapshot metadata keys, snapshot names, or tokenizer special-token order unless the PR calls out the forced rebuild.
- Flag a changed architecture, `vocab_size`, or depth without a matching `expected_parameters` update.
- Flag gitignored artifacts (data, weights, tokenizer files) being committed.

## Golden Path

- Flag any loosening of enforcement: removed or relaxed lints in `pyproject.toml`, weakened tests, or changes to the `lint:golden-path` scan.
- Flag hyperparameters hardcoded in code or given code-level defaults instead of being required config keys, and new config keys missing from any config in their family.
- Flag logic duplicated between model packages instead of living in `core/` or `gpt/`, new helpers that duplicate an existing one, and model-package entry points that do more than parse args and call shared code.
- Flag any docstring that restates its item's name or signature, and any unit carried in a doc instead of the name. Docs are allowed only for a reason, an invariant, a precondition, an ordering, a safety constraint, or a tensor shape.
- Flag magic numbers: any literal outside tests whose meaning isn't stated where it's used, such as thresholds, sizes, kernel widths, init scales, indexes, and sentinels. Each needs a named constant carrying its meaning and unit. Ruff `PLR2004` catches only comparisons.
- Flag workarounds: `cast`, `Any`, broad `except`, or special cases that hide a bug instead of fixing it.
- Flag tests that touch the network, real weights, the repo's `tokenizer_files/`, or anything outside `tmp_path`.
- Flag modules added, moved, or removed without a matching update to the Feature map in `AGENTS.md`, and mise tasks added without a matching update to Commands.
- Flag skills edited outside `.agents/skills/`.
