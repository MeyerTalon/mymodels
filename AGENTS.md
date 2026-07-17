# Agent instructions

Personal PyTorch models repo. Primary package today: `wikipedia/` (decoder-only transformer for Wikipedia sentence completion).

## Environment

- Conda env from `environment.yaml` (Python 3.10 + PyTorch)
- Prefer existing package layout over new packaging/CI unless asked

## Commands

```bash
# train
python wikipedia/training.py wikipedia/configs/wikipedia_small.yaml

# infer (see wikipedia/inference.py and docs/README-wikipedia.md)
```

## Conventions

- One model package per top-level directory (e.g. `wikipedia/`)
- Docs live in `docs/README-*.md`
- Training configs in `*/configs/*.yaml`
- Artifacts: `data/`, `weights/`, `tokenizer_files/` under each package

## Do not

- Invent Docker, CI, or packaging scaffolding unless requested
- Casually commit large weight/data churn
- Expand scope beyond the asked change

## Skills

Task-specific workflows live in:

- `.claude/skills/` (Claude Code)
- `.cursor/skills/` (Cursor)

Copy `_template` when adding a new skill; rename the directory to match the skill `name`.
