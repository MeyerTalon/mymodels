# mymodels

models of mine

## essential commands

run these commands from the repository root:

### uv environment

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# create or update .venv from the lockfile
uv sync
```

```bash
# add or remove runtime dependencies
uv add <package>
uv remove <package>
```

```bash
# add or remove development dependencies
uv add --dev <package>
uv remove --dev <package>
```

```bash
# upgrade all locked dependencies
uv lock --upgrade
uv sync
```

### tests

```bash
# run all tests
uv run pytest
```

```bash
# run only the wikipedia tests
uv run pytest wikipedia/tests
```

### wikipedia model

see [docs/README-wikipedia.md](docs/README-wikipedia.md) for architecture, configs, and usage details.

```bash
# train the wikipedia model
uv run python -m wikipedia.training wikipedia/configs/wikipedia_small.yaml
```

```bash
# generate text with trained weights
uv run python -m wikipedia.inference --model_name wikipedia_small --prompt "the history of"
```

