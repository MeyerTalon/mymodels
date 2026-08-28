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

```bash
# run only the shakespeare tests
uv run pytest shakespeare/tests
```

```bash
# run only the mnist tests
uv run pytest mnist/tests
```

```bash
# run only the western tests
uv run pytest western/tests
```

```bash
# run only the translation tests
uv run pytest translation/tests
```

### wikipedia model

see [docs/wikipedia.md](docs/wikipedia.md) for architecture, configs, and usage details.
training streams a bounded sample from `wikimedia/wikipedia` on first use and reuses the cached local snapshot afterward.

```bash
# train the wikipedia model
uv run python -m wikipedia.training wikipedia/configs/wikipedia_small.yaml
```

```bash
# generate text with trained weights
uv run python -m wikipedia.inference --model_name wikipedia_small --prompt "the history of"
```

### shakespeare model

see [docs/shakespeare.md](docs/shakespeare.md) for architecture, configs, and usage details.
training downloads the project gutenberg complete works on first use and reuses the cached local snapshot afterward.

```bash
# train the shakespeare model
uv run python -m shakespeare.training shakespeare/configs/shakespeare_small.yaml
```

```bash
# generate text with trained weights
uv run python -m shakespeare.inference --model_name shakespeare_small --prompt "to be, or not to be"
```

### mnist model

see [docs/mnist.md](docs/mnist.md) for architecture, configs, and usage details.
training downloads mnist via torchvision on first use and reuses the local cache afterward. there is no tokenizer.

```bash
# train the mnist classifier
uv run python -m mnist.training mnist/configs/mnist_small.yaml
```

```bash
# classify a local image
uv run python -m mnist.inference --model_name mnist_small --image path/to/digit.png --show_probs
```

```bash
# classify a test-set example (requires a cached download)
uv run python -m mnist.inference --model_name mnist_small --index 0 --show_probs
```

### western model

see [docs/western.md](docs/western.md) for architecture, configs, and usage details.
put plain-text novels in `western/data/corpus/` (one `.txt` file per work) before training. nothing is downloaded.

```bash
# train the western model
uv run python -m western.training western/configs/western_small.yaml
```

```bash
# generate text with trained weights
uv run python -m western.inference --model_name western_small --prompt "the wind came down off the high plains"
```

### translation model

see [docs/translation.md](docs/translation.md) for architecture, configs, and usage details.
put parallel files in `translation/data/corpus/` as `*.tsv` (`src`, `tgt`, `src_lang`, `tgt_lang`) or `*.jsonl` before training. nothing is downloaded.

```bash
# train the translation model
uv run python -m translation.training translation/configs/translation_small.yaml
```

```bash
# translate with trained weights
uv run python -m translation.inference --model_name translation_small --source_lang en --target_lang es --prompt "hello world"
```

