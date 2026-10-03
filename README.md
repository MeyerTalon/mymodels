# mymodels

models of mine

## setup

[mise](https://mise.jdx.dev) installs python, uv, and ripgrep and runs every task from the repository root.

```bash
mise install
mise run setup
```

```bash
# lint, type-check, and test everything
mise run check
```

```bash
# list every task
mise tasks
```

dependencies are managed with uv: `uv add <package>` or `uv add --dev <package>`, never by editing `uv.lock` by hand.

## models

each model trains with `mise run train:<model> <config>` and runs with `mise run infer:<model> <args>`.

### wikipedia

see [docs/wikipedia.md](docs/wikipedia.md). training streams a bounded sample from `wikimedia/wikipedia` on first use and reuses the cached snapshot afterward.

```bash
mise run train:wikipedia wikipedia/configs/wikipedia_small.yaml
mise run infer:wikipedia --model_name wikipedia_small --prompt "the history of"
```

### shakespeare

see [docs/shakespeare.md](docs/shakespeare.md). training downloads the project gutenberg complete works on first use and reuses the cached snapshot afterward.

```bash
mise run train:shakespeare shakespeare/configs/shakespeare_small.yaml
mise run infer:shakespeare --model_name shakespeare_small --prompt "to be, or not to be"
mise run infer:shakespeare --model_name shakespeare_small --prompt "to be, or not to be" --show_activations
```

### mnist

see [docs/mnist.md](docs/mnist.md). training downloads mnist via torchvision on first use and reuses the local cache afterward.

```bash
mise run train:mnist mnist/configs/mnist_small.yaml
mise run infer:mnist --model_name mnist_small --image path/to/digit.png --show_probs
mise run infer:mnist --model_name mnist_small --index 0 --show_probs
```

### western

see [docs/western.md](docs/western.md). put plain-text novels in `western/data/corpus/` (one `.txt` file per work) before training. nothing is downloaded.

```bash
mise run train:western western/configs/western_small.yaml
mise run infer:western --model_name western_small --prompt "the wind came down off the high plains"
```

### translation

see [docs/translation.md](docs/translation.md). put parallel files in `translation/data/corpus/` as `*.tsv` (`src`, `tgt`, `src_lang`, `tgt_lang`) or `*.jsonl` before training. nothing is downloaded.

```bash
mise run train:translation translation/configs/translation_small.yaml
mise run infer:translation --model_name translation_small --source_lang en --target_lang es --prompt "hello world"
```
