# Shakespeare, with a live activation view

A copy of the Shakespeare language-model stack. Training and the default inference path match `shakespeare/`. Pass `--show_activations` to open a matplotlib window that updates on every generated token.

The window has three panels:

- the logit lens at the newest position, one row per layer
- last-layer attention from that position onto the recent context
- the running top guess from each layer

Hooks are attached only for that run. Omit the flag and this package does not register them.

Configs are the Shakespeare configs, including `shakespeare/data`, `shakespeare/weights`, and `shakespeare/tokenizer_files`, so a model trained with either package loads in the other.

```bash
mise run train:shakespeare-visualized shakespeare-visualized/configs/shakespeare_small.yaml
mise run infer:shakespeare-visualized --model_name shakespeare_small --prompt "To be, or not to be"
mise run infer:shakespeare-visualized --model_name shakespeare_small --prompt "To be, or not to be" --show_activations
```

The package directory is `shakespeare_visualized`. `shakespeare-visualized` is a symlink to it, because a Python package name cannot contain a hyphen.
