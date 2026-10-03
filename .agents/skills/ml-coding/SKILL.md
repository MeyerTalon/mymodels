---
name: ml-coding
description: >-
  This repo's ML structure and practice: shared core/ and gpt/ layers with
  thin model packages, native PyTorch building blocks, required-key YAML
  configs with expected_parameters, the Trainer base class, compatible
  checkpoints and snapshots, strict-eval inference, and consumer-hardware
  defaults. Trigger whenever the user invokes "/ml-coding" or asks to add,
  edit, or review models, training, inference, data pipelines, tokenizers,
  configs, or weights handling, even when the request is implicit. Composes
  with python-coding (style) and ponytail.
---

# ML coding

Models here are trained from scratch on consumer hardware and share one set
of machinery. Python style comes from `python-coding`; this skill covers ML
structure and practice.

## Rules

- **Add a model by subtraction.** A new decoder-only LM is a package with
  `data.py` (`load_texts(config, settings) -> list[str]`), three-line
  `training.py` and `inference.py` calling `gpt.training.train_from_cli` and
  `gpt.inference.run_inference`, configs, tests, `train:`/`infer:` mise tasks,
  a `.gitignore` entry for its artifacts, and a `docs/<pkg>.md`. Anything else
  subclasses `core.training.Trainer` and implements only `compute_batch`.
- **Native PyTorch first.** Compose `nn.TransformerEncoder`,
  `nn.TransformerDecoder`, `nn.Embedding`, `nn.LayerNorm`. Hand-roll a
  component only when no built-in exists, and say why in its docstring.
- **Shape discipline.** `batch_first=True` everywhere. `forward` returns raw
  logits and documents its tensor shapes; sampling lives in `generate` via
  `core.sampling.sample_next_token`.
- **Configs are the only source of hyperparameters.** Every key is required in
  every config of its family and read through `core.config.require_*`. A new
  knob goes into every config in the family in the same change.
- **`expected_parameters` stays true.** Each config states its parameter count,
  checked by a `test_configs` test that builds the model on the `meta` device.
  Update it whenever dims, depth, or `vocab_size` change.
- **Data is snapshotted.** Corpora go through
  `core.snapshot.load_or_build_snapshot`: checksummed, atomic, reusable
  offline with `dataset_cache_only: true`. Snapshot metadata keys are a
  compatibility surface; renaming one forces a rebuild.
- **Checkpoints are a compatibility surface.** `Trainer.save_checkpoint` writes
  `_latest`, `_best`, and `_epoch_N` files containing the config and, through
  `checkpoint_extras`, tokenizer metadata. Don't rename module attributes,
  reorder `nn.Sequential`s, or change checkpoint keys without explicit user
  approval: existing weights stop loading.
- **Inference is strict eval.** Rebuild from the checkpoint's own config with
  dropout forced to 0, `load_state_dict`, `eval()`, `torch.no_grad()`, and
  autocast at `INFERENCE_PRECISION`. Sampling knobs are CLI flags.
- **Consumer hardware.** `core.device.select_device()` picks MPS, then CUDA,
  then CPU. Size configs for about 24GB of unified memory, reach larger
  effective batches with `grad_accum_steps`, and use `bf16` on accelerators.
- **Verify by exercising.** After a meaningful change: `mise run check`, a
  greedy (`--top_k 1`) inference run against existing weights compared
  before and after, and a one-epoch training run from a temp copy of a config
  with `dataset_cache_only: true` and output dirs under `/tmp`.

## Anti-goals

Don't introduce new ML frameworks or tooling (Lightning, Hydra, wandb, the
Hugging Face `Trainer`, DDP) unless asked. Don't add a code-level default for a
hyperparameter, skip checkpoint metadata, or put sampling in `forward`. Don't
let a local "best practice" override the shared layer: consistency across
packages beats local optimality.
