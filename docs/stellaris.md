# Stellaris agent

A pretrained small vision-language model from the Hugging Face Hub looks at a screenshot of Stellaris and proposes the next action as text. Unlike the other packages nothing here is trained from scratch: v1 will fine-tune the base model (LoRA or RL) on top of this loop.

## Configs

| Config | Base model | Notes |
|---|---|---|
| `stellaris_small` | `HuggingFaceTB/SmolVLM-256M-Instruct` | bf16 on MPS, greedy decoding |

Keys: `base_model_id`, `precision`, `instruction` (the prompt sent with each screenshot), `max_new_tokens`, and `screenshot_width_px` (captures wider than this are downscaled).

## Usage

```bash
mise run infer:stellaris stellaris/configs/stellaris_small.yaml
```

The base model downloads into the Hugging Face cache on first use. Screen capture uses macOS `screencapture`, so the terminal running mise needs Screen Recording permission (System Settings, Privacy & Security). The proposed action is only printed; nothing is sent to the game yet.
