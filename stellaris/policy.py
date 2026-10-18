from dataclasses import dataclass

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

from core.config import Config, require_int, require_str
from core.device import Precision, autocast_dtype

ChatMessage = dict[str, str | list[dict[str, str]]]


@dataclass(frozen=True)
class PolicyConfig:
    base_model_id: str
    precision: Precision
    instruction: str
    max_new_tokens: int
    screenshot_width_px: int

    @classmethod
    def from_config(cls, config: Config) -> 'PolicyConfig':
        settings = cls(
            base_model_id=require_str(config, 'base_model_id'),
            precision=Precision(require_str(config, 'precision')),
            instruction=require_str(config, 'instruction'),
            max_new_tokens=require_int(config, 'max_new_tokens'),
            screenshot_width_px=require_int(config, 'screenshot_width_px'),
        )
        if settings.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be at least 1')
        if settings.screenshot_width_px < 1:
            raise ValueError('screenshot_width_px must be at least 1')
        return settings


@dataclass(frozen=True)
class Policy:
    settings: PolicyConfig
    model: PreTrainedModel
    processor: ProcessorMixin


def load_policy(settings: PolicyConfig, device: torch.device) -> Policy:
    """Downloads the base model from the Hugging Face Hub on first use."""
    model = AutoModelForImageTextToText.from_pretrained(
        settings.base_model_id,
        dtype=autocast_dtype(device, settings.precision) or torch.float32,
    )
    processor = AutoProcessor.from_pretrained(settings.base_model_id)
    return Policy(settings, model.to(device).eval(), processor)


def propose_action(policy: Policy, screenshot: Image.Image) -> str:
    messages: list[ChatMessage] = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': policy.settings.instruction},
            ],
        }
    ]
    prompt = policy.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = policy.processor(text=prompt, images=[screenshot], return_tensors='pt').to(
        policy.model.device
    )
    with torch.no_grad():
        output_ids = policy.model.generate(
            **inputs, max_new_tokens=policy.settings.max_new_tokens, do_sample=False
        )
    prompt_length = inputs['input_ids'].shape[-1]
    replies: list[str] = policy.processor.batch_decode(
        output_ids[:, prompt_length:], skip_special_tokens=True
    )
    return replies[0].strip()
