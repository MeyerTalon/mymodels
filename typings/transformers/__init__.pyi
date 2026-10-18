from typing import TypeAlias

import torch
from PIL import Image
from torch import nn

ChatContent: TypeAlias = list[dict[str, str]]
ChatMessage: TypeAlias = dict[str, str | ChatContent]

class BatchFeature(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> BatchFeature: ...

class PreTrainedModel(nn.Module):
    @property
    def device(self) -> torch.device: ...
    def generate(self, **kwargs: torch.Tensor | int | bool) -> torch.Tensor: ...

class ProcessorMixin:
    def apply_chat_template(
        self, conversation: list[ChatMessage], *, add_generation_prompt: bool
    ) -> str: ...
    def __call__(
        self, *, text: str, images: list[Image.Image], return_tensors: str
    ) -> BatchFeature: ...
    def batch_decode(
        self, sequences: torch.Tensor, *, skip_special_tokens: bool
    ) -> list[str]: ...

class AutoModelForImageTextToText:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str, *, dtype: torch.dtype
    ) -> PreTrainedModel: ...

class AutoProcessor:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> ProcessorMixin: ...
