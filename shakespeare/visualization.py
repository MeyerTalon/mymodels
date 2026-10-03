from collections.abc import Callable
from typing import Self, TypeVar

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import nn

from core.tokenizer import TextTokenizer
from gpt.architecture import DecoderOnlyTransformer

LENS_TOP_TOKENS = 5
ATTENTION_CONTEXT_TOKENS = 24
LENS_HISTORY_ROWS = 12
TOKEN_LABEL_CHARS = 8
FIGURE_WIDTH_IN = 10.0
FIGURE_HEIGHT_IN = 8.0
FRAME_PAUSE_S = 0.001
PANEL_ROWS = 3
PANEL_COLS = 1
LENS_PANEL = 1
ATTENTION_PANEL = 2
HISTORY_PANEL = 3
LAST_QUERY = -1
HEADED_ATTENTION_RANK = 4
T = TypeVar('T')


class ActivationView:
    """forward hooks exist only between attach and detach."""

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer: TextTokenizer,
        *,
        live: bool,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._live = live
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_modules: list[
            tuple[nn.TransformerEncoderLayer, nn.MultiheadAttention]
        ] = []
        self._attention: list[torch.Tensor | None] = []
        self._lens: list[list[tuple[str, float]] | None] = []
        self.lens_history: list[list[str]] = []
        self.latest_attention: torch.Tensor | None = None
        self.latest_labels: list[str] = []
        self._figure: Figure | None = None
        self._lens_axis: Axes | None = None
        self._attention_axis: Axes | None = None
        self._history_axis: Axes | None = None

    def attach(self) -> None:
        if self._handles:
            return
        layers = self._model.encoder.layers
        self._attention = [None] * len(layers)
        self._lens = [None] * len(layers)
        for layer_index, layer in enumerate(layers):
            if not isinstance(layer, nn.TransformerEncoderLayer):
                raise TypeError('encoder layer is not a TransformerEncoderLayer')
            self._capture_attention(layer_index, layer)
            self._handles.append(
                layer.register_forward_hook(self._layer_hook(layer_index))
            )
        self._handles.append(self._model.register_forward_hook(self._model_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer, attention in self._attention_modules:
            layer.add_module('self_attn', attention)
        self._attention_modules.clear()

    def block(self) -> None:
        if self._figure is None:
            return
        plt.ioff()
        plt.show(block=True)

    def __enter__(self) -> Self:
        self.attach()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.detach()

    def _capture_attention(
        self, layer_index: int, layer: nn.TransformerEncoderLayer
    ) -> None:
        attention = layer.self_attn

        def record(weights: torch.Tensor) -> None:
            self._attention[layer_index] = weights

        capture = _AttentionCapture(attention, record)
        if not self._model.training:
            capture.eval()
        layer.add_module('self_attn', capture)
        self._attention_modules.append((layer, attention))

    def _layer_hook(
        self, layer_index: int
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError('encoder layer output is not a tensor')
            hidden = output[0, LAST_QUERY].detach().float()
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                logits = self._model.head(self._model.ln_f(hidden))
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probabilities, LENS_TOP_TOKENS)
            choices: list[tuple[str, float]] = []
            for position in range(LENS_TOP_TOKENS):
                token_id = int(indices[position].item())
                probability = float(values[position].item())
                choices.append((_token_label(self._tokenizer, token_id), probability))
            self._lens[layer_index] = choices

        return hook

    def _model_hook(
        self, _module: nn.Module, inputs: tuple[object, ...], _output: object
    ) -> None:
        token_ids = inputs[0]
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError('decoder input is not a token tensor')
        lens = _rows(self._lens, 'logit lens')
        attention_rows = _rows(self._attention, 'attention')
        self.lens_history.append([choices[0][0] for choices in lens])
        self.latest_attention = attention_rows[-1]
        shown = self.latest_attention[:, -ATTENTION_CONTEXT_TOKENS:]
        labels = [
            _token_label(self._tokenizer, int(token_id))
            for token_id in token_ids[0, -ATTENTION_CONTEXT_TOKENS:].tolist()
        ]
        self.latest_labels = labels[-shown.size(-1) :]
        self.latest_attention = shown
        if self._live:
            self._render(lens)

    def _render(self, lens: list[list[tuple[str, float]]]) -> None:
        lens_axis, attention_axis, history_axis = self._axes()
        lens_axis.clear()
        lens_lines = [
            f'L{layer}  '
            + '  '.join(f'{token} {probability:.2f}' for token, probability in choices)
            for layer, choices in enumerate(lens)
        ]
        lens_axis.text(0.01, 0.99, '\n'.join(lens_lines), va='top', family='monospace')
        lens_axis.set_title('logit lens at the newest position')
        lens_axis.axis('off')
        attention = self.latest_attention
        if attention is None:
            raise RuntimeError('attention was not recorded')
        attention_axis.clear()
        attention_axis.imshow(attention.numpy(), aspect='auto', vmin=0.0, vmax=1.0)
        attention_axis.set_yticks(
            range(attention.size(0)),
            [f'head {index}' for index in range(attention.size(0))],
        )
        attention_axis.set_xticks(range(len(self.latest_labels)), self.latest_labels)
        attention_axis.set_title('last-layer attention from the newest token')
        history_axis.clear()
        history_lines = []
        visible = self.lens_history[-LENS_HISTORY_ROWS:]
        for step, row in enumerate(
            visible, start=len(self.lens_history) - len(visible) + 1
        ):
            guesses = ' '.join(f'L{layer}:{token}' for layer, token in enumerate(row))
            history_lines.append(f'{step}  {guesses}')
        history_axis.text(
            0.01,
            0.99,
            '\n'.join(history_lines),
            va='top',
            family='monospace',
        )
        history_axis.set_title('top guess per layer')
        history_axis.axis('off')
        figure = self._figure
        if figure is None:
            raise RuntimeError('figure was not created')
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(FRAME_PAUSE_S)

    def _axes(self) -> tuple[Axes, Axes, Axes]:
        if (
            self._figure is not None
            and self._lens_axis is not None
            and self._attention_axis is not None
            and self._history_axis is not None
        ):
            return self._lens_axis, self._attention_axis, self._history_axis
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=True
        )
        lens_axis = figure.add_subplot(PANEL_ROWS, PANEL_COLS, LENS_PANEL)
        attention_axis = figure.add_subplot(PANEL_ROWS, PANEL_COLS, ATTENTION_PANEL)
        history_axis = figure.add_subplot(PANEL_ROWS, PANEL_COLS, HISTORY_PANEL)
        self._figure = figure
        self._lens_axis = lens_axis
        self._attention_axis = attention_axis
        self._history_axis = history_axis
        plt.ion()
        return lens_axis, attention_axis, history_axis


class _AttentionCapture(nn.Module):
    def __init__(
        self,
        attention: nn.MultiheadAttention,
        record: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.attention = attention
        self._record = record
        self.batch_first = attention.batch_first
        self.num_heads = attention.num_heads
        self._qkv_same_embed_dim = attention._qkv_same_embed_dim

    @property
    def in_proj_bias(self) -> torch.Tensor | None:
        return self.attention.in_proj_bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, weights = self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        if (
            not isinstance(weights, torch.Tensor)
            or weights.ndim != HEADED_ATTENTION_RANK
        ):
            raise RuntimeError('attention weights were not returned per head')
        self._record(weights[0, :, LAST_QUERY, :].detach().float().cpu())
        return output, weights


def _token_label(tokenizer: TextTokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id]).replace('\n', ' ').strip()
    if text == '':
        return str(token_id)
    return text[:TOKEN_LABEL_CHARS]


def _rows(rows: list[T | None], kind: str) -> list[T]:
    recorded: list[T] = []
    for row in rows:
        if row is None:
            raise RuntimeError(f'a layer did not record {kind}')
        recorded.append(row)
    return recorded
