from core.tokenizer import TextTokenizer
from gpt.architecture import DecoderOnlyTransformer
from gpt.inference import ActivationSession, run_inference


def _activation_view(
    model: DecoderOnlyTransformer, tokenizer: TextTokenizer
) -> ActivationSession:
    """Imports the view here so a normal run does not load matplotlib."""
    from shakespeare.visualization import ActivationView

    return ActivationView(model, tokenizer, live=True)


def main() -> None:
    run_inference(package='shakespeare', activation_view=_activation_view)


if __name__ == '__main__':
    main()
