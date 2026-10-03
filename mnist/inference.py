import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from core.checkpoint import find_checkpoint, load_checkpoint
from core.device import Precision, autocast, select_device
from core.paths import resolve_repo_path
from mnist.architecture import IMAGE_SIDE_PX, CnnConfig, MnistCNN
from mnist.data import load_mnist

INFERENCE_PRECISION = Precision.FP16
INFERENCE_DROPOUT = 0.0
GRAYSCALE_MODE = 'L'


def load_model(model_name: str, weights_dir: Path, device: torch.device) -> MnistCNN:
    path = find_checkpoint(weights_dir, model_name)
    print(f'Loading model from {path}')
    checkpoint = load_checkpoint(path, device)
    settings = CnnConfig.from_config(
        {**checkpoint.config, 'dropout': INFERENCE_DROPOUT}
    )
    model = MnistCNN(settings).to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    return model


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """Any image becomes a (1, 1, 28, 28) grayscale tensor with values in [0, 1]."""
    image = (
        Image.open(image_path)
        .convert(GRAYSCALE_MODE)
        .resize((IMAGE_SIDE_PX, IMAGE_SIDE_PX))
    )
    tensor: torch.Tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0)


def predict(model: MnistCNN, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(batch, 1, 28, 28) images to (batch,) classes and (batch, num_classes) probabilities."""
    device = next(model.parameters()).device
    with torch.no_grad(), autocast(device, INFERENCE_PRECISION):
        probabilities = F.softmax(model(images.to(device)).float(), dim=-1)
    return probabilities.argmax(dim=-1), probabilities


def load_test_example(data_dir: Path, index: int) -> tuple[torch.Tensor, int]:
    image, label = load_mnist(data_dir, train=False, cache_only=True)[index]
    return image.unsqueeze(0), label


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Classify a digit using a trained MNIST model'
    )
    parser.add_argument('--model_name', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--index', type=int)
    parser.add_argument('--data_dir', default='mnist/data')
    parser.add_argument('--show_probs', action='store_true')
    parser.add_argument('--weights_dir', default='mnist/weights')
    args = parser.parse_args()
    try:
        model = load_model(
            args.model_name, resolve_repo_path(args.weights_dir), select_device()
        )
        if args.image is not None:
            images = load_image_tensor(resolve_repo_path(args.image))
            true_label = None
            description = str(args.image)
        else:
            images, true_label = load_test_example(
                resolve_repo_path(args.data_dir), args.index
            )
            description = f'test[{args.index}]'
    except (FileNotFoundError, TypeError, ValueError) as error:
        sys.exit(f'Error: {error}')
    classes, probabilities = predict(model, images)
    print(f'Source: {description}')
    print(f'Predicted class: {int(classes[0].item())}')
    if true_label is not None:
        print(f'True label: {true_label}')
    if args.show_probs:
        for digit, probability in enumerate(probabilities[0].tolist()):
            print(f'  {digit}: {probability:.4f}')


if __name__ == '__main__':
    main()
