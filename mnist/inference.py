"""inference script for classifying MNIST digits from trained models.

loads a checkpoint and classifies either a single image path or one example
from the MNIST test set. prints the predicted class and optional probabilities.
"""

import argparse
import contextlib
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from mnist.architecture import MnistCNN
from mnist.data import load_mnist_datasets
from mnist.utils import resolve_repo_path, select_autocast_dtype, select_device


def load_model(
    model_name: str,
    weights_dir: str = "mnist/weights",
    device: Optional[torch.device] = None,
) -> Tuple[MnistCNN, torch.device]:
    """loads a trained CNN from a checkpoint.

    Args:
        model_name: base name of the model checkpoint files.
        weights_dir: directory containing model weights.
        device: explicit device to load the model on. if ``None``, the best
            available device is selected (MPS, then CUDA, then CPU).

    Returns:
        a tuple ``(model, device)`` where ``model`` is a ``MnistCNN`` in
        evaluation mode.

    Raises:
        FileNotFoundError: if no checkpoint file is found.
    """
    if device is None:
        device = select_device()

    weights_dir = resolve_repo_path(weights_dir)

    checkpoint_paths = [
        os.path.join(weights_dir, f"{model_name}_best.pt"),
        os.path.join(weights_dir, f"{model_name}_latest.pt"),
        os.path.join(weights_dir, f"{model_name}.pt"),
    ]

    checkpoint_path: Optional[str] = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Model checkpoint not found. Tried: {checkpoint_paths}"
        )

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = MnistCNN(
        in_channels=config.get("in_channels", 1),
        conv1_channels=config.get("conv1_channels", 32),
        conv2_channels=config.get("conv2_channels", 64),
        hidden_dim=config.get("hidden_dim", 128),
        num_classes=config.get("num_classes", 10),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def load_image_tensor(image_path: str) -> torch.Tensor:
    """loads a path as a single MNIST-shaped tensor.

    Args:
        image_path: path to an image file (any size; converted to 28x28 grayscale).

    Returns:
        float tensor of shape (1, 1, 28, 28) with values in ``[0, 1]``.
    """
    image = Image.open(image_path).convert("L").resize((28, 28))
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0)


def predict(
    model: MnistCNN,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """classifies a batch of images.

    Args:
        model: trained ``MnistCNN`` in eval mode.
        images: float tensor of shape (batch, 1, 28, 28).

    Returns:
        a tuple ``(predicted_classes, probabilities)`` with shapes (batch,)
        and (batch, num_classes).
    """
    device = next(model.parameters()).device
    images = images.to(device)
    amp_dtype = select_autocast_dtype(device, "fp16")
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        logits = model(images)
        probs = F.softmax(logits.float(), dim=-1)
        classes = probs.argmax(dim=-1)
    return classes, probs


def main() -> None:
    """CLI entry point for classifying a digit from a trained model.

    Command-line arguments (``argparse``):
        --model_name: checkpoint prefix in the weights dir to load (required).
        --image: path to an image file to classify (optional; one of
            ``--image`` or ``--index`` is required).
        --index: 0-based index into the MNIST test set (optional).
        --data_dir: MNIST cache directory used with ``--index``
            (default ``mnist/data``).
        --show_probs: if set, print class probabilities (flag, default off).
        --weights_dir: directory containing model weights
            (default ``mnist/weights``).
    """
    parser = argparse.ArgumentParser(
        description="Classify a digit using a trained MNIST model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to load (used as checkpoint prefix)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file to classify",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index into the MNIST test set",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="mnist/data",
        help="Directory containing the MNIST cache (used with --index)",
    )
    parser.add_argument(
        "--show_probs",
        action="store_true",
        help="Print class probabilities",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="mnist/weights",
        help="Directory containing model weights",
    )

    args = parser.parse_args()

    if args.image is None and args.index is None:
        print("error: provide --image PATH or --index N", file=sys.stderr)
        sys.exit(1)

    try:
        model, device = load_model(args.model_name, args.weights_dir)

        true_label: Optional[int] = None
        if args.image is not None:
            images = load_image_tensor(resolve_repo_path(args.image))
            source = args.image
        else:
            _train, _val, test_set = load_mnist_datasets(
                data_dir=resolve_repo_path(args.data_dir),
                val_fraction=0.0,
                dataset_cache_only=True,
            )
            image, label = test_set[args.index]
            images = image.unsqueeze(0)
            true_label = int(label)
            source = f"test[{args.index}]"

        classes, probs = predict(model, images)
        predicted = int(classes[0].item())
        print(f"Source: {source}")
        print(f"Predicted class: {predicted}")
        if true_label is not None:
            print(f"True label: {true_label}")
        if args.show_probs:
            for digit, prob in enumerate(probs[0].tolist()):
                print(f"  {digit}: {prob:.4f}")

    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
