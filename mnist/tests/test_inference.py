from pathlib import Path

import pytest
from PIL import Image

from mnist.architecture import CnnConfig, MnistCNN
from mnist.inference import load_image_tensor, predict
from mnist.tests.fixtures import SMALL_CNN


def test_predict_and_load_image(tmp_path: Path) -> None:
    image_path = tmp_path / 'digit.png'
    Image.new('L', (40, 40), color=255).save(image_path)
    images = load_image_tensor(image_path)
    assert images.shape == (1, 1, 28, 28)
    model = MnistCNN(CnnConfig.from_config(SMALL_CNN)).eval()
    classes, probabilities = predict(model, images)
    assert classes.shape == (1,)
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)
