from mnist.data import create_dataloaders
from mnist.tests.fixtures import digits


def test_create_dataloaders_shapes() -> None:
    train, val = create_dataloaders(digits(8), None, batch_size=4, num_workers=0)
    images, labels = next(iter(train))
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)
    assert val is None


def test_create_dataloaders_with_validation() -> None:
    _, val = create_dataloaders(digits(8), digits(3), batch_size=4, num_workers=0)
    assert val is not None
    assert len(val) == 1
