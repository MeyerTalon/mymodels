from PIL.Image import Image
from torch import Tensor

class ToTensor:
    def __call__(self, picture: Image) -> Tensor: ...
