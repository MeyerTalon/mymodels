from PIL import Image

from stellaris.screen import fit_width


def test_fit_width_downscales_keeping_aspect_ratio() -> None:
    fitted = fit_width(Image.new('RGBA', (2000, 1000)), 500)
    assert fitted.size == (500, 250)
    assert fitted.mode == 'RGB'


def test_fit_width_never_upscales() -> None:
    assert fit_width(Image.new('RGB', (300, 200)), 500).size == (300, 200)
