from PIL import Image, ImageGrab


def fit_width(image: Image.Image, max_width_px: int) -> Image.Image:
    """Preserves aspect ratio and never upscales."""
    rgb = image.convert('RGB')
    if rgb.width <= max_width_px:
        return rgb
    height_px = round(rgb.height * max_width_px / rgb.width)
    return rgb.resize((max_width_px, height_px))


def capture_screen(max_width_px: int) -> Image.Image:
    """On macOS the terminal needs Screen Recording permission, or the grab shows only the wallpaper."""
    return fit_width(ImageGrab.grab(), max_width_px)
