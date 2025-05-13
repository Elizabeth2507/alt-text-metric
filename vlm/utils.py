from pathlib import Path
from PIL import Image
from typing import List

def load_region_crops(folder: str = "region_crops") -> List[Image.Image]:
    folder_path = Path(folder)
    image_paths = sorted(folder_path.glob("region_*.png"))  # Ensure consistent order
    crops = [Image.open(p).convert("RGB") for p in image_paths]
    return crops


def load_full_image(path: str) -> Image.Image:
    """
    Loads a full image from the given file path.
    """
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {path}: {e}")
