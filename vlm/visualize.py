import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import os


def show_attribution_overlay(
    image: Image.Image,
    attribution: torch.Tensor,
    alpha: float = 0.8,
    save_path: str = "attribution_overlay.png",
    show: bool = True
):
    """
    Overlay a heatmap (H, W) over a PIL image and save to file.

    Args:
        image (PIL.Image): Input image.
        attribution (torch.Tensor): Tensor of shape (H, W), normalized.
        alpha (float): Heatmap opacity.
        save_path (str): Where to save the output image.
        show (bool): Whether to also display the figure.
    """
    image = image.convert("RGB").resize(attribution.shape[::-1])
    heatmap = attribution.to(dtype=torch.float32).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="inferno", alpha=alpha)
    plt.axis("off")
    plt.title("Attribution Overlay")
    plt.tight_layout()

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    if show:
        plt.show()
    plt.close()
