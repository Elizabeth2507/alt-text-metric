from PIL import Image
import numpy as np
from pathlib import Path

from segmentation.segment_image import load_sam_model, run_sam_on_image, run_sam_with_points, run_sam_with_box, extract_region_crops
from segmentation.visualize_regions import show_region_crops

if __name__ == "__main__":
    image_path = Path("data/school_bus.jpg")
    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    image_np = np.array(image)

    model = load_sam_model()
    masks = segment_image(model, image_np)
    region_crops = extract_region_crops(image_np, masks)
    #
    show_region_crops(region_crops)
