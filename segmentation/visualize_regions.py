import matplotlib.pyplot as plt

from pathlib import Path

def show_region_crops(region_crops):
    output_dir = Path("region_outputs")
    output_dir.mkdir(exist_ok=True)

    for i, crop in enumerate(region_crops):
        plt.figure()
        plt.title(f"Region {i}")
        plt.imshow(crop)
        plt.axis("off")
        plt.savefig(output_dir / f"region_{i}.png")
        plt.close()
