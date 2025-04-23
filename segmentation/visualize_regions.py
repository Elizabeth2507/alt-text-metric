import matplotlib.pyplot as plt

def show_region_crops(region_crops):
    for i, crop in enumerate(region_crops):
        plt.figure()
        plt.title(f"Region {i}")
        plt.imshow(crop)
        plt.axis("off")
    plt.show()
