
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to your PNG images
image_files = [
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-22 15-00-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 00-10-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 18-20-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 23-45-00.jpg',
]

# Corresponding date–time annotations
annotations = [
    "22 Jul 2022 15:10",
    "23 Jul 2022 00:10",
    "23 Jul 2022 18:20",
    "23 Jul 2022 23:40",
]

fig, axes = plt.subplots(2, 2, figsize=(21, 12))
axes = axes.flatten()
letter_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, img_path, text, letter in zip(axes, image_files, annotations, letter_labels):
    img = mpimg.imread(img_path)
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # Date/time annotation (bottom-right corner)
    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=26,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
    )

    # Letter label (top-right corner)
    ax.text(
        0.02, 0.97, letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=30,
        color="black",
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", linewidth=0.7)
    )

plt.tight_layout()
plt.savefig('/home/waynedj/Projects/swath_based_framework/figures/publication/Figure01_v001.png', dpi=500, bbox_inches='tight')


# %%
