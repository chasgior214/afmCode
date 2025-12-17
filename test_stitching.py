import matplotlib.pyplot as plt
import AFMImageCollection
import path_loader as pl
import stitching
import numpy as np


# Load images
print("Loading images...")
collection = AFMImageCollection.AFMImageCollection(pl.afm_images_path, pl.depressurized_datetime)
filtered_collection = collection.filter_images(image_range='0221-0227')
images = filtered_collection.images
print(f"Stitching {images[0].bname} to {images[-1].bname}")

if not images:
    print("No images found.")
    exit(1)

modes = ['average', 'latest', 'earliest']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

results = []
global_min = float('inf')
global_max = float('-inf')

for mode in modes:
    print(f"Stitching with mode: {mode}")
    stitched_data, x_coords, y_coords, extent = stitching.stitch_maps(images, overlap_mode=mode)
    results.append((mode, stitched_data, extent))
    
    if stitched_data is not None:
        with np.errstate(invalid='ignore'):
            local_min = np.nanmin(stitched_data)
            local_max = np.nanmax(stitched_data)
        if not np.isnan(local_min):
            global_min = min(global_min, local_min)
        if not np.isnan(local_max):
            global_max = max(global_max, local_max)

if global_min == float('inf'):
    global_min = 0
    global_max = 1

im = None
for i, (mode, stitched_data, extent) in enumerate(results):
    ax = axes[i]
    if stitched_data is None:
        ax.text(0.5, 0.5, "Stitching Failed", ha='center', va='center')
        ax.set_title(f"Mode: {mode}")
        continue
        
    im = ax.imshow(stitched_data, extent=extent, origin='lower', cmap='viridis', vmin=global_min, vmax=global_max)
    ax.set_title(f"Mode: {mode}")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    
if im:
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Height (nm)")
    
plt.show()

# show just the earliest image
fig, ax = plt.subplots(1, 1, figsize=(18, 6))
ax.imshow(results[2][1], extent=results[2][2], origin='lower', cmap='viridis', vmin=global_min, vmax=global_max)
ax.set_title("Mode: earliest")
ax.set_xlabel("X (um)")
ax.set_ylabel("Y (um)")
fig.colorbar(im, ax=ax, label="Height (nm)")
plt.show()
