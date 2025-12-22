import numpy as np
from scipy.interpolate import RegularGridInterpolator
import membrane_relative_positions as mrp
from AFMImage import compute_x_pixel_coords, compute_y_pixel_coords
from datetime import datetime

def stitch_maps(images, pixel_size=None, overlap_mode='average'):
    """
    Stitch together multiple AFM maps into a single large map.

    Args:
        images (list): List of AFMImage objects.
        pixel_size (float, optional): Pixel size in microns. If None, uses median of inputs.
        overlap_mode (str): Strategy for overlapping regions.
            'average': Average values.
            'latest': Later images overwrite earlier ones.
            'earliest': Earlier images take precedence.

    Returns:
        tuple: (stitched_data, x_coords, y_coords, extent)
            stitched_data (np.ndarray): 2D array of height values (nm).
            x_coords (np.ndarray): 1D array of absolute x coordinates (um).
            y_coords (np.ndarray): 1D array of absolute y coordinates (um).
            extent (tuple): (min_x, max_x, min_y, max_y) for plotting.
    """
    if not images:
        return None, None, None, None

    # 1. Determine Global Bounds
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    image_bounds = []
    for img in images:
        bounds = mrp.image_bounds_absolute_positions(img)
        image_bounds.append(bounds)
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    # 2. Determine Pixel Size
    if pixel_size is None:
        pixel_sizes = [img.get_pixel_size() for img in images]
        pixel_size = np.median(pixel_sizes)

    # 3. Create Global Grid
    # Add a small buffer to ensure coverage
    width_um = max_x - min_x
    height_um = max_y - min_y
    
    n_x = int(np.ceil(width_um / pixel_size))
    n_y = int(np.ceil(height_um / pixel_size))
    
    x_coords = min_x + compute_x_pixel_coords(n_x, pixel_size)
    y_coords = min_y + compute_x_pixel_coords(n_y, pixel_size)
    
    # 4. Initialize Canvas
    if overlap_mode == 'average':
        stitched_sum = np.zeros((n_y, n_x))
        counts = np.zeros((n_y, n_x))
    else:
        stitched_data = np.full((n_y, n_x), np.nan)

    # Meshgrid for global coordinates (used for interpolation query if needed, 
    # but RegularGridInterpolator works on points)

    # 5. Populate Canvas
    for i, img in enumerate(images):
        # Prefer flat height, fallback to raw height
        data = img.get_flat_height_retrace()
        if data is None:
            data = img.get_height_retrace()
        
        if data is None:
            print(f"Warning: No height data for image {img.bname}")
            continue

        _stitch_single_channel(img, data, image_bounds[i], x_coords, y_coords, n_x, n_y, overlap_mode, stitched_sum if overlap_mode == 'average' else stitched_data, counts if overlap_mode == 'average' else None)

    # 6. Finalize
    if overlap_mode == 'average':
        with np.errstate(invalid='ignore'):
            stitched_data = stitched_sum / counts
        stitched_data[counts == 0] = np.nan
        
    extent = (min_x, max_x, min_y, max_y)
    
    return stitched_data, x_coords, y_coords, extent

def _stitch_single_channel(img, data, img_bounds, x_coords, y_coords, n_x, n_y, overlap_mode, target_buffer, count_buffer=None):
    """Helper to stitch a single channel's data into the target buffer."""
    img_ny, img_nx = data.shape
    img_pixel_size = img.get_pixel_size()
    
    # Construct local axes for the image
    img_x_local = compute_x_pixel_coords(img_nx, img_pixel_size)
    
    # y axis: index 0 corresponds to max y (top of image), index max to min y (bottom)
    img_y_local = compute_y_pixel_coords(img_ny, img_pixel_size)
    
    # RegularGridInterpolator requires strictly increasing coordinates
    if img_y_local[0] > img_y_local[-1]:
        img_y_local = img_y_local[::-1]
        data = data[::-1, :] # Flip rows to match increasing y
        
    # Add absolute offsets to match global coordinate system
    origin_x, origin_y = mrp.offset_image_origin_to_absolute_piezo_position(img)
    
    img_x_abs = img_x_local + origin_x
    img_y_abs = img_y_local + origin_y
    
    interpolator = RegularGridInterpolator((img_y_abs, img_x_abs), data, bounds_error=False, fill_value=np.nan)
    
    # Find overlap with global grid
    x_idx_start = np.searchsorted(x_coords, img_x_abs[0])
    x_idx_end = np.searchsorted(x_coords, img_x_abs[-1])
    y_idx_start = np.searchsorted(y_coords, img_y_abs[0])
    y_idx_end = np.searchsorted(y_coords, img_y_abs[-1])
    
    # Clip to valid range
    x_idx_start = max(0, x_idx_start)
    x_idx_end = min(n_x, x_idx_end + 1)
    y_idx_start = max(0, y_idx_start)
    y_idx_end = min(n_y, y_idx_end + 1)
    
    if x_idx_end <= x_idx_start or y_idx_end <= y_idx_start:
        return
        
    # Create meshgrid for the target slice
    target_x = x_coords[x_idx_start:x_idx_end]
    target_y = y_coords[y_idx_start:y_idx_end]
    target_Y, target_X = np.meshgrid(target_y, target_x, indexing='ij')
    
    # Interpolate
    pts = np.array([target_Y.ravel(), target_X.ravel()]).T
    interpolated_patch = interpolator(pts).reshape(target_Y.shape)
    
    # Mask NaNs
    valid_mask = ~np.isnan(interpolated_patch)
    
    if overlap_mode == 'average':
        current_sum_slice = target_buffer[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
        current_count_slice = count_buffer[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
        
        current_sum_slice[valid_mask] += interpolated_patch[valid_mask]
        current_count_slice[valid_mask] += 1
        
    elif overlap_mode == 'latest':
        current_data_slice = target_buffer[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
        current_data_slice[valid_mask] = interpolated_patch[valid_mask]
        
    elif overlap_mode == 'earliest':
        current_data_slice = target_buffer[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
        # Write only if currently NaN AND patch is valid
        write_mask = np.isnan(current_data_slice) & valid_mask
        current_data_slice[write_mask] = interpolated_patch[write_mask]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import AFMImageCollection
    import path_loader as pl

    # Load images
    print("Loading images...")
    collection = AFMImageCollection.AFMImageCollection(pl.afm_images_path, pl.depressurized_datetime)
    filtered_collection = collection.filter_images(image_range='0002-0008')
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
        stitched_data, x_coords, y_coords, extent = stitch_maps(images, overlap_mode=mode)
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