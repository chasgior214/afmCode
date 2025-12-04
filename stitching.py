import numpy as np
from scipy.interpolate import RegularGridInterpolator
import membrane_relative_positions as mrp
import AFMImage
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
    
    x_coords = np.linspace(min_x, max_x, n_x)
    y_coords = np.linspace(min_y, max_y, n_y)
    
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
    img_x_local = np.linspace(0, img.get_scan_size(), img_nx)
    
    # y axis: index 0 corresponds to max y (top of image), index max to min y (bottom)
    img_y_local = np.array([(img_ny - (idx + 0.5)) * img_pixel_size for idx in range(img_ny)])
    
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