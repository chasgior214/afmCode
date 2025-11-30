import numpy as np


def fit_paraboloid(z_data, center_x_idx, center_y_idx, diameter_um, pixel_size, scan_size):
    """
    Fit a paraboloid to a circular region of the data map.
    
    Args:
        z_data (np.ndarray): 2D array of z values (e.g. height, z-retrace).
        center_x_idx (int): X index of the center of the fit region.
        center_y_idx (int): Y index of the center of the fit region.
        diameter_um (float): Diameter of the fit region in microns.
        pixel_size (float): Size of one pixel in microns.
        scan_size (float): Total physical size of the scan in microns (used for coordinate generation).
        
    Returns:
        dict: Dictionary containing vertex coordinates ('vertex_x_um', 'vertex_y_um', 'vertex_z_nm'),
              'r2' of the fit, and region info ('center_x_um', 'center_y_um', 'radius_um').
              Returns None if fitting fails.
    """
    if center_x_idx is None or center_y_idx is None:
        return None
    if diameter_um is None or diameter_um <= 0:
        return None
    
    y_pixel_count, x_pixel_count = z_data.shape
    
    # Helper to convert index to y-center (matching visualizations.py logic)
    # Note: This assumes y-axis is inverted relative to array index (standard image coordinates)
    def _index_to_y_center(idx):
        idx = int(np.clip(idx, 0, y_pixel_count - 1))
        return (y_pixel_count - (idx + 0.5)) * pixel_size

    half_um = diameter_um / 2.0
    if pixel_size <= 0:
        return None
    half_px = max(1, int(round(half_um / pixel_size)))
    
    x_start = max(0, center_x_idx - half_px)
    x_end = min(x_pixel_count, center_x_idx + half_px + 1)
    y_start = max(0, center_y_idx - half_px)
    y_end = min(y_pixel_count, center_y_idx + half_px + 1)
    
    if x_end <= x_start or y_end <= y_start:
        return None

    sub_heights = z_data[y_start:y_end, x_start:x_end]

    # Generate coordinates for the subregion
    x_coords = np.linspace(0, scan_size, x_pixel_count)
    xs = x_coords[x_start:x_end]
    ys_idx = np.arange(y_start, y_end)
    ys = np.array([_index_to_y_center(idx) for idx in ys_idx])
    
    X_grid, Y_grid = np.meshgrid(xs, ys)
    
    center_x_um = x_coords[center_x_idx]
    center_y_um = _index_to_y_center(center_y_idx)
    
    distance_sq = (X_grid - center_x_um) ** 2 + (Y_grid - center_y_um) ** 2
    circle_mask = distance_sq <= (half_um ** 2)
    
    if not circle_mask.any():
        return None
        
    Z = sub_heights[circle_mask]
    if not np.isfinite(Z).any():
        return None
        
    X_flat = X_grid[circle_mask]
    Y_flat = Y_grid[circle_mask]
    Z_flat = Z.flatten()
    
    finite = np.isfinite(Z_flat)
    if not finite.any():
        return None
        
    X_flat = X_flat[finite]
    Y_flat = Y_flat[finite]
    Z_flat = Z_flat[finite]

    # Center data for stability
    xc = np.mean(X_flat)
    yc = np.mean(Y_flat)
    Xc = X_flat - xc
    Yc = Y_flat - yc

    # Form design matrix for second order polynomial: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    G = np.column_stack(
        [
            Xc ** 2,
            Yc ** 2,
            Xc * Yc,
            Xc,
            Yc,
            np.ones_like(Xc),
        ]
    )
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(G, Z_flat, rcond=None)
    except np.linalg.LinAlgError:
        return None
        
    a, b, c, d, e, f_const = coeffs
    
    # Find vertex
    hessian = np.array([[2 * a, c], [c, 2 * b]])
    rhs = np.array([-d, -e])
    
    if np.linalg.det(hessian) == 0:
        return None
        
    try:
        vx_c, vy_c = np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError:
        return None
        
    vx = vx_c + xc
    vy = vy_c + yc
    vz = (
        a * vx_c ** 2
        + b * vy_c ** 2
        + c * vx_c * vy_c
        + d * vx_c
        + e * vy_c
        + f_const
    )

    # Calculate R^2
    pred = G @ coeffs
    z_mean = np.mean(Z_flat)
    ss_tot = np.sum((Z_flat - z_mean) ** 2)
    ss_res = np.sum((Z_flat - pred) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {
        'vertex_x_um': float(vx),
        'vertex_y_um': float(vy),
        'vertex_z_nm': float(vz),
        'r2': float(r2),
        'center_x_um': float(center_x_um),
        'center_y_um': float(center_y_um),
        'radius_um': float(half_um),
    }

def compute_extremum_square_info(z_data, center_x_idx, center_y_idx, pixel_size, scan_size):
    """
    Calculate statistics for a 4um square region around a selected point.
    
    Args:
        z_data (np.ndarray): 2D array of z values.
        center_x_idx (int): X index of the center point.
        center_y_idx (int): Y index of the center point.
        pixel_size (float): Size of one pixel in microns.
        scan_size (float): Total physical size of the scan in microns.
        
    Returns:
        dict: Dictionary containing 'delta_nm', 'x_um', 'y_um', 'x_idx', 'y_idx'.
              Returns None if calculation fails.
    """
    if center_x_idx is None or center_y_idx is None:
        return None
        
    y_pixel_count, x_pixel_count = z_data.shape
    
    def _index_to_y_center(idx):
        idx = int(np.clip(idx, 0, y_pixel_count - 1))
        return (y_pixel_count - (idx + 0.5)) * pixel_size

    # Determine pixel span corresponding to a 4 μm square (±2 μm from center).
    half_span_um = 2.0
    if pixel_size == 0:
        return None
    half_span_px = max(0, int(round(half_span_um / pixel_size)))
    
    x_start = max(0, center_x_idx - half_span_px)
    x_end = min(x_pixel_count, center_x_idx + half_span_px + 1)
    y_start = max(0, center_y_idx - half_span_px)
    y_end = min(y_pixel_count, center_y_idx + half_span_px + 1)

    if x_end <= x_start or y_end <= y_start:
        return None

    subregion = z_data[y_start:y_end, x_start:x_end]
    if subregion.size == 0:
        return None
    finite_sub = np.isfinite(subregion)
    if not finite_sub.any():
        return None

    # Locate the maximum height within the subregion.
    sub_max_index = np.nanargmax(subregion)
    sub_max_height = float(np.nanmax(subregion))
    local_y, local_x = np.unravel_index(sub_max_index, subregion.shape)
    max_y_idx = y_start + local_y
    max_x_idx = x_start + local_x
    
    x_coords = np.linspace(0, scan_size, x_pixel_count)

    # Compute the mode of the cross section at the y value of the max point.
    row_data = z_data[max_y_idx, :]
    mode_height = calculate_substrate_height(row_data)
    if mode_height is None:
        return None

    delta_nm = float(sub_max_height - mode_height)
    x_coord_um = float(x_coords[max_x_idx])
    y_coord_um = _index_to_y_center(max_y_idx)

    return {
        'delta_nm': delta_nm,
        'x_um': x_coord_um,
        'y_um': y_coord_um,
        'x_idx': int(max_x_idx),
        'y_idx': int(max_y_idx),
    }

def iterative_paraboloid_fit(
    height_map,
    x_coords,
    start_x_um,
    start_y_um,
    pixel_size,
    scan_size,
    paraboloid_window_um=1,
    max_iterations=10,
    convergence_tol=1e-5
):
    """
    Iteratively fit a paraboloid to the data around a given point.
    Updates the center to the fitted vertex and repeats until convergence
    or a maximum number of iterations is reached.

    Args:
        height_map (np.ndarray): 2D array of height values.
        x_coords (np.ndarray): 1D array of x coordinates in microns.
        start_x_um (float): Initial x coordinate of the center in microns.
        start_y_um (float): Initial y coordinate of the center in microns.
        paraboloid_window_um (float): Diameter of the fit region in microns.
        pixel_size (float): Size of one pixel in microns.
        scan_size (float): Total physical size of the scan in microns.
        max_iterations (int): Maximum number of iterations.
        convergence_tol (float): Tolerance for convergence check.

    Returns:
        dict: Dictionary containing the best fit result with keys:
              'vx', 'vy', 'vz', 'r2', 'fit_result'.
              Returns None if fitting fails or no history is generated.
    """
    y_pixel_count = height_map.shape[0]

    def _y_to_index(y_val):
        # Convert y-value (microns) to row index, matching visualizations.py logic
        # _index_to_y_center(idx) = (y_pixel_count - (idx + 0.5)) * pixel_size
        idx = int(np.clip(round(y_pixel_count - 0.5 - y_val / pixel_size), 0, y_pixel_count - 1))
        return idx

    current_x = start_x_um
    current_y = start_y_um
    
    history = []
    converged = False

    for _ in range(max_iterations):
        # Find indices for the current center coordinates
        c_x_idx = int(np.argmin(np.abs(x_coords - current_x)))
        try:
            c_y_idx = _y_to_index(current_y)
        except Exception:
            break

        # Fit paraboloid
        fit_result = fit_paraboloid(
            height_map, c_x_idx, c_y_idx, paraboloid_window_um, pixel_size, scan_size
        )

        if fit_result is None:
            break

        vx = fit_result['vertex_x_um']
        vy = fit_result['vertex_y_um']
        vz = fit_result['vertex_z_nm']
        r2 = fit_result['r2']

        history.append({
            'vx': vx, 'vy': vy, 'vz': vz, 'r2': r2,
            'fit_result': fit_result
        })

        # Check for convergence
        if np.isclose(vx, current_x, atol=convergence_tol) and np.isclose(vy, current_y, atol=convergence_tol):
            converged = True
            break

        # Update current center for next iteration
        current_x = vx
        current_y = vy

    if not history:
        return None

    # Decide which result to keep
    if converged:
        best_result = history[-1]
    else:
        # If not converged, pick the one with highest R^2
        best_result = max(history, key=lambda item: item['r2'])
        
    return best_result


def calculate_substrate_height(row_data, bin_size=0.5):
    """
    Calculate the substrate height from a row of data using the mode.
    
    Args:
        row_data (np.ndarray): 1D array of height values.
        bin_size (float): Size of the bin for histogramming. Units same as row_data.
        
    Returns:
        float: The calculated substrate height, or None if calculation fails.
    """
    finite_row_indices = np.where(np.isfinite(row_data))[0]
    if finite_row_indices.size == 0:
        return None
    row_values = row_data[finite_row_indices]
    row_min = row_values.min()
    row_max = row_values.max()
    if not np.isfinite(row_min) or not np.isfinite(row_max):
        return None
    bins = np.arange(row_min, row_max + bin_size, bin_size)
    if bins.size < 2:
        return None
    hist, edges = np.histogram(row_values, bins=bins)
    if hist.size == 0:
        return None
    mode_idx = int(np.argmax(hist))
    mode_center = (edges[mode_idx] + edges[mode_idx + 1]) / 2
    nearest_idx = finite_row_indices[np.argmin(np.abs(row_values - mode_center))]
    return float(row_data[nearest_idx])
