import numpy as np
import matplotlib.pyplot as plt
import AFMImage
import AFMImageCollection
import surface_analysis as sa
import path_loader as pl
import visualizations as vis
import csv
import os
from datetime import datetime
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from matplotlib.colors import is_color_like

# Image Filtering Configuration
# Set any of these to filter which images are processed:
filter_start_datetime = None  # datetime object, e.g., datetime(2025, 12, 5, 17, 0, 0)
filter_end_datetime = None    # datetime object
filter_image_range = None     # String "AAAA-BBBB" e.g., "0001-0050"

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are, gets the deflections autonomously, and logs the data

# Functions for relating absolute piezo positions to map coordinates
def offset_image_origin_to_absolute_piezo_position(image: AFMImage.AFMImage):
    """Get the offset to translate the image origin (the bottom left corner of the image) to absolute piezo position."""
    x_size, y_size = image.get_x_y_size()
    slow_scan_size = image.get_SlowScanSize()
    scan_direction = image.get_scan_direction()
    image_origin_x_offset_from_image_centre = -0.5 * x_size
    if scan_direction:  # scan down
        image_origin_y_offset_from_image_centre = 0.5 * slow_scan_size - y_size
    else: # scan up
        image_origin_y_offset_from_image_centre = -0.5 * slow_scan_size
    # adjust for image offsets
    image_origin_absolute_x = image.get_x_offset() + image_origin_x_offset_from_image_centre
    image_origin_absolute_y = image.get_y_offset() + image_origin_y_offset_from_image_centre
    return (image_origin_absolute_x, image_origin_absolute_y)

def image_bounds_absolute_positions(image: AFMImage.AFMImage):
    """Get the bounds of the image in absolute piezo positions.
    Returns (x_min, y_min, x_max, y_max)"""
    image_origin_absolute_x, image_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(image)
    x_size, y_size = image.get_x_y_size()
    return (image_origin_absolute_x, image_origin_absolute_y, image_origin_absolute_x + x_size, image_origin_absolute_y + y_size)

# TODO update x_spacing, y_spacing based on an average over a big image
x_spacing = 7.63
y_spacing = 4.6
class MembraneNavigator:
    def __init__(self, well_diameter=4, time_between_images_for_safe_minimal_drift=15):
        self.well_diameter = well_diameter
        self.well_radius = well_diameter / 2
        self.time_between_images_for_safe_minimal_drift = time_between_images_for_safe_minimal_drift

    def well_positions_grid(self, x_coords, y_coords):
        """Generate a grid of well positions. My dies are made such that if position (0,0) is on a well, other wells are at any combination of movements away from that, where a movement is either (+x_spacing, +y_spacing), (+2*x_spacing, 0), or (0, +2*y_spacing)."""
        positions = []
        for i in range(x_coords):
            for j in range(y_coords):
                if (i + j) % 2 == 0:
                    positions.append((i * x_spacing, j * y_spacing))
        return positions

    def predict_position_from_change_in_coordinates(self, pos, pos_coords, final_coords):
        """Predict the position of a well based on the position of another well and each well's coordinates."""
        x_coords_change = final_coords[0] - pos_coords[0]
        y_coords_change = final_coords[1] - pos_coords[1]
        predicted_x_pos = pos[0] + x_coords_change * x_spacing
        predicted_y_pos = pos[1] + y_coords_change * y_spacing
        return (predicted_x_pos, predicted_y_pos)

    def track_wells(self, image_collection, initial_well_name, initial_well_coords, well_map, initial_well_absolute_pos=None, initial_well_fixed_z=None, edge_tolerance=0, each_found_well_updates_all_well_positions=False):
        """
        Track wells across an image collection.
        
        Args:
            image_collection: AFMImageCollection object or list of AFMImage objects.
            initial_well_name: Name of the well identified in the first image.
            initial_well_coords: map coordinates of the initial well (x, y).
            well_map: Dictionary mapping well names to their map coordinates (x, y).
            initial_well_absolute_pos: Optional (x, y) absolute position of the initial well. 
                                       If None, it will be determined from the first image (requires user interaction or pre-selection).
            edge_tolerance: Optional tolerance distance (in μm) to allow fitting wells just outside image bounds. 
                          If a well's predicted center is within this distance from the image edge, 
                          fitting will be attempted using the closest point within the image. Default is 0 (disabled).
        
        Returns:
            List of dictionaries containing:
            - 'Well': well name
            - 'Time (minutes)': time since depressurization
            - 'Deflection (nm)': calculated deflection
            - 'Point 1 X Pixel': x pixel position of point 1
            - 'Point 1 Y Pixel': y pixel position of point 1
            - 'Point 1 X um': x position of point 1 in um
            - 'Point 1 Y um': y position of point 1 in um
            - 'Point 1 Z nm': z position of point 1 in nm
            - 'Point 2 X Pixel': x pixel position of point 2
            - 'Point 2 Y Pixel': y pixel position of point 2
            - 'Point 2 X um': x position of point 2 in um
            - 'Point 2 Y um': y position of point 2 in um
            - 'Point 2 Z nm': z position of point 2 in nm
        """
        
        well_positions = {name: None for name in well_map}
        
        # If we have an initial position, set it and predict others
        if initial_well_absolute_pos:
            well_positions[initial_well_name] = initial_well_absolute_pos
            for well in well_positions:
                if well != initial_well_name:
                    well_positions[well] = self.predict_position_from_change_in_coordinates(
                        initial_well_absolute_pos, initial_well_coords, well_map[well]
                    )
        
        results = []
        images = image_collection.images if hasattr(image_collection, 'images') else image_collection

        for image in images:
            print(f"Processing image {image.bname}")
            absolute_image_bounds = image_bounds_absolute_positions(image)
            pixel_size = image.get_pixel_size()
            scan_size = image.get_scan_size()
            x_pixel_count, y_pixel_count = image.get_x_y_pixel_counts()
            x_coords = np.linspace(0, scan_size, x_pixel_count)
            print(f"Image bounds: {absolute_image_bounds}")

            # Determine height map once per image
            height_map = image.get_flat_height_retrace()
            if height_map is None:
                height_map = image.get_height_retrace()

            scan_direction = image.get_scan_direction()

            # Determine which wells are expected to appear in this image
            def well_expected_in_image(well_name):
                if well_positions[well_name] is None:
                    return None

                well_x, well_y = well_positions[well_name]
                x_min, y_min, x_max, y_max = absolute_image_bounds

                dist_to_left = well_x - x_min
                dist_to_right = x_max - well_x
                dist_to_bottom = well_y - y_min
                dist_to_top = y_max - well_y

                inside_bounds = (x_min < well_x < x_max and y_min < well_y < y_max)
                within_tolerance = (
                    dist_to_left >= -edge_tolerance and dist_to_right >= -edge_tolerance and
                    dist_to_bottom >= -edge_tolerance and dist_to_top >= -edge_tolerance
                )

                if inside_bounds or (edge_tolerance > 0 and within_tolerance):
                    clamped_x = np.clip(well_x, x_min, x_max)
                    clamped_y = np.clip(well_y, y_min, y_max)
                    return {
                        'name': well_name,
                        'position': (well_x, well_y),
                        'clamped': (clamped_x, clamped_y),
                        'inside_bounds': inside_bounds,
                    }

                return None

            remaining_wells = {
                w['name'] for w in filter(None, (well_expected_in_image(name) for name in well_positions))
            }
            expect_multiple_in_image = len(remaining_wells) > 1

            def ordered_expected_wells():
                expected = []
                missing = []
                for well in remaining_wells:
                    expectation = well_expected_in_image(well)
                    if expectation:
                        expected.append(expectation)
                    else:
                        missing.append(well)

                # Drop any wells no longer expected (e.g., moved out of bounds after an update)
                for well in missing:
                    remaining_wells.discard(well)

                reverse = bool(scan_direction)  # scan down -> earlier lines are higher y values
                expected.sort(key=lambda item: item['position'][1], reverse=reverse)
                return expected

            while remaining_wells:
                expected_wells = ordered_expected_wells()
                if not expected_wells:
                    break

                current = expected_wells[0]
                well = current['name']
                well_x, well_y = current['position']

                print(f"\tChecking {well} with estimated position {current['position']}")

                if current['inside_bounds']:
                    print(f"Found {well} in image {image.bname}")
                else:
                    print(f"Found {well} near edge of image {image.bname} (within {edge_tolerance} μm tolerance), using clamped position")

                clamped_x, clamped_y = current['clamped']

                # Calculate relative position for fitting (using clamped position)
                rel_x = clamped_x - absolute_image_bounds[0]
                rel_y = clamped_y - absolute_image_bounds[1]

                # Check if should use fixed position for the initial well in the first image
                if image == images[0] and well == initial_well_name and initial_well_fixed_z is not None:
                    print(f"Using user-supplied position for {well} in {image.bname}")
                    fit_result = {'vx': rel_x, 'vy': rel_y, 'vz': initial_well_fixed_z}
                else:
                    fit_result = sa.iterative_paraboloid_fit(
                    height_map,
                    x_coords,
                    rel_x,
                    rel_y,
                    pixel_size,
                    scan_size
                )

                if fit_result:
                    vertex_x_um = fit_result['vx']
                    vertex_y_um = fit_result['vy']
                    vertex_z_nm = fit_result['vz']

                    # Calculate Substrate Height
                    y_idx = int(np.clip(round(y_pixel_count - 0.5 - vertex_y_um / pixel_size), 0, y_pixel_count - 1))
                    row_data = height_map[y_idx, :]
                    substrate_z_nm = sa.calculate_substrate_height(row_data)

                    if substrate_z_nm is not None:
                        deflection = vertex_z_nm - substrate_z_nm

                        # Calculate acquisition time for the well's line
                        well_time = image.get_line_acquisition_datetime(y_idx)

                        # Point 1 (Substrate) calculations
                        # Find x index closest to substrate height in the row
                        finite_indices = np.where(np.isfinite(row_data))[0]
                        closest_idx_in_finite = np.argmin(np.abs(row_data[finite_indices] - substrate_z_nm))
                        p1_x_pixel = int(finite_indices[closest_idx_in_finite])

                        p1_x_um = float(x_coords[p1_x_pixel])
                        p1_y_pixel = int(y_idx)
                        p1_y_um = float((y_pixel_count - (y_idx + 0.5)) * pixel_size)
                        p1_z_nm = float(substrate_z_nm)

                        # Point 2 (Extremum) calculations
                        p2_x_um = float(vertex_x_um)
                        p2_y_um = float(vertex_y_um)
                        p2_z_nm = float(vertex_z_nm)
                        p2_x_pixel = int(np.argmin(np.abs(x_coords - vertex_x_um)))
                        p2_y_pixel = int(np.clip(round(y_pixel_count - 0.5 - vertex_y_um / pixel_size), 0, y_pixel_count - 1))

                        result_entry = {
                            'Well': well,
                            'Image Name': image.bname,
                            'Time (minutes)': (well_time - pl.depressurized_datetime).total_seconds() / 60.0,
                            'Deflection (nm)': float(deflection),
                            'Point 1 X Pixel': p1_x_pixel,
                            'Point 1 Y Pixel': p1_y_pixel,
                            'Point 1 X (um)': p1_x_um,
                            'Point 1 Y (um)': p1_y_um,
                            'Point 1 Z (nm)': p1_z_nm,
                            'Point 2 X Pixel': p2_x_pixel,
                            'Point 2 Y Pixel': p2_y_pixel,
                            'Point 2 X (um)': p2_x_um,
                            'Point 2 Y (um)': p2_y_um,
                            'Point 2 Z (nm)': p2_z_nm,
                        }

                        results.append(result_entry)

                        # Update position to account for drift
                        absolute_vertex_x = vertex_x_um + absolute_image_bounds[0]
                        absolute_vertex_y = vertex_y_um + absolute_image_bounds[1]
                        well_positions[well] = (absolute_vertex_x, absolute_vertex_y)

                        if each_found_well_updates_all_well_positions or expect_multiple_in_image:
                            # Update predictions for remaining wells based on the newly found position
                            for other_well in well_map:
                                if other_well != well:
                                    well_positions[other_well] = self.predict_position_from_change_in_coordinates(
                                        well_positions[well], well_map[well], well_map[other_well]
                                    )
                        else:
                            print(f"Failed to calculate substrate height for {well}")
                    else:
                        print(f"Fit failed for {well}")

                remaining_wells.discard(well)

        return results

# sample37 Configuration
sample37_well_map = {
    'orange': (0, 0),
    'blue': (0, 2),
    'green': (1, 1),
    'red': (1, 3),
    'black': (2, 0)
}


# sample53 o(5,1) wells
sample53_o_5_1_well_coords = [
    (1,5), (1,7), (1,9),
    (2,6), (2,8), (2,10),
    (3,5), (3,7), (3,9), (3,11),
    (4,4), (4,6), (4,8), (4,10), (4,12),
    (5,3), (5,5), (5,7), (5,9), (5,11),
    (6,2), (6,4), (6,6), (6,8), (6,10), (6,12),
    (7,3), (7,5), (7,7), (7,9), (7,11),
    (8,6), (8,8), (8,10),
    (9,7), (9,9),
    (10,6), (10,8), (10,10), (10,14),
    (11,9), (11,13), (11,15),
    (12,8), (12,10), (12,12), (12,14), (12,16),
    (13,9), (13,11), (13,13), (13,15),
    (14,12), (14,14)
]

# make the well map from the list of well coordinates, using the coordinates as both the well names and the well positions
sample53_o_5_1_well_map = {
    str((x_idx, y_idx)): (x_idx, y_idx) for (x_idx, y_idx) in sample53_o_5_1_well_coords
}

class WellPositionsReviewer:
    def __init__(self, navigator, image_collection, results, well_map):
        self.navigator = navigator
        self.image_collection = image_collection
        self.results = results
        self.well_map = well_map
        self.image_map = {img.bname: img for img in self.image_collection.images}
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.secax_map_x = None
        self.secax_map_y = None
        self.slider = None
        self.scatter_map_ax1 = {}
        self.scatter_map_ax2 = {}
        self.abs_points = []
        self.times = []
        self.min_time = 0
        self.max_time = 1
        self.initial_xlim = (0, 1)
        self.initial_ylim = (0, 1)
        self.well_colors = {}
        self.well_markers = {}
        self.future_span = None

    def _assign_well_colors(self):
        unique_wells = sorted(list(set(entry['Well'] for entry in self.results)))
        palette = plt.cm.tab10.colors  # Use a standard palette
        palette_idx = 0
        markers = ['o', '^', 's', 'D']
        self.well_colors = {}
        self.well_markers = {}
        
        for well in unique_wells:
            if is_color_like(well):
                self.well_colors[well] = well
                self.well_markers[well] = 'o'
            else:
                # Assign a color from the palette
                self.well_colors[well] = palette[palette_idx % len(palette)]
                self.well_markers[well] = markers[(palette_idx // 10) % 4]
                palette_idx += 1

    def plot(self):
        self.refresh_data()
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 2]})
        # Pull the top plot a bit closer to the top of the window while leaving extra room between the plots
        plt.subplots_adjust(bottom=0.08, top=0.975, hspace=0.42)
        
        self.draw_deflection_plot()
        self.draw_absolute_positions(self.max_time)
        self.setup_slider_and_buttons()
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Maximize window if possible
        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()
        except Exception:
            try:
                manager.window.state('zoomed')
            except Exception:
                pass
        plt.show()

    def refresh_data(self):
        # Precompute absolute coordinates
        self._assign_well_colors()
        self.abs_points = []
        for entry in self.results:
            img = self.image_map.get(entry['Image Name'])
            if img is None:
                continue
            ox, oy = offset_image_origin_to_absolute_piezo_position(img)
            abs_x = ox + entry['Point 2 X (um)']
            abs_y = oy + entry['Point 2 Y (um)']
            # Store full entry for retrieval
            self.abs_points.append({
                'time': entry['Time (minutes)'],
                'x': abs_x,
                'y': abs_y,
                'well': entry['Well'],
                'deflection': entry['Deflection (nm)'],
                'entry': entry
            })
        
        if not self.abs_points:
            self.min_time, self.max_time = 0, 1
            return

        self.abs_points.sort(key=lambda p: p['time'])
        self.times = [p['time'] for p in self.abs_points]
        self.min_time, self.max_time = min(self.times), max(self.times)
        
        xs = [p['x'] for p in self.abs_points]
        ys = [p['y'] for p in self.abs_points]
        margin_x = max(1.0, 0.05 * (max(xs) - min(xs)))
        margin_y = max(1.0, 0.05 * (max(ys) - min(ys)))
        self.initial_xlim = (min(xs) - margin_x, max(xs) + margin_x)
        self.initial_ylim = (min(ys) - margin_y, max(ys) + margin_y)

    def draw_deflection_plot(self):
        # Clear any existing future shading before wiping the axes to avoid
        # stale artists that cannot be removed afterwards.
        if self.future_span and getattr(self.future_span, 'axes', None) is not None:
            try:
                self.future_span.remove()
            except NotImplementedError:
                pass
        self.future_span = None

        self.ax1.clear()
        self.scatter_map_ax1 = {}

        for well_name in set(entry['Well'] for entry in self.results):
            well_data = [entry for entry in self.results if entry['Well'] == well_name]
            t = [entry['Time (minutes)'] for entry in well_data]
            d = [entry['Deflection (nm)'] for entry in well_data]
            color = self.well_colors.get(well_name, 'black')
            marker = self.well_markers.get(well_name, 'o')
            sc = self.ax1.scatter(t, d, label=well_name, picker=5, edgecolors=color, facecolors='none', marker=marker)
            self.scatter_map_ax1[sc] = well_data
            
        self.ax1.set_xlabel('Time (min)')
        self.ax1.set_ylabel('Deflection (nm)')
        if not all(is_color_like(well) for well in self.well_colors):
            self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        # Re-apply any future shading if a slider already exists
        if self.slider is not None:
            self.update_deflection_future_background(self.slider.val)
        else:
            self.future_span = None

    def draw_absolute_positions(self, time_cut):
        self.ax2.clear()
        if self.secax_map_x:
            self.secax_map_x.remove()
            self.secax_map_x = None
        if self.secax_map_y:
            self.secax_map_y.remove()
            self.secax_map_y = None
        self.scatter_map_ax2 = {}
        
        # Group points by well
        points_by_well = {}
        for p in self.abs_points:
            if p['time'] <= time_cut:
                if p['well'] not in points_by_well:
                    points_by_well[p['well']] = []
                points_by_well[p['well']].append(p)
        
        drawn_labels = set()
        # Determine last displayed time across points that are within `time_cut`.
        # This ensures star marks most recent point shown by the slider.
        displayed_last_time = None
        if points_by_well:
            displayed_last_time = max(p['time'] for pts in points_by_well.values() for p in pts)
        for well, points in points_by_well.items():
            points.sort(key=lambda p: p['time'])
            
            # Previous points
            if len(points) > 1:
                prev_points = points[:-1]
                xs = [p['x'] for p in prev_points]
                ys = [p['y'] for p in prev_points]
                color = self.well_colors.get(well, 'black')
                marker = self.well_markers.get(well, 'o')
                sc = self.ax2.scatter(xs, ys, edgecolors=color, facecolors='none', s=100, alpha=0.6, marker=marker, picker=5)
                self.scatter_map_ax2[sc] = prev_points
            
            # Last point
            last_point = points[-1]
            label = well if well not in drawn_labels else None
            color = self.well_colors.get(well, 'black')
            # If last point is most recent among displayed points, mark it with a star
            if displayed_last_time is not None and last_point['time'] == displayed_last_time:
                sc = self.ax2.scatter([last_point['x']], [last_point['y']], color=color, s=250, alpha=1.0, marker='*', linewidths=2, label=label, picker=5)
            else:
                sc = self.ax2.scatter([last_point['x']], [last_point['y']], color=color, s=150, alpha=1.0, marker='x', linewidths=3, label=label, picker=5)
            self.scatter_map_ax2[sc] = [last_point]
            if label:
                drawn_labels.add(label)

        self.ax2.set_xlabel('Absolute X Position (μm)')
        self.ax2.set_ylabel('Absolute Y Position (μm)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax2.set_xlim(self.initial_xlim)
        self.ax2.set_ylim(self.initial_ylim)
        # Move the absolute position axes to the top and right
        self.ax2.xaxis.set_label_position('top')
        self.ax2.xaxis.tick_top()
        self.ax2.tick_params(axis='x', labelbottom=False)
        self.ax2.yaxis.set_label_position('right')
        self.ax2.yaxis.tick_right()
        self.ax2.tick_params(axis='y', labelleft=False)

        # Add bottom/left axes showing well map coordinates anchored on the last displayed point
        anchor_point = self.get_last_displayed_point(time_cut)
        if anchor_point and anchor_point['well'] in self.well_map:
            anchor_abs_x = anchor_point['x']
            anchor_abs_y = anchor_point['y']
            anchor_map_x, anchor_map_y = self.well_map[anchor_point['well']]

            def abs_to_map_x(x):
                return (x - anchor_abs_x) / x_spacing + anchor_map_x

            def map_to_abs_x(mx):
                return (mx - anchor_map_x) * x_spacing + anchor_abs_x

            def abs_to_map_y(y):
                return (y - anchor_abs_y) / y_spacing + anchor_map_y

            def map_to_abs_y(my):
                return (my - anchor_map_y) * y_spacing + anchor_abs_y

            self.secax_map_x = self.ax2.secondary_xaxis('bottom', functions=(abs_to_map_x, map_to_abs_x))
            self.secax_map_x.set_xlabel(f"Well Map X (anchor: {anchor_point['well']})")
            self.secax_map_x.tick_params(axis='x', labelbottom=True, direction='out')

            self.secax_map_y = self.ax2.secondary_yaxis('left', functions=(abs_to_map_y, map_to_abs_y))
            self.secax_map_y.set_ylabel(f"Well Map Y (anchor: {anchor_point['well']})")
            self.secax_map_y.tick_params(axis='y', labelleft=True, direction='out')
        
        self.fig.canvas.draw_idle()

    def update_deflection_future_background(self, time_cut):
        """Shade the portion of the deflection plot after ``time_cut``."""
        if self.future_span:
            remove_method = getattr(self.future_span, 'remove', None)
            has_axes = getattr(self.future_span, 'axes', None) is not None
            if remove_method and has_axes:
                try:
                    remove_method()
                except (ValueError, NotImplementedError):
                    pass
            self.future_span = None

        if time_cut < self.max_time:
            self.future_span = self.ax1.axvspan(
                time_cut,
                self.max_time,
                facecolor='lightgrey',
                alpha=0.2,
                zorder=0
            )
        self.fig.canvas.draw_idle()

    def get_last_displayed_point(self, time_cut):
        """Return the last displayed absolute point (with well metadata) up to ``time_cut``."""
        pts = [p for p in self.abs_points if p['time'] <= time_cut]
        if not pts:
            return None

        max_time = max(p['time'] for p in pts)
        candidates = [p for p in pts if p['time'] == max_time]
        if not candidates:
            return None
        return candidates[-1]

    def get_last_displayed_entry(self, time_cut):
        """Return the original results entry for the last point with time <= time_cut.

        If no points are displayed for the given cut, returns None.
        """
        # Guard if no points
        if not self.abs_points:
            return None

        # Filter points that are within the displayed time cut
        pts = [p for p in self.abs_points if p['time'] <= time_cut]
        if not pts:
            return None

        # Find the most recent time among displayed points
        max_time = max(p['time'] for p in pts)
        # Choose the last point with that time (preserves ordering)
        candidates = [p for p in pts if p['time'] == max_time]
        if not candidates:
            return None
        return candidates[-1]['entry']

    def setup_slider_and_buttons(self):
        # Slider for time cut: place between the two plots and make its width match ax1
        axcolor = 'lightgoldenrodyellow'
        # Ensure figure layout is drawn so axes positions are accurate
        try:
            self.fig.canvas.draw()
        except Exception:
            pass

        pos1 = self.ax1.get_position()
        pos2 = self.ax2.get_position()

        slider_height = 0.025
        # x and width match ax1 so slider lines up with the deflection plot x-axis
        slider_x = pos1.x0
        slider_width = pos1.x1 - pos1.x0
        # place vertically between ax1 bottom and ax2 top
        slider_y = pos2.y1 + (pos1.y0 - pos2.y1) / 2.0 - slider_height / 2.0

        # Keep slider clear of the top axis label and tick labels on the lower plot
        axis_top = pos2.y1
        try:
            renderer = self.fig.canvas.get_renderer()
            tight_bbox = self.ax2.xaxis.get_tightbbox(renderer=renderer)
            if tight_bbox is not None:
                tight_bbox = tight_bbox.transformed(self.fig.transFigure.inverted())
                axis_top = tight_bbox.y1
        except Exception:
            pass

        padding = 0.015
        min_slider_y = axis_top + padding + slider_height
        max_slider_y = max(pos1.y0 - slider_height - padding, min_slider_y)
        slider_y = min(max(slider_y, min_slider_y), max_slider_y)

        # Clamp slider_y to reasonable figure bounds
        slider_y = max(0.01, min(0.95, slider_y))

        ax_slider = self.fig.add_axes([slider_x, slider_y, slider_width, slider_height], facecolor=axcolor)
        # Remove label text (only keep buttons); also hide the value text
        self.slider = Slider(ax_slider, '', self.min_time, self.max_time,
                             valinit=self.max_time, valstep=max((self.max_time - self.min_time) / 200.0, 0.1))
        try:
            # hide the text that shows the current slider value
            self.slider.valtext.set_visible(False)
        except Exception:
            pass
        def on_slider(val):
            self.draw_absolute_positions(val)
            self.update_deflection_future_background(val)

        self.slider.on_changed(on_slider)

        # Buttons: Prev / Next — position them just outside the slider ends
        button_width = min(0.08, slider_width * 0.08 + 0.02)
        gap = 0.01
        left_x = max(0.0, slider_x - button_width - gap)
        right_x = min(0.99 - button_width, slider_x + slider_width + gap)
        axprev = self.fig.add_axes([left_x, slider_y, button_width, slider_height])
        axnext = self.fig.add_axes([right_x, slider_y, button_width, slider_height])
        bprev = Button(axprev, 'Prev')
        bnext = Button(axnext, 'Next')
        
        # Keep references to prevent garbage collection
        self.bprev = bprev
        self.bnext = bnext

        def show_n(n):
            n = int(max(1, min(len(self.abs_points), n)))
            if n > 0:
                t = self.abs_points[n - 1]['time']
                self.slider.set_val(t)

        def on_prev(event):
            # Find current count based on slider value
            current_time = self.slider.val
            current_n = sum(1 for p in self.abs_points if p['time'] <= current_time)
            show_n(current_n - 1)

        def on_next(event):
            current_time = self.slider.val
            current_n = sum(1 for p in self.abs_points if p['time'] <= current_time)
            show_n(current_n + 1)

        bprev.on_clicked(on_prev)
        bnext.on_clicked(on_next)

        def on_key(event):
            if event.key in ('left', 'arrowleft'):
                on_prev(None)
            elif event.key in ('right', 'arrowright'):
                on_next(None)
            elif event.key in (' ', 'space'):
                # Spacebar: retrack the last displayed point (the most recent point
                # up to the current slider value). This mirrors clicking the star.
                try:
                    entry = self.get_last_displayed_entry(self.slider.val)
                except Exception:
                    entry = None
                if entry:
                    self.handle_retrack(entry)
            elif event.key == 'e':
                current_cut = self.slider.val if self.slider is not None else None
                self.export_deflation_curves(time_cut=current_cut)

        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def on_pick(self, event):
        artist = event.artist
        # Just take the first point if multiple are close
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        ind = event.ind[0] 
        
        selected_entry = None
        
        if artist in self.scatter_map_ax1:
            entries = self.scatter_map_ax1[artist]
            if ind < len(entries):
                selected_entry = entries[ind]
        elif artist in self.scatter_map_ax2:
            points = self.scatter_map_ax2[artist]
            if ind < len(points):
                selected_entry = points[ind]['entry']
        
        if selected_entry:
            self.handle_retrack(selected_entry)

    def export_deflation_curves(self, time_cut=None):
        print("Exporting deflation curves...")
        print(self.results)

        if not pl.editing_mode:
            print("Editing mode is disabled (pl.editing_mode == False); skipping CSV export.")
            return

        results_to_export = (
            self.results if time_cut is None else [
                entry for entry in self.results if entry['Time (minutes)'] <= time_cut
            ]
        )

        if not results_to_export:
            print("No results available within the selected time cut to export.")
            return

        os.makedirs(pl.deflation_curves_path, exist_ok=True)

        header = [
            'Time (minutes)',
            'Deflection (nm)',
            'Point 1 X Pixel',
            'Point 1 Y Pixel',
            'Point 1 X (um)',
            'Point 1 Y (um)',
            'Point 1 Z (nm)',
            'Point 2 X Pixel',
            'Point 2 Y Pixel',
            'Point 2 X (um)',
            'Point 2 Y (um)',
            'Point 2 Z (nm)'
        ]

        wells = sorted(set(entry['Well'] for entry in results_to_export))
        for well in wells:
            well_entries = [entry for entry in results_to_export if entry['Well'] == well]
            if not well_entries:
                continue

            well_entries.sort(key=lambda e: e.get('Time (minutes)', 0))
            
            # Calculate time bounds of new results
            new_min_time = min(e.get('Time (minutes)', 0) for e in well_entries)
            new_max_time = max(e.get('Time (minutes)', 0) for e in well_entries)
            
            file_path = pl.get_deflation_curve_path(
                pl.sample_ID,
                pl.depressurized_date,
                pl.depressurized_time,
                pl.transfer_location,
                well
            )

            # Load existing data if file exists
            existing_entries = []
            if os.path.exists(file_path):
                with open(file_path, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        try:
                            row_time = float(row.get('Time (minutes)', 0))
                            # Keep rows outside the new data's time window
                            if row_time < new_min_time or row_time > new_max_time:
                                existing_entries.append(row)
                        except (ValueError, TypeError):
                            # Skip malformed rows
                            pass
                print(f"Loaded {len(existing_entries)} existing entries outside time window [{new_min_time:.2f}, {new_max_time:.2f}] min")
            
            # Convert new entries to same format as CSV rows
            new_rows = []
            for entry in well_entries:
                new_rows.append({
                    'Time (minutes)': entry.get('Time (minutes)'),
                    'Deflection (nm)': entry.get('Deflection (nm)'),
                    'Point 1 X Pixel': entry.get('Point 1 X Pixel'),
                    'Point 1 Y Pixel': entry.get('Point 1 Y Pixel'),
                    'Point 1 X (um)': entry.get('Point 1 X (um)'),
                    'Point 1 Y (um)': entry.get('Point 1 Y (um)'),
                    'Point 1 Z (nm)': entry.get('Point 1 Z (nm)'),
                    'Point 2 X Pixel': entry.get('Point 2 X Pixel'),
                    'Point 2 Y Pixel': entry.get('Point 2 Y Pixel'),
                    'Point 2 X (um)': entry.get('Point 2 X (um)'),
                    'Point 2 Y (um)': entry.get('Point 2 Y (um)'),
                    'Point 2 Z (nm)': entry.get('Point 2 Z (nm)'),
                })
            
            # Merge and sort
            all_rows = existing_entries + new_rows
            all_rows.sort(key=lambda r: float(r.get('Time (minutes)', 0)))
            
            action = "Merging with" if existing_entries else "Saving"
            print(f"{action} deflation curve for well '{well}' to {file_path}")
            print(f"  Total entries: {len(all_rows)} ({len(new_rows)} new, {len(existing_entries)} preserved)")

            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for row in all_rows:
                    writer.writerow([row.get(h, '') for h in header])

        print("Deflation curve export complete.")


    def handle_retrack(self, entry):
        image_name = entry['Image Name']
        well_name = entry['Well']
        print(f"\nSelected {well_name} in {image_name}")
        
        image = self.image_map.get(image_name)
        if not image:
            return

        # Open select_heights
        print("Opening image for selection... Please select the new extremum.")
        # Prepare initial selection from existing entry
        p1 = (
            entry.get('Point 1 Z (nm)'),
            entry.get('Point 1 X (um)'),
            entry.get('Point 1 Y (um)'),
            entry.get('Point 1 X Pixel'),
            entry.get('Point 1 Y Pixel')
        )
        p2 = (
            entry.get('Point 2 Z (nm)'),
            entry.get('Point 2 X (um)'),
            entry.get('Point 2 Y (um)'),
            entry.get('Point 2 X Pixel'),
            entry.get('Point 2 Y Pixel')
        )
        initial_slots = [p1, p2]
        initial_y = entry.get('Point 2 Y (um)', 0)

        res = vis.select_heights(image, initial_line_height=initial_y, initial_selected_slots=initial_slots)
        print(f"DEBUG: select_heights returned: {res}")
        slots = res.get('selected_slots', [None, None])
        extremum = slots[1]
        
        if extremum:
            new_well_name = input(f"Enter well name (default = {well_name}): ")
            if not new_well_name:
                new_well_name = well_name
            
            # Calculate absolute position
            ox, oy = offset_image_origin_to_absolute_piezo_position(image)
            abs_x = extremum[1] + ox
            abs_y = extremum[2] + oy
            initial_pos = (abs_x, abs_y)
            initial_z = extremum[0]
            
            # Slice images
            all_images = self.image_collection.images
            try:
                start_idx = all_images.index(image)
            except ValueError:
                return
                
            images_to_process = all_images[start_idx:]
            
            # Truncate results: Keep results from images BEFORE the clicked image
            images_to_keep = set(img.bname for img in all_images[:start_idx])
            self.results = [r for r in self.results if r['Image Name'] in images_to_keep]
            
            print(f"Retracking from {image_name} with {new_well_name}...")
            
            new_results = self.navigator.track_wells(
                images_to_process,
                new_well_name,
                self.well_map.get(new_well_name, (0,0)), 
                self.well_map,
                initial_well_absolute_pos=initial_pos,
                initial_well_fixed_z=initial_z,
                edge_tolerance=0.3,
                each_found_well_updates_all_well_positions=True
            )
            
            self.results.extend(new_results)
            
            # Refresh plot
            self.refresh_data()
            self.draw_deflection_plot()
            self.draw_absolute_positions(self.max_time)
            
            # Update slider limits
            self.slider.valmin = self.min_time
            self.slider.valmax = self.max_time
            self.slider.ax.set_xlim(self.min_time, self.max_time)
            self.slider.set_val(self.max_time)
            
            print("Retracking complete. Plot updated.")
        else:
            print("No extremum selected. Retracking cancelled.")


sample_ID_and_location_to_well_map = {
    ('37', '$(6,3)'): sample37_well_map,
    ('53', 'o(5,1)'): sample53_o_5_1_well_map
}

# Script
if __name__ == "__main__":
    navigator = MembraneNavigator()
    sample_ID = pl.sample_ID
    location = pl.transfer_location
    well_map = sample_ID_and_location_to_well_map.get((sample_ID, location), None)

    image_collection = AFMImageCollection.AFMImageCollection(pl.afm_images_path, pl.depressurized_datetime)
    
    # Apply image filters
    original_count = len(image_collection)
    filtered_collection = image_collection.filter_images(
        start_datetime=filter_start_datetime,
        end_datetime=filter_end_datetime,
        image_range=filter_image_range
    )
    
    if not filtered_collection.images:
        print("No images matched the filter criteria.")
        exit(1)
    
    print(f"Processing {len(filtered_collection)} images (filtered from {original_count} total)")
    
    # Show the first filtered image to pick a well
    first_image = filtered_collection.images[0]
    first_image_origin_absolute_x, first_image_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(first_image)
    
    print(f"First image origin absolute position: x={first_image_origin_absolute_x:.3f} μm, y={first_image_origin_absolute_y:.3f} μm")
    
    # User Interaction
    res = vis.select_heights(first_image)
    slots = res.get('selected_slots', [None, None])
    well_clicked_on_extremum = slots[1]
    
    if well_clicked_on_extremum:
        well_clicked_on = input("Enter well name (e.g., green, orange): ")
        if well_clicked_on in well_map:
            well_clicked_on_coords = well_map[well_clicked_on]
            
            well_clicked_on_absolute_x = well_clicked_on_extremum[1] + first_image_origin_absolute_x
            well_clicked_on_absolute_y = well_clicked_on_extremum[2] + first_image_origin_absolute_y
            
            initial_pos = (well_clicked_on_absolute_x, well_clicked_on_absolute_y)
            
            print(f"Tracking wells starting from {well_clicked_on} at {initial_pos}")
            
            results = navigator.track_wells(
                filtered_collection,
                well_clicked_on, 
                well_clicked_on_coords, 
                well_map, 
                initial_well_absolute_pos=initial_pos,
                edge_tolerance=0.3,
                each_found_well_updates_all_well_positions=True
            )
            
            plotter = WellPositionsReviewer(navigator, filtered_collection, results, well_map)
            plotter.plot()
            
        else:
            print(f"Well '{well_clicked_on}' not found in coordinates map.")
    else:
        print("No well selected.")

    # plot the sample53 well coordinates as blue 4um diameter circles
    # for (x_idx, y_idx) in sample53_o_5_1_well_coords:
    #     center_x = x_idx * x_spacing
    #     center_y = y_idx * y_spacing
    #     circle = plt.Circle((center_x, center_y), 2, color='blue', fill=True, linewidth=2)
    #     plt.gca().add_artist(circle)
    # plt.xlim(0, x_spacing*15)
    # plt.ylim(0, y_spacing*17)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel('X Position (um)')
    # plt.ylabel('Y Position (um)')
    # plt.title('Well Positions for Sample 53')
    # plt.grid()
    # plt.show()


    # plot a 4um diameter circle at each position in sample37_well_map
    # for color, (x_idx, y_idx) in sample37_well_map.items():
    #     center_x = x_idx * x_spacing
    #     center_y = y_idx * y_spacing
    #     circle = plt.Circle((center_x, center_y), 2, color=color, fill=False, linewidth=2)
    #     plt.gca().add_artist(circle)
    # plt.xlim(-5, 20)
    # plt.ylim(-5, 20)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel('X Position (um)')
    # plt.ylabel('Y Position (um)')
    # plt.title('Well Positions for Sample 37')
    # plt.grid()
    # plt.show()