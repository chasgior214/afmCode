import numpy as np
import matplotlib.pyplot as plt
import AFMImage
import AFMImageCollection
import surface_analysis as sa
import path_loader as pl
import visualizations as vis
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button  # added import (Slider + Button)

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are, gets the deflections autonomously, and logs the data. Will be important for sample53, but also speeds up sample37 (only have to pick one well per image cuts time down aby about 40%, and some persistence between images when there's not much time between them could cut 90+% of the time spent picking wells)
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

    def offset_image_origin_to_absolute_piezo_position(self, image: AFMImage.AFMImage):
        """Get the offset to translate the image origin (the bottom left corner of the image) to absolute piezo position."""
        x_size, y_size = image.get_x_y_size()
        slow_scan_size = image.get_SlowScanSize()
        scan_direction = image.get_scan_direction()
        image_origin_x_offset_to_image_centre = -0.5 * x_size
        if scan_direction:  # scan down
            image_origin_y_offset_to_image_centre = 0.5 * slow_scan_size - y_size
        else:
            print('scan up not implemented yet')
            image_origin_y_offset_to_image_centre = 0.5 * slow_scan_size - y_size # Fallback/Placeholder

        # adjust for image offsets
        image_origin_absolute_x = image.get_x_offset() + image_origin_x_offset_to_image_centre
        image_origin_absolute_y = image.get_y_offset() + image_origin_y_offset_to_image_centre
        return (image_origin_absolute_x, image_origin_absolute_y)

    def predict_position_from_change_in_coordinates(self, pos, pos_coords, final_coords):
        """Predict the position of a well based on the position of another well and each well's coordinates."""
        x_coords_change = final_coords[0] - pos_coords[0]
        y_coords_change = final_coords[1] - pos_coords[1]
        predicted_x_pos = pos[0] + x_coords_change * x_spacing
        predicted_y_pos = pos[1] + y_coords_change * y_spacing
        return (predicted_x_pos, predicted_y_pos)

    def image_bounds_absolute_positions(self, image: AFMImage.AFMImage):
        """Get the bounds of the image in absolute piezo positions.
        Returns (x_min, y_min, x_max, y_max)"""
        image_origin_absolute_x, image_origin_absolute_y = self.offset_image_origin_to_absolute_piezo_position(image)
        x_size, y_size = image.get_x_y_size()
        return (image_origin_absolute_x, image_origin_absolute_y, image_origin_absolute_x + x_size, image_origin_absolute_y + y_size)

    def track_wells(self, image_collection, initial_well_name, initial_well_coords, well_map, initial_well_absolute_pos=None, edge_tolerance=0, each_found_well_updates_all_well_positions=False):
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
            absolute_image_bounds = self.image_bounds_absolute_positions(image)
            pixel_size = image.get_pixel_size()
            scan_size = image.get_scan_size()
            x_pixel_count, y_pixel_count = image.get_x_y_pixel_counts()
            x_coords = np.linspace(0, scan_size, x_pixel_count)
            print(f"Image bounds: {absolute_image_bounds}")

            # Determine height map once per image
            imaging_mode = image.get_imaging_mode()
            if imaging_mode == 'AC Mode' and image.wave_data.shape[2] > 4:
                height_map = image.get_FlatHeight()
            elif imaging_mode == 'Contact' and image.wave_data.shape[2] > 3:
                height_map = image.get_FlatHeight()
            else:
                height_map = image.get_height_retrace() # Fallback

            for well in well_positions:
                if well_positions[well] is None:
                    continue

                print(f"\tChecking {well} with estimated position {well_positions[well]}")
                
                # Check if well is within image bounds or within edge tolerance
                well_x, well_y = well_positions[well]
                x_min, y_min, x_max, y_max = absolute_image_bounds
                
                # Calculate distance from well to image bounds
                dist_to_left = well_x - x_min
                dist_to_right = x_max - well_x
                dist_to_bottom = well_y - y_min
                dist_to_top = y_max - well_y
                
                # Check if well is strictly inside bounds
                inside_bounds = (x_min < well_x < x_max and y_min < well_y < y_max)
                
                # Check if well is within tolerance of bounds
                within_tolerance = (
                    dist_to_left >= -edge_tolerance and dist_to_right >= -edge_tolerance and
                    dist_to_bottom >= -edge_tolerance and dist_to_top >= -edge_tolerance
                )
                
                if inside_bounds or (edge_tolerance > 0 and within_tolerance):
                    
                    # Clamp position to image bounds for fitting
                    clamped_x = np.clip(well_x, x_min, x_max)
                    clamped_y = np.clip(well_y, y_min, y_max)
                    
                    if inside_bounds:
                        print(f"Found {well} in image {image.bname}")
                    else:
                        print(f"Found {well} near edge of image {image.bname} (within {edge_tolerance} μm tolerance), using clamped position")
                    
                    # Calculate relative position for fitting (using clamped position)
                    rel_x = clamped_x - absolute_image_bounds[0]
                    rel_y = clamped_y - absolute_image_bounds[1]

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
                            well_time = image.get_line_acquisition_time(y_idx)
                            
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
                            
                            if each_found_well_updates_all_well_positions:
                                # It can cause one well not being found correctly to mess up all the others
                                for other_well in well_map:
                                    if other_well != well:
                                        well_positions[other_well] = self.predict_position_from_change_in_coordinates(well_positions[well], well_map[well], well_map[other_well])
                        else:
                            print(f"Failed to calculate substrate height for {well}")
                    else:
                        print(f"Fit failed for {well}")
        
        return results

# Sample 37 Configuration
sample37_wells_as_coords = {
    'orange': (0, 0),
    'blue': (0, 2),
    'green': (1, 1),
    'red': (1, 3),
    'black': (2, 0)
}

# plot a 4um diameter circle at each position in sample37_wells_as_coords
# for color, (x_idx, y_idx) in sample37_wells_as_coords.items():
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

# Example Usage / Script
if __name__ == "__main__":
    navigator = MembraneNavigator()

    image_collection = AFMImageCollection.AFMImageCollection(pl.afm_images_path, pl.depressurized_datetime)
    
    # Show the first image to pick a well
    first_image = image_collection.images[0]
    first_image_origin_absolute_x, first_image_origin_absolute_y = navigator.offset_image_origin_to_absolute_piezo_position(first_image)
    
    print(f"First image origin absolute position: x={first_image_origin_absolute_x:.3f} μm, y={first_image_origin_absolute_y:.3f} μm")
    
    # User Interaction
    res = vis.select_heights(first_image)
    slots = res.get('selected_slots', [None, None])
    well_clicked_on_extremum = slots[1]
    
    if well_clicked_on_extremum:
        well_clicked_on = input("Enter well name (e.g., green, orange): ")
        if well_clicked_on in sample37_wells_as_coords:
            well_clicked_on_coords = sample37_wells_as_coords[well_clicked_on]
            
            well_clicked_on_absolute_x = well_clicked_on_extremum[1] + first_image_origin_absolute_x
            well_clicked_on_absolute_y = well_clicked_on_extremum[2] + first_image_origin_absolute_y
            
            initial_pos = (well_clicked_on_absolute_x, well_clicked_on_absolute_y)
            
            print(f"Tracking wells starting from {well_clicked_on} at {initial_pos}")
            
            results = navigator.track_wells(
                image_collection, 
                well_clicked_on, 
                well_clicked_on_coords, 
                sample37_wells_as_coords, 
                initial_well_absolute_pos=initial_pos,
                edge_tolerance=0.3,
                each_found_well_updates_all_well_positions=True
            )
            
            print("Results:")
            for entry in results:
                # onlt print first 5 and last 5 entries to avoid flooding the output
                if results.index(entry) < 5 or results.index(entry) >= len(results) - 5:
                    print(entry)
            
            # Plot results
            for entry in results:
                plt.scatter(entry['Time (minutes)'], entry['Deflection (nm)'], color=entry['Well'], label=entry['Well'])
            plt.xlabel('Time (min)')
            plt.ylabel('Deflection (nm)')
            plt.show()
            
            # Plot the found well positions from results converted to absolute positions
            # Create image lookup map
            image_map = {img.bname: img for img in image_collection.images}
            
            # Prepare interactive plot with slider for time window
            if not results:
                print("No results to plot.")
            else:
                # Precompute absolute coordinates for all results (skip entries with missing images)
                abs_points = []
                for entry in results:
                    img = image_map.get(entry['Image Name'])
                    if img is None:
                        continue
                    ox, oy = navigator.offset_image_origin_to_absolute_piezo_position(img)
                    abs_x = ox + entry['Point 2 X (um)']
                    abs_y = oy + entry['Point 2 Y (um)']
                    abs_points.append((entry['Time (minutes)'], abs_x, abs_y, entry['Well']))

                if not abs_points:
                    print("No valid absolute points to plot.")
                else:
                    # sort by time so "next"/"prev" operate in chronological order
                    abs_points.sort(key=lambda p: p[0])
                    xs = [p[1] for p in abs_points]
                    ys = [p[2] for p in abs_points]
                    # small margin so points aren't on the edge
                    margin_x = max(1.0, 0.05 * (max(xs) - min(xs)))
                    margin_y = max(1.0, 0.05 * (max(ys) - min(ys)))
                    initial_xlim = (min(xs) - margin_x, max(xs) + margin_x)
                    initial_ylim = (min(ys) - margin_y, max(ys) + margin_y)

                    times = [p[0] for p in abs_points]
                    min_time, max_time = min(times), max(times)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.subplots_adjust(bottom=0.22)  # extra room for buttons

                    def draw_points(time_cut):
                        ax.clear()
                        drawn_labels = set()
                        for t, x, y, well in abs_points:
                            if t <= time_cut:
                                label = well if well not in drawn_labels else None
                                ax.scatter(x, y, color=well, s=100, alpha=0.6, label=label)
                                if label:
                                    drawn_labels.add(label)
                        ax.set_xlabel('Absolute X Position (μm)')
                        ax.set_ylabel('Absolute Y Position (μm)')
                        ax.set_title(f'Found Well Positions up to {time_cut:.1f} min')
                        ax.grid(True, alpha=0.3)
                        ax.set_aspect('equal', adjustable='box')
                        # always enforce the same limits computed from all valid points
                        ax.set_xlim(initial_xlim)
                        ax.set_ylim(initial_ylim)
                        if drawn_labels:
                            ax.legend(loc='upper right')
                        fig.canvas.draw_idle()

                    # Slider
                    axcolor = 'lightgoldenrodyellow'
                    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04], facecolor=axcolor)
                    time_slider = Slider(ax_slider, 'Minutes', min_time, max_time,
                                         valinit=max_time, valstep=max((max_time - min_time) / 200.0, 0.1))
                    time_slider.on_changed(lambda val: draw_points(val))

                    # Buttons: Prev / Next (show/remove one more result)
                    axprev = plt.axes([0.02, 0.08, 0.08, 0.04])
                    axnext = plt.axes([0.9, 0.08, 0.08, 0.04])
                    bprev = Button(axprev, 'Prev')
                    bnext = Button(axnext, 'Next')

                    # use a mutable container so nested callbacks can update the count without 'nonlocal'
                    current_n = [len(abs_points)]  # how many points are currently shown (1..N)

                    def show_n(n):
                        n = int(max(1, min(len(abs_points), n)))
                        current_n[0] = n
                        # set slider to time of nth point (this triggers draw_points via on_changed)
                        t = abs_points[n - 1][0]
                        time_slider.set_val(t)

                    def on_prev(event):
                        show_n(current_n[0] - 1)

                    def on_next(event):
                        show_n(current_n[0] + 1)

                    bprev.on_clicked(on_prev)
                    bnext.on_clicked(on_next)

                    # allow left/right arrow keys to navigate like the buttons
                    def on_key(event):
                        # event.key is typically 'left' / 'right' for arrow keys
                        if event.key in ('left', 'arrowleft'):
                            show_n(current_n[0] - 1)
                        elif event.key in ('right', 'arrowright'):
                            show_n(current_n[0] + 1)

                    fig.canvas.mpl_connect('key_press_event', on_key)

                    # initial draw with all points visible
                    draw_points(max_time)
                    plt.show()
        else:
            print(f"Well '{well_clicked_on}' not found in coordinates map.")
    else:
        print("No well selected.")