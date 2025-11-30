import numpy as np
import matplotlib.pyplot as plt
import AFMImage
import AFMImageCollection
import surface_analysis as sa
import path_loader as pl
import visualizations as vis
from matplotlib.patches import Circle

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are, gets the deflections autonomously, and logs the data. Will be important for sample53, but also speeds up sample37 (only have to pick one well per image cuts time down aby about 40%, and some persistence between images when there's not much time between them could cut 90+% of the time spent picking wells)
# TODO update default x_spacing, y_spacing based on an average over a big image
class MembraneNavigator:
    def __init__(self, x_spacing=7.63, y_spacing=4.6, well_diameter=4, time_between_images_for_safe_minimal_drift=15):
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.well_diameter = well_diameter
        self.well_radius = well_diameter / 2
        self.time_between_images_for_safe_minimal_drift = time_between_images_for_safe_minimal_drift

    def well_positions_grid(self, x_coords, y_coords):
        """Generate a grid of well positions. My dies are made such that if position (0,0) is on a well, other wells are at any combination of movements away from that, where a movement is either (+x_spacing, +y_spacing), (+2*x_spacing, 0), or (0, +2*y_spacing)."""
        positions = []
        for i in range(x_coords):
            for j in range(y_coords):
                if (i + j) % 2 == 0:
                    positions.append((i * self.x_spacing, j * self.y_spacing))
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
        predicted_x_pos = pos[0] + x_coords_change * self.x_spacing
        predicted_y_pos = pos[1] + y_coords_change * self.y_spacing
        return (predicted_x_pos, predicted_y_pos)

    def image_bounds_absolute_positions(self, image: AFMImage.AFMImage):
        """Get the bounds of the image in absolute piezo positions.
        Returns (x_min, y_min, x_max, y_max)"""
        image_origin_absolute_x, image_origin_absolute_y = self.offset_image_origin_to_absolute_piezo_position(image)
        x_size, y_size = image.get_x_y_size()
        return (image_origin_absolute_x, image_origin_absolute_y, image_origin_absolute_x + x_size, image_origin_absolute_y + y_size)

    def track_wells(self, image_collection, initial_well_name, initial_well_coords, well_map, initial_well_absolute_pos=None, edge_tolerance=0):
        """
        Track wells across an image collection.
        
        Args:
            image_collection: AFMImageCollection object or list of AFMImage objects.
            initial_well_name: Name of the well identified in the first image.
            initial_well_coords: (x, y) coordinates of the initial well in the grid.
            well_map: Dictionary mapping well names to their grid coordinates (x, y).
            initial_well_absolute_pos: Optional (x, y) absolute position of the initial well. 
                                       If None, it will be determined from the first image (requires user interaction or pre-selection).
            edge_tolerance: Optional tolerance distance (in μm) to allow fitting wells just outside image bounds. 
                          If a well's predicted center is within this distance from the image edge, 
                          fitting will be attempted using the closest point within the image. Default is 0 (disabled).
        
        Returns:
            List of results: [(well_name, time, deflection), ...]
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
                            
                            results.append((well, well_time, deflection))
                            
                            # Update position to account for drift
                            absolute_vertex_x = vertex_x_um + absolute_image_bounds[0]
                            absolute_vertex_y = vertex_y_um + absolute_image_bounds[1]
                            well_positions[well] = (absolute_vertex_x, absolute_vertex_y)
                            
                            # Update other wells based on this drift? 
                            # It can cause one well not being found correctly to mess up all the others
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
                edge_tolerance=0.3
            )
            
            print("Results:", results)
            
            # Plot results
            for well, time, deflection in results:
                plt.scatter((time - pl.depressurized_datetime).total_seconds() / 60, deflection, color=well, label=well)
            plt.xlabel('Time (min)')
            plt.ylabel('Deflection (nm)')
            plt.show()
        else:
            print(f"Well '{well_clicked_on}' not found in coordinates map.")
    else:
        print("No well selected.")