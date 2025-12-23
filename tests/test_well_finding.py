"""
Tests for MembraneNavigator well tracking within a single image.

These tests verify that given a well map and initial well position,
MembraneNavigator correctly tracks all wells in a single image within tight tolerances.
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AFMImage
import membrane_relative_positions as mrp

# Path to test images directory (to be populated with actual test images)
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'test_data', 'well_finding')

TEST_IMAGE_CONFIGS = {
    "Image0015.ibw": {
        "well_map": mrp.sample53_o_5_1_well_map,
        "initial_well": "(1, 5)",
        "initial_well_position": (-51.787, -6.115),  # absolute piezo position (μm)
        "expected_wells": {
            "(1, 5)":  (-51.787, -6.115, 124.870),  # (abs_x, abs_y, deflection_nm)
            "(1, 7)":  (-51.689,  3.141, 166.709),
            "(1, 9)":  (-51.787, 12.130, 162.104),

            "(2, 6)":  (-43.682, -1.102, 177.501),
            "(2, 8)":  (-43.779,  7.834, 185.380),
            "(2, 10)": (-43.877, 16.728, 180.460),

            "(3, 5)":  (-35.967, -5.624, 178.473),
            "(3, 7)":  (-35.967,  3.362, 190.088),
            "(3, 9)":  (-36.064, 12.318, 190.097),
        },
    },
}

# Tolerances for position matching
XY_TOLERANCE_UM = 0.2  # tolerance for x/y position
DEFLECTION_TOLERANCE_NM = 20.0  # tolerance for deflection


class TestMembraneNavigatorSingleImage:
    """
    Test suite for MembraneNavigator.track_wells on a single image.
    
    Tests that given an initial well position and well map, the navigator
    correctly finds all wells visible in the image within tolerance.
    """

    @pytest.fixture
    def navigator(self):
        """Create a MembraneNavigator instance."""
        return mrp.MembraneNavigator()

    @pytest.fixture
    def test_images(self):
        """Load all test images with configurations."""
        images = {}
        for filename in TEST_IMAGE_CONFIGS.keys():
            filepath = os.path.join(TEST_IMAGES_DIR, filename)
            if os.path.exists(filepath):
                images[filename] = AFMImage.AFMImage(filepath)
        return images

    def test_tracks_all_wells_in_single_image(self, navigator, test_images):
        """
        Verify that MembraneNavigator finds all expected wells in a single image.
        
        For each test image:
        1. Run track_wells with the initial well position
        2. Verify all expected wells are found
        3. Check positions are within tolerance
        """
        for filename, config in TEST_IMAGE_CONFIGS.items():
            if filename not in test_images:
                pytest.skip(f"Test image not found: {filename}")
            
            image = test_images[filename]
            well_map = config["well_map"]
            initial_well = config["initial_well"]
            initial_pos = config["initial_well_position"]
            expected_wells = config["expected_wells"]
            
            # Run track_wells on single image
            results = navigator.track_wells(
                image_collection=[image],
                initial_well_name=initial_well,
                initial_well_coords=well_map[initial_well],
                well_map=well_map,
                initial_well_absolute_pos=initial_pos
            )
            
            # Build lookup of results by well name
            results_by_well = {}
            for entry in results:
                well_name = entry['Well']
                if well_name not in results_by_well:
                    results_by_well[well_name] = entry
            
            # Verify all expected wells were found
            for well_name, (exp_x, exp_y, exp_deflection) in expected_wells.items():
                assert well_name in results_by_well, (
                    f"Well '{well_name}' not found in results for {filename}"
                )
                
                entry = results_by_well[well_name]
                
                # Get found position (Point 2 is the extremum/vertex)
                found_x = entry['Point 2 X (um)']
                found_y = entry['Point 2 Y (um)']
                found_deflection = entry['Deflection (nm)']
                
                # Convert to absolute position using image offset
                offset_x, offset_y = image.offset_image_origin_to_absolute_piezo_position()
                abs_x = offset_x + found_x
                abs_y = offset_y + found_y
                
                # Check XY position
                xy_distance = np.sqrt((abs_x - exp_x)**2 + (abs_y - exp_y)**2)
                deflection_error = abs(found_deflection - exp_deflection)
                
                # Print accuracy for each well
                print(f"  {well_name}: XY error = {xy_distance*1000:.1f} nm, "
                      f"deflection error = {deflection_error:.2f} nm")
                
                assert xy_distance < XY_TOLERANCE_UM, (
                    f"Well '{well_name}' XY mismatch in {filename}: "
                    f"expected ({exp_x:.2f}, {exp_y:.2f}), "
                    f"found ({abs_x:.2f}, {abs_y:.2f}), "
                    f"distance = {xy_distance:.3f} μm"
                )
                
                # Check deflection
                assert deflection_error < DEFLECTION_TOLERANCE_NM, (
                    f"Well '{well_name}' deflection mismatch in {filename}: "
                    f"expected {exp_deflection:.1f} nm, found {found_deflection:.1f} nm, "
                    f"error = {deflection_error:.1f} nm"
                )

    def test_no_missing_wells(self, navigator, test_images):
        """
        Verify the navigator doesn't miss any expected wells.
        """
        missing_wells = []
        
        for filename, config in TEST_IMAGE_CONFIGS.items():
            if filename not in test_images:
                continue
            
            image = test_images[filename]
            well_map = config["well_map"]
            initial_well = config["initial_well"]
            initial_pos = config["initial_well_position"]
            expected_wells = config["expected_wells"]
            
            results = navigator.track_wells(
                image_collection=[image],
                initial_well_name=initial_well,
                initial_well_coords=well_map[initial_well],
                well_map=well_map,
                initial_well_absolute_pos=initial_pos
            )
            
            found_wells = set(entry['Well'] for entry in results)
            
            for well_name in expected_wells.keys():
                if well_name not in found_wells:
                    missing_wells.append((filename, well_name))
        
        assert len(missing_wells) == 0, (
            f"Navigator missed {len(missing_wells)} expected wells:\n" +
            "\n".join(f"  {fn}: '{well}'" for fn, well in missing_wells)
        )

    def test_accuracy_statistics(self, navigator, test_images):
        """
        Compute and report accuracy statistics across all test wells.
        """
        errors = []
        
        for filename, config in TEST_IMAGE_CONFIGS.items():
            if filename not in test_images:
                continue
            
            image = test_images[filename]
            well_map = config["well_map"]
            initial_well = config["initial_well"]
            initial_pos = config["initial_well_position"]
            expected_wells = config["expected_wells"]
            
            results = navigator.track_wells(
                image_collection=[image],
                initial_well_name=initial_well,
                initial_well_coords=well_map[initial_well],
                well_map=well_map,
                initial_well_absolute_pos=initial_pos
            )
            
            results_by_well = {entry['Well']: entry for entry in results}
            offset_x, offset_y = image.offset_image_origin_to_absolute_piezo_position()
            
            for well_name, (exp_x, exp_y, exp_deflection) in expected_wells.items():
                if well_name in results_by_well:
                    entry = results_by_well[well_name]
                    abs_x = offset_x + entry['Point 2 X (um)']
                    abs_y = offset_y + entry['Point 2 Y (um)']
                    found_deflection = entry['Deflection (nm)']
                    
                    xy_dist = np.sqrt((abs_x - exp_x)**2 + (abs_y - exp_y)**2)
                    deflection_err = abs(found_deflection - exp_deflection)
                    errors.append({'xy': xy_dist, 'deflection': deflection_err, 'well': well_name})
        
        if errors:
            xy_errors = [e['xy'] for e in errors]
            deflection_errors = [e['deflection'] for e in errors]
            print(f"\nMembraneNavigator accuracy statistics:")
            print(f"  XY         - Mean: {np.mean(xy_errors):.4f} μm, Max: {np.max(xy_errors):.4f} μm, Std: {np.std(xy_errors):.4f} μm")
            print(f"  Deflection - Mean: {np.mean(deflection_errors):.1f} nm, Max: {np.max(deflection_errors):.1f} nm, Std: {np.std(deflection_errors):.1f} nm")
            print(f"  Total wells tested: {len(errors)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
