"""
Well map module for managing well geometry and loading well map configurations.

This module contains:
- Well geometry/spacing constants (x_spacing, y_spacing, well_diameter, well_radius)
- WellMap dataclass for type-safe well map handling
- Geometry functions for well position calculations
- Functions for loading well maps from JSON files
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import json
import os
import path_loader as pl

# Well geometry/spacing constants, all um
x_spacing = 7.79
y_spacing = 4.50
well_diameter = 4.0
well_radius = well_diameter / 2


@dataclass
class WellMap:
    """A well map configuration with sample ID, location, and well coordinates."""
    sample_id: str
    location: str
    wells: Dict[str, Tuple[int, int]]
    
    def get_coords(self, well_name: str) -> Tuple[int, int]:
        """Get coordinates for a well by name."""
        if well_name not in self.wells:
            raise ValueError(f"Well {well_name} not found in map")
        return self.wells[well_name]
    
    def __contains__(self, well_name: str) -> bool:
        """Check if a well name exists in this map."""
        return well_name in self.wells
    
    def __iter__(self):
        """Iterate over (well_name, coords) pairs."""
        return iter(self.wells.items())


def well_positions_grid(x_count: int, y_count: int) -> List[Tuple[float, float]]:
    """
    Generate a grid of well positions.
    
    Wells are positioned on the substrate such that if (0,0) is on a well, other 
    wells are at positions (i,j) where (i+j) is even.
    
    Args:
        x_count: Number of grid positions in x direction
        y_count: Number of grid positions in y direction
        
    Returns:
        List of (x, y) positions in micrometers
    """
    positions = []
    for i in range(x_count):
        for j in range(y_count):
            if (i + j) % 2 == 0:
                positions.append((i * x_spacing, j * y_spacing))
    return positions


def predict_position_from_change_in_coordinates(
    pos: Tuple[float, float], 
    pos_coords: Tuple[int, int], 
    final_coords: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Predict a well's position based on another well's position and coordinates.
    
    Args:
        pos: Known (x, y) position of a well in micrometers
        pos_coords: Grid coordinates (i, j) of the known well
        final_coords: Grid coordinates (i, j) of the well to predict
        
    Returns:
        Predicted (x, y) position in micrometers
    """
    dx = (final_coords[0] - pos_coords[0]) * x_spacing
    dy = (final_coords[1] - pos_coords[1]) * y_spacing
    return (pos[0] + dx, pos[1] + dy)


def load_well_map(sample_id: str, location: str) -> WellMap:
    """
    Load a well map from the well_maps_path directory.
    
    Args:
        sample_id: Sample identifier (e.g. "37")
        location: Transfer location (e.g. "$(6,3)")
        
    Returns:
        WellMap object
        
    Raises:
        FileNotFoundError: If the well map file does not exist
    """
    filename = f"sample{sample_id}_{location}.json"
    filepath = os.path.join(pl.well_maps_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Well map file {filepath} not found")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    wells = {k: tuple(v) for k, v in data["wells"].items()}
    return WellMap(
        sample_id=data["sample_id"],
        location=data["location"],
        wells=wells
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Plot sample53 well coordinates as blue 4um diameter circles
    sample53 = load_well_map("53", "o(5,1)")
    if sample53:
        for well_name, (x_idx, y_idx) in sample53.wells.items():
            center_x = x_idx * x_spacing
            center_y = y_idx * y_spacing
            circle = plt.Circle((center_x, center_y), 2, color='blue', fill=True, linewidth=2)
            plt.gca().add_artist(circle)
        plt.xlim(0, x_spacing*15)
        plt.ylim(0, y_spacing*17)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X Position (um)')
        plt.ylabel('Y Position (um)')
        plt.title('Well Positions for Sample 53')
        plt.grid()
        plt.show()

    # Plot sample37 well map with colored circles
    sample37 = load_well_map("37", "$(6,3)")
    if sample37:
        for color, (x_idx, y_idx) in sample37.wells.items():
            center_x = x_idx * x_spacing
            center_y = y_idx * y_spacing
            circle = plt.Circle((center_x, center_y), 2, color=color, fill=False, linewidth=2)
            plt.gca().add_artist(circle)
        plt.xlim(-5, 20)
        plt.ylim(-5, 20)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X Position (um)')
        plt.ylabel('Y Position (um)')
        plt.title('Well Positions for Sample 37')
        plt.grid()
        plt.show()
