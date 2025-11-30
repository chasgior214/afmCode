import numpy as np
import matplotlib.pyplot as plt

import AFMImage

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are, gets the deflections autonomously, and logs the data. Will be important for sample53, but also speeds up sample37 (only have to pick one well per image cuts time down aby about 40%, and some persistence between images when there's not much time between them could cut 90+% of the time spent picking wells)

x_spacing = 7.63 # um TODO update with an average over a big image
y_spacing = 4.6 # um TODO update with an average over a big image

# if position (0,0) is on a well, other wells are at any combination of movements away from that, where a movement is either (+x_spacing, +y_spacing), (+2*x_spacing, 0), or (0, +2*y_spacing)


def well_positions(x_coords, y_coords):
    positions = []
    for i in range(x_coords):
        for j in range(y_coords):
            if (i + j) % 2 == 0:
                positions.append((i*x_spacing, j*y_spacing))
    return positions

positions = well_positions(15, 17)

# plt.scatter(*zip(*positions))
# plt.show()

# next, code relating coordinates on my maps to this




###########################################################
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


###########################################################
def offset_image_origin_to_absolute_piezo_position(image : AFMImage.AFMImage):
    """Get the offset to translate the image origin (the bottom left corner of the image) to absolute piezo position (defined where (0,0) is the neutral position of the x/y piezos respectively)."""
    x_size, y_size = image.get_x_y_size()
    slow_scan_size = image.get_SlowScanSize()
    scan_direction = image.get_scan_direction()
    image_origin_x_offset_to_image_centre = -0.5 * x_size
    if scan_direction: # scan down
        image_origin_y_offset_to_image_centre = 0.5 * slow_scan_size - y_size
    else:
        print('scan up not implemented yet')
    # adjust for image offsets
    image_origin_absolute_x = image.get_x_offset() + image_origin_x_offset_to_image_centre
    image_origin_absolute_y = image.get_y_offset() + image_origin_y_offset_to_image_centre
    return (image_origin_absolute_x, image_origin_absolute_y)


############################################################


test_image_path = 'Image0030.ibw'
afm_image = AFMImage.AFMImage(test_image_path)
x_offset = afm_image.get_x_offset()
y_offset = afm_image.get_y_offset()
import visualizations as vis
from matplotlib.patches import Circle
res = vis.select_heights(afm_image)
slots = res.get('selected_slots', [None, None])
green_extremum = slots[1]
green_x_um = green_extremum[1]
green_y_um = green_extremum[2]
if len(green_extremum) >= 5:
    x_px = green_extremum[3]
    y_px = green_extremum[4]
    print(f"Vertex: x={green_x_um:.3f} μm, y={green_y_um:.3f} μm, px=({x_px},{y_px})")

image_origin_absolute_x, image_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(afm_image)
absolute_x_vertex_green, absolute_y_vertex_green = image_origin_absolute_x + green_x_um, image_origin_absolute_y + green_y_um
print(f"Absolute piezo position of vertex: x={absolute_x_vertex_green:.3f} μm, y={absolute_y_vertex_green:.3f} μm")

# plot the vertex position on a map of absolute position going from -20 to 20 um in x and y (just a point on a blank plot)
plt.scatter(absolute_x_vertex_green, absolute_y_vertex_green, color='green')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X Position (μm)')
plt.ylabel('Y Position (μm)')
plt.title('Absolute Piezo Position of Vertex')
plt.grid()
plt.show()

# pick black too
res_black = vis.select_heights(afm_image)
slots_black = res_black.get('selected_slots', [None, None])
black_extremum = slots_black[1]
black_x_um = black_extremum[1]
black_y_um = black_extremum[2]
if len(black_extremum) >= 5:
    x_px_black = black_extremum[3]
    y_px_black = black_extremum[4]
    print(f"Vertex: x={black_x_um:.3f} μm, y={black_y_um:.3f} μm, px=({x_px_black},{y_px_black})")
absolute_x_vertex_black, absolute_y_vertex_black = image_origin_absolute_x + black_x_um, image_origin_absolute_y + black_y_um
print(f"Absolute piezo position of vertex: x={absolute_x_vertex_black:.3f} μm, y={absolute_y_vertex_black:.3f} μm")
# plot both points on the same map
plt.scatter(absolute_x_vertex_green, absolute_y_vertex_green, color='green', label='Green Well')
plt.scatter(absolute_x_vertex_black, absolute_y_vertex_black, color='black', label='Black Well')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X Position (μm)')
plt.ylabel('Y Position (μm)')
plt.title('Absolute Piezo Positions of Vertices')
plt.legend()
plt.grid()
plt.show()

# add 4 dots representing the corners of the image
x_size, y_size = afm_image.get_x_y_size()
image_corners = [
    (image_origin_absolute_x, image_origin_absolute_y),  # bottom-left
    (image_origin_absolute_x + x_size, image_origin_absolute_y),  # bottom-right
    (image_origin_absolute_x, image_origin_absolute_y + y_size),  # top-left
    (image_origin_absolute_x + x_size, image_origin_absolute_y + y_size)  # top-right
]
print(f"Image origin absolute position: x={image_origin_absolute_x:.3f} μm, y={image_origin_absolute_y:.3f} μm")
print(f"Image x offset: {x_offset:.3f} μm, y offset: {y_offset:.3f} μm")
for corner_x, corner_y in image_corners:
    plt.scatter(corner_x, corner_y, color='gray', marker='o', s=20)
ax = plt.gca()
r = 2  # radius in μm -> 4 μm diameter
ax.add_patch(Circle((absolute_x_vertex_green, absolute_y_vertex_green), r, edgecolor='green', facecolor='none', linewidth=2, label='Green Well'))
ax.add_patch(Circle((absolute_x_vertex_black, absolute_y_vertex_black), r, edgecolor='black', facecolor='none', linewidth=2, label='Black Well'))
# keep center markers if desired
ax.scatter([absolute_x_vertex_green, absolute_x_vertex_black], [absolute_y_vertex_green, absolute_y_vertex_black], color=['green','black'])
ax = plt.gca()
plt.xlim(-10, 10)
plt.ylim(-15, 0)
ax.set_aspect('equal', adjustable='box')
plt.show()
