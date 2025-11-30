import numpy as np
import matplotlib.pyplot as plt

import AFMImage

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are, gets the deflections autonomously, and logs the data. Will be important for sample53, but also speeds up sample37 (only have to pick one well per image cuts time down aby about 40%, and some persistence between images when there's not much time between them could cut 90+% of the time spent picking wells)


######################## CONSTANTS ########################
x_spacing = 7.63 # um TODO update with an average over a big image
y_spacing = 4.6 # um TODO update with an average over a big image
well_diameter = 4 # um
well_radius = well_diameter / 2

time_between_images_for_safe_minimal_drift = 10 # seconds
###########################################################

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

def predict_position_from_change_in_coordinates(pos, pos_coords, final_coords):
    """Predict the position of a well based on the position of another well and each well's coordinates."""
    x_coords_change = final_coords[0] - pos_coords[0]
    y_coords_change = final_coords[1] - pos_coords[1]
    predicted_x_pos = pos[0] + x_coords_change * x_spacing
    predicted_y_pos = pos[1] + y_coords_change * y_spacing
    return (predicted_x_pos, predicted_y_pos)

############################################################
import visualizations as vis
from matplotlib.patches import Circle

if 1:
    test_image_path = 'Image0030.ibw'
    afm_image = AFMImage.AFMImage(test_image_path)
    x_offset = afm_image.get_x_offset()
    y_offset = afm_image.get_y_offset()

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
    r = well_radius
    ax.add_patch(Circle((absolute_x_vertex_green, absolute_y_vertex_green), r, edgecolor='green', facecolor='none', linewidth=2, label='Green Well'))
    ax.add_patch(Circle((absolute_x_vertex_black, absolute_y_vertex_black), r, edgecolor='black', facecolor='none', linewidth=2, label='Black Well'))
    # add a black x where black is predicted to be as offset from green
    ax.scatter(predict_position_from_change_in_coordinates((absolute_x_vertex_green, absolute_y_vertex_green), sample37_wells_as_coords['green'], sample37_wells_as_coords['black'])[0], predict_position_from_change_in_coordinates((absolute_x_vertex_green, absolute_y_vertex_green), sample37_wells_as_coords['green'], sample37_wells_as_coords['black'])[1], color='black', marker='x', s=20)
    # keep center markers if desired
    ax.scatter([absolute_x_vertex_green, absolute_x_vertex_black], [absolute_y_vertex_green, absolute_y_vertex_black], color=['green','black'])
    ax = plt.gca()
    plt.xlim(-10, 10)
    plt.ylim(-15, 0)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# do the same for Image0037
if 0:
    image_0037_path = 'Image0037.ibw'
    afm_image_0037 = AFMImage.AFMImage(image_0037_path)
    x_offset_0037 = afm_image_0037.get_x_offset()
    y_offset_0037 = afm_image_0037.get_y_offset()
    image_origin_absolute_x, image_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(afm_image_0037)
    res = vis.select_heights(afm_image_0037)
    slots = res.get('selected_slots', [None, None])
    teal_extremum = slots[1]
    teal_x_um = teal_extremum[1]
    teal_y_um = teal_extremum[2]
    if len(teal_extremum) >= 5:
        x_px_teal = teal_extremum[3]
        y_px_teal = teal_extremum[4]
        print(f"Vertex: x={teal_x_um:.3f} μm, y={teal_y_um:.3f} μm, px=({x_px_teal},{y_px_teal})")
    absolute_x_vertex_teal, absolute_y_vertex_teal = image_origin_absolute_x + teal_x_um, image_origin_absolute_y + teal_y_um
    print(f"Absolute piezo position of vertex: x={absolute_x_vertex_teal:.3f} μm, y={absolute_y_vertex_teal:.3f} μm")

    res = vis.select_heights(afm_image_0037)
    slots = res.get('selected_slots', [None, None])
    brown_extremum = slots[1]
    brown_x_um = brown_extremum[1]
    brown_y_um = brown_extremum[2]
    if len(brown_extremum) >= 5:
        x_px_brown = brown_extremum[3]
        y_px_brown = brown_extremum[4]
        print(f"Vertex: x={brown_x_um:.3f} μm, y={brown_y_um:.3f} μm, px=({x_px_brown},{y_px_brown})")
    absolute_x_vertex_brown, absolute_y_vertex_brown = image_origin_absolute_x + brown_x_um, image_origin_absolute_y + brown_y_um
    print(f"Absolute piezo position of vertex: x={absolute_x_vertex_brown:.3f} μm, y={absolute_y_vertex_brown:.3f} μm")

    # plot both points on the same map with dots representing the corners of the image
    x_size, y_size = afm_image_0037.get_x_y_size()
    image_corners = [
        (image_origin_absolute_x, image_origin_absolute_y),  # bottom-left
        (image_origin_absolute_x + x_size, image_origin_absolute_y),  # bottom-right
        (image_origin_absolute_x, image_origin_absolute_y + y_size),  # top-left
        (image_origin_absolute_x + x_size, image_origin_absolute_y + y_size)  # top-right
    ]
    print(f"Image origin absolute position: x={image_origin_absolute_x:.3f} μm, y={image_origin_absolute_y:.3f} μm")
    print(f"Image x offset: {x_offset_0037:.3f} μm, y offset: {y_offset_0037:.3f} μm")
    for corner_x, corner_y in image_corners:
        plt.scatter(corner_x, corner_y, color='gray', marker='o', s=20)
    ax = plt.gca()
    r = 2  # radius in μm -> 4 μm diameter
    ax.add_patch(Circle((absolute_x_vertex_teal, absolute_y_vertex_teal), r, edgecolor='teal', facecolor='none', linewidth=2, label='Teal Well'))
    ax.add_patch(Circle((absolute_x_vertex_brown, absolute_y_vertex_brown), r, edgecolor='brown', facecolor='none', linewidth=2, label='Brown Well'))
    ax.scatter([absolute_x_vertex_teal, absolute_x_vertex_brown], [absolute_y_vertex_teal, absolute_y_vertex_brown], color=['teal','brown'])

    # add a brown x where brown is predicted to be as offset from teal
    teal_coords = (0,0)
    brown_coords = (3,5)
    brown_predicted_position = predict_position_from_change_in_coordinates((absolute_x_vertex_teal, absolute_y_vertex_teal), teal_coords, brown_coords)
    ax.scatter(brown_predicted_position[0], brown_predicted_position[1], color='brown', marker='x', s=20)

    ax.set_aspect('equal', adjustable='box')
    plt.show()


################################################
if 1:
    image_0020 = AFMImage.AFMImage('Image0020.ibw')
    image_0021 = AFMImage.AFMImage('Image0021.ibw')

    image_0020_x_offset = image_0020.get_x_offset()
    image_0020_y_offset = image_0020.get_y_offset()
    image_0021_x_offset = image_0021.get_x_offset()
    image_0021_y_offset = image_0021.get_y_offset()

    image_0020_origin_absolute_x, image_0020_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(image_0020)
    image_0021_origin_absolute_x, image_0021_origin_absolute_y = offset_image_origin_to_absolute_piezo_position(image_0021)

    res = vis.select_heights(image_0020)
    slots = res.get('selected_slots', [None, None])
    green_extremum = slots[1]
    green_x_um = green_extremum[1]
    green_y_um = green_extremum[2]
    if len(green_extremum) >= 5:
        x_px_green = green_extremum[3]
        y_px_green = green_extremum[4]
        print(f"Vertex: x={green_x_um:.3f} μm, y={green_y_um:.3f} μm, px=({x_px_green},{y_px_green})")
    absolute_x_vertex_green, absolute_y_vertex_green = image_0020_origin_absolute_x + green_x_um, image_0020_origin_absolute_y + green_y_um
    print(f"Absolute piezo position of green vertex: x={absolute_x_vertex_green:.3f} μm, y={absolute_y_vertex_green:.3f} μm")

    res = vis.select_heights(image_0021)
    slots = res.get('selected_slots', [None, None])
    orange_extremum = slots[1]
    orange_x_um = orange_extremum[1]
    orange_y_um = orange_extremum[2]
    if len(orange_extremum) >= 5:
        x_px_orange = orange_extremum[3]
        y_px_orange = orange_extremum[4]
        print(f"Vertex: x={orange_x_um:.3f} μm, y={orange_y_um:.3f} μm, px=({x_px_orange},{y_px_orange})")
    absolute_x_vertex_orange, absolute_y_vertex_orange = image_0021_origin_absolute_x + orange_x_um, image_0021_origin_absolute_y + orange_y_um
    print(f"Absolute piezo position of orange vertex: x={absolute_x_vertex_orange:.3f} μm, y={absolute_y_vertex_orange:.3f} μm")

    # # now try prediction between images
    orange_predicted_position = predict_position_from_change_in_coordinates((absolute_x_vertex_green, absolute_y_vertex_green), (sample37_wells_as_coords['green'][0], sample37_wells_as_coords['green'][1]), (sample37_wells_as_coords['orange'][0], sample37_wells_as_coords['orange'][1]))
    print(f"Orange predicted position: x={orange_predicted_position[0]:.3f} μm, y={orange_predicted_position[1]:.3f} μm")
    print(f"Image 0021 origin absolute position: x={image_0021_origin_absolute_x:.3f} μm, y={image_0021_origin_absolute_y:.3f} μm")

    # plot the origin as an x, the vertex as a dot, and the corners as dots
    plt.scatter(image_0021_origin_absolute_x, image_0021_origin_absolute_y, color='gray', marker='x', s=100)
    plt.scatter(absolute_x_vertex_orange, absolute_y_vertex_orange, color='orange', marker='o', s=20)

    # # plot the predicted position, the vertex, and four dots for the corners of the image
    corners = [(image_0021_origin_absolute_x, image_0021_origin_absolute_y), (image_0021_origin_absolute_x + image_0021.get_x_y_size()[0], image_0021_origin_absolute_y), (image_0021_origin_absolute_x, image_0021_origin_absolute_y + image_0021.get_x_y_size()[1]), (image_0021_origin_absolute_x + image_0021.get_x_y_size()[0], image_0021_origin_absolute_y + image_0021.get_x_y_size()[1])]
    for corner in corners:
        plt.scatter(corner[0], corner[1], color='gray', marker='o', s=20)
    plt.scatter(absolute_x_vertex_orange, absolute_y_vertex_orange, color='orange', marker='o', s=20)
    plt.scatter(orange_predicted_position[0], orange_predicted_position[1], color='orange', marker='x', s=20)
    plt.show()


    ####################################################
    # incorporate using the predicted position of orange to get its deflection without ever opening image0021 in the GUI
    