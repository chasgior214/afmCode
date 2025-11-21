import numpy as np
import matplotlib.pyplot as plt

# GOAL: have my system know where the wells are relative to each other, so for any images with multiple wells, I only point out one well, and it figures out where the others are and logs their data. Will be important for sample53, but also speeds up sample37 (only have to pick one well per image cuts time down aby about 40%)

x_spacing = 7.63 # um TODO update with an average over a big image
y_spacing = 4.6 # um TODO update with an average over a big image

# if position (0,0) is on a well, other wells are at any combination of movements away from that, where a movement is either (+x_spacing, +y_spacing), (+2*x_spacing, 0), or (0, +2*y_spacing)

# make a plot of the wells in the square bounded by (0,0), (10*x_spacing, 10*y_spacing)

def well_positions(x_coords, y_coords):
    positions = []
    for i in range(x_coords):
        for j in range(y_coords):
            if (i + j) % 2 == 0:
                positions.append((i*x_spacing, j*y_spacing))
    return positions

positions = well_positions(15, 17)

plt.scatter(*zip(*positions))
plt.show()

# next, code relating coordinates on my maps to this