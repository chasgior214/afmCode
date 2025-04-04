from AFMImageCollection import AFMImageCollection
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import visualizations as vis

folder_path = "C:/Users/chasg/afmCode/DataFolder" # Erfan's is D:/afmCode/DataFolder
collection = AFMImageCollection(folder_path)
num_images = collection.get_number_of_images()
print("=================================================")
print(f"Number of images in the array: {num_images}")
print("=================================================")

for i in range(num_images):
    image = collection.get_image(i)
    # vis.export_heightmap_3d_surface(image)
    vis.height_and_defln_row_selector(image)
    # note height_and_defln is currently modified to correct for a specific slope
    # vis.height_and_defln(image, 2.383)


# collection.review_phase()


# collection.export_deflection_time(3)
# collection.export_shift(3)

# t = collection.get_image(0).get_datetime()

# T = [] 
# max_height = [] 

# for i in range(num_images):
    
#     x,y,max_value, shift = collection.get_image(i).get_trimmed_trace_z(3)
#     timediff = collection.get_image(i).get_datetime() - t 
#     T.append(timediff.total_seconds()/3600)
#     max_height.append(max_value)
    
# x,y,max_value, shift = collection.get_image(0).get_trimmed_trace_z(3)


# plt.scatter(T,max_height)
# plt.xlabel('Time (hours)')
# plt.ylabel('Max Height (nm)')
# plt.show()