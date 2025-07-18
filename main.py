from AFMImageCollection import AFMImageCollection
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import visualizations as vis

folder_path = "C:/Users/chasg/afmCode/DataFolder" # Erfan's is D:/afmCode/DataFolder
collection = AFMImageCollection(folder_path)
num_images = collection.get_number_of_images()
print("=================================================")
print(f"Number of images in the array: {num_images}")
print("=================================================")

# TODO: Make it take the exact time based on line and scan rate

times = []
deflections = []
for i in range(num_images):
    image = collection.get_image(i)
    # vis.export_heightmap_3d_surface(image)
    # vis.height_and_defln_row_selector(image)


    taken = image.get_datetime()
    depressurized = taken.replace(hour=12, minute=46, second=1)
    time_unpressurized = taken - depressurized
    print(f"Time since depressurized: {time_unpressurized}")

    try:
        h1, h2 = vis.height_and_defln_row_selector(image)
        deflections.append(h2 - h1)
        times.append(time_unpressurized.total_seconds() / 60)
        print(deflections[i])
        print(times[i])
    except Exception as e:
        print(f"Error processing image {i}: {e}")
        continue

print(deflections)
print(times)
# plot deflection vs time
plt.scatter(times, deflections)
plt.xlabel('Time since depressurization (minutes)')
plt.ylabel('Deflection (nm)')
plt.title('Deflection vs Time')
plt.grid(True)
plt.show()

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