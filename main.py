from AFMImageCollection import AFMImageCollection
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import visualizations as vis

folder_path = "C:/Users/chasg/afmCode/DataFolder" # Erfan's is D:/afmCode/DataFolder
collection = AFMImageCollection(folder_path)
num_images = len(collection)
print("=================================================")
print(f"Number of images in the array: {num_images}")
print("=================================================")

times = []
deflections = []
depressurized = collection[0].get_datetime()
depressurized = depressurized.replace(hour=17, minute=31, second=20) # comment out to set t = 0 to first image time
if collection[0].get_datetime() < depressurized: # depressurization and first image time likely cross midnight
    depressurized = depressurized - timedelta(days=1)

print(f"Depressurized at {depressurized}\n")

for image in collection:
    # vis.export_heightmap_3d_surface(image)

    taken = image.get_datetime()
    print(f"Image {image.bname} saved {taken - depressurized} minutes after depressurization")
    try:
        h1, h2, line_time_offset = vis.select_heights(image)
        deflections.append(h2 - h1)
        time_unpressurized = (taken - depressurized).total_seconds() + line_time_offset
        times.append(time_unpressurized/60)
        print(deflections[-1])
        print(times[-1])
    except Exception as e:
        print(f"Error processing image {image.bname}: {e}")
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

# t = collection[0].get_datetime()

# T = [] 
# max_height = [] 

# for image in collection:
#     x,y,max_value, shift = image.get_trimmed_trace_z(3)
#     timediff = image.get_datetime() - t 
#     T.append(timediff.total_seconds()/3600)
#     max_height.append(max_value)

# x,y,max_value, shift = collection[0].get_trimmed_trace_z(3)

# plt.scatter(T,max_height)
# plt.xlabel('Time (hours)')
# plt.ylabel('Max Height (nm)')
# plt.show()