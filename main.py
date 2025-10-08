from AFMImageCollection import AFMImageCollection
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import visualizations as vis
import csv
import os

folder_path = "C:/Users/chasg/afmCode/DataFolder"
collection = AFMImageCollection(folder_path)
num_images = len(collection)
print("=================================================")
print(f"Number of images in the array: {num_images}")
print("=================================================")

############################ INPUTS ############################
depressurized_time = '16:05:25' # 'HH:MM:SS' format
save_to_csv = 1  # set true to save to CSV
# set if saving to CSV:
sample_number = 37
transfer_location = '$(6,3)'
cavity_position = 'red'



times = []
deflections = []
depressurized = collection[0].get_datetime()
depressurized = depressurized.replace(hour=int(depressurized_time.split(':')[0]), minute=int(depressurized_time.split(':')[1]), second=int(depressurized_time.split(':')[2])) # comment out to set t = 0 to first image time
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

# save to CSV
if save_to_csv:
    date_time_depressurized = depressurized.strftime('%Y%m%d_%H%M%S')
    filename = f'deflation_curve_{sample_number}_depressurized{date_time_depressurized}_loc{transfer_location}_cav{cavity_position}.csv'
    dir_path = os.path.join(folder_path, 'deflation_curves')
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if os.path.exists(file_path):
        print(f"File {file_path} already exists")
        # if RENAME_ME.csv also already exists, tell user to rename it first, pause program
        if os.path.exists('RENAME_ME.csv'):
            input("Please rename or delete RENAME_ME.csv and press Enter to continue...") 
        file_path = 'RENAME_ME.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (minutes)', 'Deflection (nm)'])
        for t, d in zip(times, deflections):
            writer.writerow([t, d])

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