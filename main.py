from AFMImageCollection import AFMImageCollection
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import visualizations as vis
import csv
import os
import path_loader as pl

################################################################################
save_to_csv = 1  # set true to save to CSV
end_hour = None # None to not give end limit
end_day = None # None to be same day as depressurization date
################################################################################

depressurized_datetime = pl.depressurized_datetime
if end_day is not None:
    end_datetime = depressurized_datetime.replace(day=end_day)
    if end_hour is not None:
        end_datetime = end_datetime.replace(hour=end_hour)
elif end_hour is not None:
    end_datetime = depressurized_datetime.replace(hour=end_hour)

folder_path = pl.afm_images_path

if end_day is not None or end_hour is not None:
    collection = AFMImageCollection(folder_path,
                                    start_datetime=depressurized_datetime,
                                    end_datetime=end_datetime)
else:
    collection = AFMImageCollection(folder_path,
                                    start_datetime=depressurized_datetime)
num_images = len(collection)

print("=================================================")
print(f"Depressurized at {depressurized_datetime}\n")
print(f"Number of images in the array: {num_images}")
print("=================================================")


times = []
deflections = []
pixel_coords = []

# Open an interactive navigator so the user can pick images in any order
selections = collection.navigate_images()

# selections is a dict keyed by image index -> {'selected_slots': [slot0, slot1], 'time_offset': val}
if selections is None:
    selections = {}

for idx in sorted(selections.keys()):
    try:
        res = selections[idx]
        image = collection[idx]
        taken = image.get_datetime()
        print(f"Image {image.bname} saved {taken - depressurized_datetime} after depressurization")

        slots = res.get('selected_slots', [None, None])
        time_offset = res.get('time_offset')

        # Only record deflection if both slots selected
        if slots[0] is not None and slots[1] is not None:
            h1 = slots[0][0]
            h2 = slots[1][0]
            delta_deflection = h2 - h1
            print(f"Deflection: {delta_deflection:.3f} nm")
            if time_offset is None:
                print(f"No time offset for image {image.bname}; skipping time entry")
                continue
            time_unpressurized = (taken - depressurized_datetime).total_seconds() + time_offset
            times.append(time_unpressurized / 60)
            deflections.append(delta_deflection)
            p1x = p1y = p2x = p2y = None
            if len(slots[0]) >= 5 and len(slots[1]) >= 5:
                p1x, p1y = int(slots[0][3]), int(slots[0][4])
                p2x, p2y = int(slots[1][3]), int(slots[1][4])
            pixel_coords.append((p1x, p1y, p2x, p2y))
            print(f"Time: {times[-1]:.3f} minutes")
        else:
            print(f"Image {image.bname} has incomplete selections; skipping deflection/time entry")
    except Exception as e:
        print(f"Error processing selection for image index {idx}: {e}")
        continue

print(deflections)
print(times)

# save to CSV
if save_to_csv:
    # only save if there is data
    if len(deflections) == 0:
        print("No deflection or time data to save; skipping CSV export")
    else:
        filename = pl.deflation_curve_filename
        dir_path = pl.deflation_curves_path
        file_path = pl.deflation_curve_path
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
            writer.writerow(['Time (minutes)', 'Deflection (nm)',
                             'Point 1 X Pixel', 'Point 1 Y Pixel',
                             'Point 2 X Pixel', 'Point 2 Y Pixel'])
            for t, d, (p1x, p1y, p2x, p2y) in zip(times, deflections, pixel_coords):
                writer.writerow([t, d, p1x, p1y, p2x, p2y])

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