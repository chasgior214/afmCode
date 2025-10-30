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


def _get_height_map_for_selection(image):
    mode = image.get_imaging_mode()
    depth = image.wave_data.shape[2] if getattr(image, 'wave_data', None) is not None else 0
    if mode == 'AC Mode' and depth > 4:
        return image.get_FlatHeight()
    if mode == 'Contact' and depth > 3:
        return image.get_FlatHeight()
    return image.get_height_retrace()


def _build_initial_selections(collection, csv_path, depressurized_dt):
    if not os.path.exists(csv_path):
        return None

    try:
        with open(csv_path, newline='') as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        print(f"Could not preload selections from {csv_path}: {exc}")
        return None

    if not rows:
        print(f"Existing CSV at {csv_path} has no rows to preload")
        return None

    image_windows = []
    for idx, image in enumerate(collection):
        end_dt = image.get_datetime()
        if end_dt is None:
            continue
        height_map = _get_height_map_for_selection(image)
        if height_map is None:
            continue
        scan_rate = image.get_scan_rate()
        if scan_rate in (None, 0):
            continue
        try:
            scan_rate = float(scan_rate)
        except (TypeError, ValueError):
            continue
        y_pixels, x_pixels = height_map.shape
        if x_pixels == 0 or y_pixels == 0:
            continue
        imaging_duration = y_pixels / scan_rate
        start_dt = end_dt - timedelta(seconds=imaging_duration)
        start_offset = (start_dt - depressurized_dt).total_seconds()
        end_offset = (end_dt - depressurized_dt).total_seconds()
        scan_size = image.get_scan_size()
        try:
            scan_size = float(scan_size)
        except (TypeError, ValueError):
            continue
        if scan_size == 0:
            continue
        pixel_size = scan_size / x_pixels
        x_coords = np.linspace(0, scan_size, x_pixels)

        image_windows.append({
            'index': idx,
            'image': image,
            'height_map': height_map,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'duration': imaging_duration,
            'pixel_size': pixel_size,
            'x_coords': x_coords,
            'x_pixels': x_pixels,
            'y_pixels': y_pixels,
        })

    if not image_windows:
        print("No image metadata available to preload selections")
        return None

    def _parse_index(val):
        if val is None:
            return None
        val = str(val).strip()
        if not val or val.lower() == 'none':
            return None
        try:
            return int(float(val))
        except ValueError:
            return None

    initial = {}
    matched_count = 0
    for row in rows:
        time_val = row.get('Time (minutes)')
        if time_val is None:
            continue
        time_str = str(time_val).strip()
        if not time_str:
            continue
        try:
            time_minutes = float(time_str)
        except ValueError:
            continue
        time_seconds = time_minutes * 60.0

        window = None
        for info in image_windows:
            start = info['start_offset']
            end = info['end_offset']
            if start <= time_seconds <= end:
                window = info
                break
        if window is None:
            tolerance = 0.5
            for info in image_windows:
                start = info['start_offset']
                end = info['end_offset']
                if abs(time_seconds - start) <= tolerance or abs(time_seconds - end) <= tolerance:
                    window = info
                    break
        if window is None:
            print(f"Could not match saved point at {time_minutes} minutes to any image window")
            continue

        idx = window['index']
        if idx in initial:
            print(f"Multiple saved entries matched image index {idx}; keeping the first")
            continue

        p1x = _parse_index(row.get('Point 1 X Pixel'))
        p1y = _parse_index(row.get('Point 1 Y Pixel'))
        p2x = _parse_index(row.get('Point 2 X Pixel'))
        p2y = _parse_index(row.get('Point 2 Y Pixel'))

        selected_slots = [None, None]
        for slot_idx, (px, py) in enumerate(((p1x, p1y), (p2x, p2y))):
            if px is None or py is None:
                continue
            if not (0 <= px < window['x_pixels'] and 0 <= py < window['y_pixels']):
                continue
            h_val = window['height_map'][py, px]
            if h_val is None or not np.isfinite(h_val):
                continue
            x_val = float(window['x_coords'][px])
            y_val = float((window['y_pixels'] - 1 - py) * window['pixel_size'])
            selected_slots[slot_idx] = (float(h_val), x_val, y_val, int(px), int(py))

        if all(slot is None for slot in selected_slots):
            continue

        time_offset = time_seconds - window['end_offset']
        duration = window['duration']
        if duration <= 0:
            time_offset = None
        else:
            upper = 0.0
            lower = -duration
            if time_offset < lower - 1e-3 or time_offset > upper + 1e-3:
                time_offset = max(lower, min(upper, time_offset))

        initial[idx] = {
            'selected_slots': selected_slots,
            'time_offset': time_offset,
        }
        matched_count += 1

    if matched_count:
        print(f"Preloaded {matched_count} saved selection(s) from {csv_path}")
    return initial if matched_count else None


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

initial_selections = _build_initial_selections(collection, pl.deflation_curve_path, depressurized_datetime)

# Open an interactive navigator so the user can pick images in any order
selections = collection.navigate_images(initial_selections=initial_selections)

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
            print(f"Overwriting existing file {file_path}")
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