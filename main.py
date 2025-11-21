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

"""TODO

make a 3D map of the image data with the paraboloid superimposed

when looking for the "substrate" (what I really mean is the graphene height outside the well), if it's bimodal/multimodal, can have it pick the one that's closest to the paraboloid vertex. Fixes edge cases where more of the image line off the well is either on the substrate or on a different step height of graphene

make right click zoom and left click selections? Then can do away with button to enter/exit zoom mode and auto entering zoom mode on startup. Right drag could be box zoom and right click to zoom to 4x4 um square centred on cursor

cross section shows dotted lines of the line above and below the selected line?

another line, extremum location relative to neutral piezo in um (take position relative to where the middle of the scan would be, and add offset)

could I do the fourier transform thing that the drift thing uses to align all my images? Could feed it all the images from a depressurization and it could match the wells together, then I could pinpoint centeres and do completely automated data extraction, could be amazing with sample53

Try using a denoising model to smooth images before paraboloid fitting?

Add filters to the list of images, could show only ones within a certain range of offsets to pick specific wells

Make it able to plot the deflection vs time points from multiple depressurizations on the same plot to compare them, potentially to use points from multiple depressurizations to get a better slope estimate

Have it make a map, then I select the regions that a well is in over the whole imaging session given drift, and it automatically shows me the same well over and over instead of navigating through images to find them (and could also have it automatically output a curve of using the highest point for that well, which I could compare to mine, and maybe if I get a denoising model to work well enough it could do basically everything automatically. Could also have it make timelapses of a 3d image of a single well changing over time given it would know how to center it)

Now that I'm saving pixel coordinates, try to have it read those back in and use them to get the deflection from the z-sensor data to compare to the height data. Can also make comparisons to curves made from max height within a few microns of the selected point (or the min of the 8 pixels surrounding it) vs the mode for the y value cross section that the max sits on. Could also have it make a map, I can pick a location, and it can give the curve from that location in all images containing it

Clean up buttons?

- Plot of deflections so far visible? In other window on second screen?

- Use mouse wheel for something
    - Zoom in/out? Centred on x,y of most recently selected point or centre of FOV or cursor? Maybe both, one active normally, another with shift + scroll, another with ctrl + scroll?
    - use horizontal scroll for something?
- Use left/right arrow keys for something
    - next/previous image in the collection or something else

- Display any other metadata I might want (drive amplitude/frequency, etc)
    - Maybe in another panel
    - Note that if I change some settings partway through an image, it tracks that in the metadata. Example, initial drive amplitude line looks like "DriveAmplitude: 0.089596", but later on if it was changed it would have a line that looks like "DriveAmplitude: 0.09@Line: 162"
    - Initial drive amplitude in for now, but make sure to have it take the one for the line later
    - Maybe imaging parameters on the right of the panel, separate from other data

- add a button "View 3D Heightmap", which renders a 3d heightmap in a new window. The heightmap should maintain the data's aspect ratio. Use the Viridis colour scale. Can select points on the 3d heightmap which get added as selected points in the main window.
"""

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

    def _parse_float(val):
        if val is None:
            return None
        val = str(val).strip()
        if not val or val.lower() == 'none':
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def _index_to_y_center(idx, y_pixels, pixel_size):
        if idx is None:
            return None
        idx = int(np.clip(idx, 0, y_pixels - 1))
        return (y_pixels - (idx + 0.5)) * pixel_size

    def _y_um_to_index(y_um, y_pixels, pixel_size):
        if y_um is None or pixel_size <= 0:
            return None
        idx_float = y_pixels - (y_um / pixel_size) - 0.5
        return int(np.clip(int(round(idx_float)), 0, y_pixels - 1))

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
        slot_sources = [None, None]
        for slot_idx, (px, py) in enumerate(((p1x, p1y), (p2x, p2y))):
            xyz_x = _parse_float(row.get(f'Point {slot_idx + 1} X (um)'))
            xyz_y = _parse_float(row.get(f'Point {slot_idx + 1} Y (um)'))
            xyz_z = _parse_float(row.get(f'Point {slot_idx + 1} Z (nm)'))
            if all(val is not None and np.isfinite(val) for val in (xyz_x, xyz_y, xyz_z)):
                try:
                    x_idx = int(np.argmin(np.abs(window['x_coords'] - xyz_x)))
                except ValueError:
                    x_idx = None
                y_idx = _y_um_to_index(xyz_y, window['y_pixels'], window['pixel_size'])
                selected_slots[slot_idx] = (
                    float(xyz_z),
                    float(xyz_x),
                    float(xyz_y),
                    x_idx,
                    y_idx,
                )
                slot_sources[slot_idx] = 'xyz'
                continue

            if px is None or py is None:
                continue
            if not (0 <= px < window['x_pixels'] and 0 <= py < window['y_pixels']):
                continue
            h_val = window['height_map'][py, px]
            if h_val is None or not np.isfinite(h_val):
                continue
            x_val = float(window['x_coords'][px])
            y_val = _index_to_y_center(py, window['y_pixels'], window['pixel_size'])
            selected_slots[slot_idx] = (float(h_val), x_val, y_val, int(px), int(py))
            slot_sources[slot_idx] = 'pixel'

        if all(slot is None for slot in selected_slots):
            continue

        warnings = []
        stored_deflection = _parse_float(row.get('Deflection (nm)'))
        if (
            stored_deflection is not None
            and selected_slots[0] is not None
            and selected_slots[1] is not None
        ):
            if 'pixel' in slot_sources:
                loaded_deflection = selected_slots[1][0] - selected_slots[0][0]
                if not np.isfinite(loaded_deflection) or not np.isfinite(stored_deflection):
                    pass
                else:
                    if abs(loaded_deflection - stored_deflection) > 1e-3:
                        warnings.append(
                            "Preloaded pixel coordinates reproduce a deflection of "
                            f"{loaded_deflection:.3f} nm instead of the saved "
                            f"{stored_deflection:.3f} nm."
                        )

        time_offset = time_seconds - window['end_offset']
        duration = window['duration']
        if duration <= 0:
            time_offset = None
        else:
            upper = 0.0
            lower = -duration
            if time_offset < lower - 1e-3 or time_offset > upper + 1e-3:
                time_offset = max(lower, min(upper, time_offset))

        entry = {
            'selected_slots': selected_slots,
            'time_offset': time_offset,
        }
        if warnings:
            entry['warning_messages'] = warnings
        initial[idx] = entry
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
physical_coords = []

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
            if slots[0] is not None and len(slots[0]) >= 5 and slots[0][3] is not None and slots[0][4] is not None:
                p1x, p1y = int(slots[0][3]), int(slots[0][4])
            if slots[1] is not None and len(slots[1]) >= 5 and slots[1][3] is not None and slots[1][4] is not None:
                p2x, p2y = int(slots[1][3]), int(slots[1][4])
            pixel_coords.append((p1x, p1y, p2x, p2y))

            p1x_um = p1y_um = p1z_nm = None
            p2x_um = p2y_um = p2z_nm = None
            if slots[0] is not None:
                p1z_nm = float(slots[0][0])
                if len(slots[0]) >= 3:
                    p1x_um = float(slots[0][1]) if slots[0][1] is not None else None
                    p1y_um = float(slots[0][2]) if slots[0][2] is not None else None
            if slots[1] is not None:
                p2z_nm = float(slots[1][0])
                if len(slots[1]) >= 3:
                    p2x_um = float(slots[1][1]) if slots[1][1] is not None else None
                    p2y_um = float(slots[1][2]) if slots[1][2] is not None else None
            physical_coords.append((p1x_um, p1y_um, p1z_nm, p2x_um, p2y_um, p2z_nm))
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
        dir_path = pl.deflation_curves_path
        file_path = pl.deflation_curve_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if os.path.exists(file_path):
            print(f"Overwriting existing file {file_path}")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Time (minutes)',
                'Deflection (nm)',
                'Point 1 X Pixel',
                'Point 1 Y Pixel',
                'Point 1 X (um)',
                'Point 1 Y (um)',
                'Point 1 Z (nm)',
                'Point 2 X Pixel',
                'Point 2 Y Pixel',
                'Point 2 X (um)',
                'Point 2 Y (um)',
                'Point 2 Z (nm)'
            ])
            for (t, d, (p1x, p1y, p2x, p2y),
                 (p1x_um, p1y_um, p1z_nm, p2x_um, p2y_um, p2z_nm)) in zip(times, deflections, pixel_coords, physical_coords):
                writer.writerow([
                    t,
                    d,
                    p1x,
                    p1y,
                    p1x_um,
                    p1y_um,
                    p1z_nm,
                    p2x,
                    p2y,
                    p2x_um,
                    p2y_um,
                    p2z_nm,
                ])

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