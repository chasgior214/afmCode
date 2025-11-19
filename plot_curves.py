import os
import glob
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt

import path_loader as pl

# List of substrings to filter filenames. Empty list = no filtering.
filter_substrings = [
    'blue'
]


def _load_saved_slopes():
    """Return a mapping of deflation curve slope IDs to slope/intercept values."""
    slope_file = pl.deflation_curve_slope_path
    if not os.path.exists(slope_file):
        return {}

    slopes = {}
    with open(slope_file, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            slope_id = row.get('id')
            if not slope_id:
                continue
            slope_val = row.get('slope_nm_per_min')
            intercept_val = row.get('y_intercept_nm')
            try:
                slope_float = float(slope_val)
                intercept_float = float(intercept_val)
            except (TypeError, ValueError):
                continue
            slopes[slope_id] = (slope_float, intercept_float)
    return slopes


def _get_slope_id_from_filename(csv_path):
    """Attempt to infer the slope ID for a CSV file based on its filename."""
    filename = os.path.basename(csv_path)
    pattern = (
        r"deflation_curve_sample(?P<sample>[^_]+)"
        r"_depressurized(?P<date>\d{8})_(?P<time>\d+)"
        r"_loc(?P<loc>.+?)_cav(?P<cav>.+?)\.csv$"
    )
    match = re.match(pattern, filename)
    if not match:
        return None
    return pl.get_deflation_curve_slope_id(
        match.group('sample'),
        match.group('date'),
        match.group('time'),
        match.group('loc'),
        match.group('cav'),
    )

# Folder containing the CSV files
folder = pl.deflation_curves_path


def _get_depressurization_targets():
    """Return a list of (date, time) pairs to be plotted."""
    user_specified = getattr(pl, 'plot_depressurizations', None)
    targets = []

    if user_specified:
        for entry in user_specified:
            date = time = None
            if isinstance(entry, dict):
                date = entry.get('depressurized_date') or entry.get('date')
                time = entry.get('depressurized_time') or entry.get('time')
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                date, time = entry[0], entry[1]
            elif isinstance(entry, str):
                # Support strings formatted as 'YYYYMMDD_HHMMSS'
                parts = entry.split('_', 1)
                if len(parts) == 2:
                    date, time = parts
            if date and time:
                clean_time = str(time).replace(':', '')
                targets.append((str(date), clean_time))
    if not targets:
        targets.append((pl.depressurized_date, pl.depressurized_time))
    return targets


depressurization_targets = _get_depressurization_targets()

# Find all CSV files in the folder for specified sample and depressurization date/time
csv_entries = []
for target_idx, (date, time) in enumerate(depressurization_targets):
    pattern = f"deflation_curve_sample{pl.sample_number}_depressurized{date}_{time}*.csv"
    for csv_file in glob.glob(os.path.join(folder, pattern)):
        basename = os.path.basename(csv_file)
        if filter_substrings:
            # case-insensitive substring match: include file if any substring matches
            lname = basename.lower()
            if not any(sub.lower() in lname for sub in filter_substrings):
                continue
        csv_entries.append({'path': csv_file, 'target_idx': target_idx})

# Set up the plot
plt.figure(figsize=(8, 6))

# If the files all contain either 'red', 'blue', 'green', 'orange', or 'black' in their filenames, use those colors for those files
file_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 'orange': 'orange', 'black': 'black'}
csv_paths = [entry['path'] for entry in csv_entries]

if csv_paths and all(any(color in os.path.basename(f) for color in file_colors) for f in csv_paths):
    colors = [next(color for color in file_colors if color in os.path.basename(f)) for f in csv_paths]
else:
    # Otherwise, use a default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load saved slopes once so they can be reused for each curve
saved_slopes = _load_saved_slopes()

# collect all scatter points so axes can be set from points only (ignore slope lines)
_all_x_vals = []
_all_y_vals = []

# Plot each CSV as a scatter plot
markers = ['x', '*', '+', 'o', 's', '^', '>', 'd']

for idx, entry in enumerate(csv_entries):
    csv_file = entry['path']
    df = pd.read_csv(csv_file)
    marker = markers[entry['target_idx'] % len(markers)]
    plt.scatter(df['Time (minutes)'], df['Deflection (nm)'],
                label=os.path.basename(csv_file), color=colors[idx % len(colors)], s=60, marker=marker)

    slope_id = _get_slope_id_from_filename(csv_file)
    if slope_id and slope_id in saved_slopes:
        slope, intercept = saved_slopes[slope_id]
        if not df.empty:
            x_vals = df['Time (minutes)']
            x_min, x_max = x_vals.min(), x_vals.max()
            line_x = [x_min, x_max]
            line_y = [slope * x_min + intercept, slope * x_max + intercept]
            plt.plot(line_x, line_y, linestyle=':', color=colors[idx % len(colors)], linewidth=1.5)
    # accumulate point coordinates for later axis-lim calculation
    if 'Time (minutes)' in df and 'Deflection (nm)' in df and not df.empty:
        _all_x_vals.extend(df['Time (minutes)'].tolist())
        _all_y_vals.extend(df['Deflection (nm)'].tolist())

# Determine axis limits from points only (so slope lines don't affect autoscaling)
if _all_x_vals and _all_y_vals:
    x_min = min(_all_x_vals)
    x_max = max(_all_x_vals)
    y_min = min(_all_y_vals)
    y_max = max(_all_y_vals)

    # small padding (5%) or a sensible default when range is zero
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
else:
    # no data points found; keep previous behavior
    plt.xlim(left=0)

plt.xlabel('Time since depressurization (minutes)')
plt.ylabel('Deflection (nm)')
plt.legend()
plt.tight_layout()
plt.show()