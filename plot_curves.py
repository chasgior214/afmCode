import os
import glob
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt

import path_loader as pl


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

# Find all CSV files in the folder for specified sample and depressurization date/time
csv_files = glob.glob(os.path.join(folder, f"deflation_curve_sample{pl.sample_number}_depressurized{pl.depressurized_date}_{pl.depressurized_time}*.csv"))

# filter to only files containing certain colours in their filenames
# csv_files = [f for f in csv_files if any(color in os.path.basename(f) for color in ['blue', 'red'])]

# Set up the plot
plt.figure(figsize=(8, 6))

# If the files all contain either 'red', 'blue', 'green', 'orange', or 'black' in their filenames, use those colors for those files
file_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 'orange': 'orange', 'black': 'black'}
if all(any(color in os.path.basename(f) for color in file_colors) for f in csv_files):
    colors = [next(color for color in file_colors if color in os.path.basename(f)) for f in csv_files]
else:
    # Otherwise, use a default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load saved slopes once so they can be reused for each curve
saved_slopes = _load_saved_slopes()

# Plot each CSV as a scatter plot
for idx, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    plt.scatter(df['Time (minutes)'], df['Deflection (nm)'],
                label=os.path.basename(csv_file), color=colors[idx % len(colors)], s=60, marker='x')

    slope_id = _get_slope_id_from_filename(csv_file)
    if slope_id and slope_id in saved_slopes:
        slope, intercept = saved_slopes[slope_id]
        if not df.empty:
            x_vals = df['Time (minutes)']
            x_min, x_max = x_vals.min(), x_vals.max()
            line_x = [x_min, x_max]
            line_y = [slope * x_min + intercept, slope * x_max + intercept]
            plt.plot(line_x, line_y, linestyle=':', color=colors[idx % len(colors)], linewidth=1.5)

plt.xlim(left=0)
plt.xlabel('Time since depressurization (minutes)')
plt.ylabel('Deflection (nm)')
# plt.legend()
plt.tight_layout()
plt.show()