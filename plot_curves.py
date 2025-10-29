import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import path_loader as pl

# Folder containing the CSV files
folder = pl.deflation_curves_path

# Find all CSV files in the folder for specified sample and depressurization date/time
csv_files = glob.glob(os.path.join(folder, f"deflation_curve_sample{pl.sample_number}_depressurized{pl.depressurized_date}_{pl.depressurized_time}*.csv"))

# filter to only files containing certain colours in their filenames
# csv_files = [f for f in csv_files if any(color in os.path.basename(f) for color in ['green', 'black', 'orange'])]

# Set up the plot
plt.figure(figsize=(8, 6))

# If the files all contain either 'red', 'blue', 'green', 'orange', or 'black' in their filenames, use those colors for those files
file_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 'orange': 'orange', 'black': 'black'}
if all(any(color in os.path.basename(f) for color in file_colors) for f in csv_files):
    colors = [next(color for color in file_colors if color in os.path.basename(f)) for f in csv_files]
else:
    # Otherwise, use a default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot each CSV as a scatter plot
for idx, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    plt.scatter(df['Time (minutes)'], df['Deflection (nm)'],
                label=os.path.basename(csv_file), color=colors[idx % len(colors)], s=60, marker='x')

plt.xlim(left=0)
plt.xlabel('Time since depressurization (minutes)')
plt.ylabel('Deflection (nm)')
# plt.legend()
plt.tight_layout()
plt.show()