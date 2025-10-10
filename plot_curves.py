import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import path_loader as pl

# Folder containing the CSV files
folder = pl.deflation_curves_path

# Find all CSV files in the folder for specified sample and depressurization date/time
csv_files = glob.glob(os.path.join(folder, f"deflation_curve_{pl.sample_number}_depressurized{pl.depressurized_day}_{pl.depressurized_time}*.csv"))

# Set up the plot
plt.figure(figsize=(8, 6))

# Define a color cycle
colors = ['black', 'blue', 'green', 'orange', 'red']

# Plot each CSV as a scatter plot
for idx, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    plt.scatter(df['Time (minutes)'], df['Deflection (nm)'],
                label=os.path.basename(csv_file), color=colors[idx % len(colors)], s=60, marker='x')

plt.xlabel('Time since depressurization (minutes)')
plt.ylabel('Deflection (nm)')
plt.legend()
plt.tight_layout()
plt.show()