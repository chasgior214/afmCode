import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
import re

# ============ INPUTS ============
sample_ID = '53'
start_datetime_local = datetime.strptime('2026-02-21 23:10:00', '%Y-%m-%d %H:%M:%S')
end_datetime_local = datetime.strptime('2026-02-21 23:59:59', '%Y-%m-%d %H:%M:%S')
# ================================

def find_csv_files(sample_id):
    """Find all CSV files for a given sample ID in both in_progress and finished directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(script_dir, "in_progress"),
        os.path.join(script_dir, "finished"),
        os.path.join(script_dir, "archived")
    ]
    
    pattern = f"sample{sample_id}_cell*_pressure_log_start*.csv"
    csv_files = []
    
    for directory in directories:
        csv_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return csv_files

def extract_file_start_datetime(filepath):
    """Extract the start datetime from a CSV filename."""
    filename = os.path.basename(filepath)
    match = re.search(r'start(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M-%S')
    return None

def load_and_filter_data(sample_id, start_dt, end_dt):
    """Load all relevant CSV files and filter data by datetime range."""
    csv_files = find_csv_files(sample_id)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for sample {sample_id}")
    
    print(f"Found {len(csv_files)} CSV file(s) for sample {sample_id}:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load and concatenate all dataframes
    dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamps to local datetime
    local_tz = datetime.now().astimezone().tzinfo
    df['datetime'] = pd.to_datetime(df['timestamp_unix'], unit='s', utc=True).dt.tz_convert(local_tz)
    
    # Make start_dt and end_dt timezone-aware for comparison
    start_dt_aware = pd.Timestamp(start_dt).tz_localize(local_tz)
    end_dt_aware = pd.Timestamp(end_dt).tz_localize(local_tz)
    
    # Filter by datetime range
    df_filtered = df[(df['datetime'] >= start_dt_aware) & (df['datetime'] <= end_dt_aware)]
    
    # Sort by datetime
    df_filtered = df_filtered.sort_values('datetime')
    
    return df_filtered

# Load and filter data
df_filtered = load_and_filter_data(sample_ID, start_datetime_local, end_datetime_local)

print(f"\nPlotting {len(df_filtered)} data points from {start_datetime_local} to {end_datetime_local}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['datetime'], df_filtered['avg_pressure_kPa_gage'])
plt.xlabel('Time')
plt.ylabel('avg_pressure_kPa_gage (kPa)')
plt.title(f'Sample {sample_ID} Pressure Log')
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# Export the filtered df with the timestamp column's first value subtracted from all values in the column
# df_filtered['timestamp_unix'] = df_filtered['timestamp_unix'] - df_filtered['timestamp_unix'].iloc[0]
# df_filtered.drop('datetime', axis=1, inplace=True)
# df_filtered.to_csv(f"sample_{sample_ID}_pressure_log_filtered.csv", index=False)