to_plot = {
    '29-Jul	17:18:16': 'H2',
    '18-Sep	15:03:22': 'H2',
    '19-Sep	15:45:32': 'H2',
    '25-Sep	12:27:29': 'CH4',
    '29-Sep	14:52:14': 'H2',
    '01-Oct	14:31:11': 'N2',
    '06-Oct	14:09:05': 'H2',
    '07-Oct	11:29:29': 'He',
    '07-Oct	16:05:25': 'H2',
    '08-Oct	14:56:36': 'CO2',
# }

# to_plot = {
    '10-Oct	16:00:39': 'H2',
    '10-Oct	19:32:20': 'He',
    '10-Oct	20:59:18': 'He',
    '10-Oct	22:23:34': 'He',
    '10-Oct	23:55:25': 'He',
    '14-Oct	13:15:45': 'H2',
    '15-Oct	14:41:10': 'H2',
    '16-Oct	15:03:31': 'Ar',
    '17-Oct	15:54:29': 'H2'
}

import matplotlib.pyplot as plt
import path_loader as pl
deflation_curve_slope_path = pl.deflation_curve_slope_path

# convert date/time to YYYYMMDD_HHMMSS format, including replacing the 3 character month with 2 digit month
from datetime import datetime

def convert_to_YYYYMMDD_HHMMSS(date_str, time_str):
    # Parse the date and time
    dt = datetime.strptime(f"{date_str} {time_str}", "%d-%b %H:%M:%S")
    # Format as YYYYMMDD_HHMMSS, put 2025 as the year
    dt = dt.replace(year=2025)
    return dt.strftime("%Y%m%d_%H%M%S")

to_plot_converted = {}
for date_time_str, gas in to_plot.items():
    date_str, time_str = date_time_str.split('\t')
    converted_str = convert_to_YYYYMMDD_HHMMSS(date_str, time_str)
    to_plot_converted[converted_str] = gas

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- build potential_ids AND remember their colors ---
id_to_meta = {}  # id -> {"color": str, "dtstr": "YYYYMMDD_HHMMSS", "gas": str}
potential_ids = []

for dt_str, gas in to_plot_converted.items():
    date8 = dt_str[:8]
    time6 = dt_str[9:]
    for colour in ['red', 'blue', 'green', 'orange', 'black']:
        _id = pl.get_deflation_curve_slope_id(pl.sample_number, date8, time6, pl.transfer_location, colour)
        potential_ids.append(_id)
        id_to_meta[_id] = {"color": colour, "dtstr": dt_str, "gas": gas}

# --- load and filter data ---
slope_data = pd.read_csv(deflation_curve_slope_path)
sdf = slope_data[slope_data['id'].isin(potential_ids)].copy()

if sdf.empty:
    raise ValueError("No matching IDs found in slope_data for the requested timestamps/positions.")

# --- attach plotting metadata (color, gas, datetime) from our mapping ---
def parse_dt(dtstr: str) -> datetime:
    # dtstr is "YYYYMMDD_HHMMSS"
    return datetime.strptime(dtstr, "%Y%m%d_%H%M%S")

sdf["color"] = sdf["id"].map(lambda _id: id_to_meta.get(_id, {}).get("color", "black"))
sdf["gas"]   = sdf["id"].map(lambda _id: id_to_meta.get(_id, {}).get("gas", "Unknown"))
sdf["dtstr"] = sdf["id"].map(lambda _id: id_to_meta.get(_id, {}).get("dtstr", None))
sdf["dt"]    = sdf["dtstr"].map(lambda s: parse_dt(s) if isinstance(s, str) else pd.NaT)

# keep only rows we can place on the x-axis
sdf = sdf.dropna(subset=["dt"])

from datetime import datetime

# --- group by unique depressurization times (categorical x-axis) ---
sdf = sdf.sort_values("dt")  # ensure chronological order
unique_times = sdf["dtstr"].unique()
x_pos_map = {t: i for i, t in enumerate(unique_times)}

plt.figure(figsize=(10, 6))

# plot all points, same depressurization -> same x position (no annotations)
color_to_marker = {
    'black': 'x',
    'green': '+',
    'orange': 'o',
    'blue': '*',
    'red': '^'
}
unfilled_colors = {'orange', 'red'}

for _, row in sdf.iterrows():
    x = x_pos_map[row["dtstr"]]
    col = row["color"]
    marker = color_to_marker.get(col, 'x')
    if col in unfilled_colors:
        face = 'none'
        edge = col
    else:
        face = col
        edge = col
    plt.scatter(x, row["slope_nm_per_min"], s=100, marker=marker, facecolors=face, edgecolors=edge)

# build labels: "Mon D — GAS" (e.g., "Oct 2 — H2")
# get one gas per dtstr
gas_per_dt = sdf.drop_duplicates("dtstr").set_index("dtstr")["gas"]

def fmt_mon_day(dtstr):
    dt = datetime.strptime(dtstr, "%Y%m%d_%H%M%S")
    # Windows doesn't support %-d; fall back to %d and strip leading zero
    try:
        return dt.strftime("%b %-d")
    except ValueError:
        return dt.strftime("%b %d").replace(" 0", " ")

x_labels = [f"{fmt_mon_day(t)} — {gas_per_dt[t]}" for t in unique_times]

plt.xticks(range(len(unique_times)), x_labels, rotation=45, ha="right")
plt.xlabel("Depressurization Date — Gas")
plt.ylabel("Initial Slope (nm/min)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
