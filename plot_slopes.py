import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import path_loader as pl
import gas_constants as gc
deflation_curve_slope_path = pl.deflation_curve_slope_path

overall_slopes = {
    'H2': {
        'green': 2.1,
        'red': 44.4,
        'blue': 28,
        'orange': 2.2,
        'black': 2.3
    },
    'He': {
        'green': 9,
        'red': 25.1,
        'blue': 39.7,
        'orange': 8.8,
        'black': 8.6
    },
    'CO2': {
        'green': 0.86,
        'orange': 0.67,
        'black': 0.83
    },
    'Ar': {
        'red': 0.95,
        'blue': 2.7,
        'green': 0.088,
        'orange': 0.059,
        'black': 0.080
    },
    'CH4': {
        'red': 1.32,
        'blue': 3.56,
        'green': 0.036,
        'black': 0.033,
        'orange': 0.021
    },
    'N2': {
        'green': 0.026,
        'red': 1.35,
        'blue': 1.51,
        'orange': 0.019,
        'black': 0.017
    },
    'C2H4': {
        'red': 3.42,
        'blue': 7.52,
        'green': 0.049,
        'black': 0.043,
        'orange': 0.036
    },
    'C3H8': {
        'blue': 1.03,
        'red': 0.047
    }
}

def plot_recent_deflation_curve_slopes():
    # --- define which depressurizations to plot ---

    to_plot = {
    #     '29-Jul	17:18:16': 'H2',
    #     '18-Sep	15:03:22': 'H2',
    #     '19-Sep	15:45:32': 'H2',
    #     '25-Sep	12:27:29': 'CH4',
    #     '29-Sep	14:52:14': 'H2',
    #     '01-Oct	14:31:11': 'N2',
    #     '06-Oct	14:09:05': 'H2',
    #     '07-Oct	11:29:29': 'He',
    #     '07-Oct	16:05:25': 'H2',
    #     '08-Oct	14:56:36': 'CO2',
    # # }
    #     '10-Oct	16:00:39': 'H2',
    #     '10-Oct	19:32:20': 'He',
    #     '10-Oct	20:59:18': 'He',
    #     '10-Oct	22:23:34': 'He',
    #     '10-Oct	23:55:25': 'He',
    #     '14-Oct	13:15:45': 'H2',
    #     '15-Oct	14:41:10': 'H2',
    #     '16-Oct	15:03:31': 'Ar',
    #     '17-Oct	15:54:29': 'H2',
    # }
        # '23-Oct	13:41:06': 'CH4',
        # '27-Oct	19:57:29': 'H2',
        # '28-Oct	00:36:28': 'He',
        # '28-Oct	16:57:19': 'N2',
    # }
        '30-Oct	15:26:29': 'H2',
        '31-Oct	13:47:46': 'H2',
        '02-Nov	21:31:34': 'C2H4',
        '05-Nov	19:36:49': 'H2',
        '06-Nov	17:00:36': 'C3H8',
    }

    # convert date/time to YYYYMMDD_HHMMSS format, including replacing the 3 character month with 2 digit month
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

    # take absolute value of slope data
    sdf['slope_nm_per_min'] = sdf['slope_nm_per_min'].abs()

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

    # if the gas/colour combination exists in the overall_slopes dictionary, plot a dotted horizontal line at that slope value that extends twice the width of the markers
    # compute marker width in data coordinates so horizontal lines are twice that width
    fig = plt.gcf()
    ax = plt.gca()
    marker_s = 100  # same as used for scatter
    marker_diameter_pts = np.sqrt(marker_s)
    marker_diameter_px = marker_diameter_pts * fig.dpi / 72.0
    desired_line_px = 2 * marker_diameter_px  # twice the marker width in pixels

    # convert pixel length to data coordinates (horizontal)
    disp0 = ax.transData.transform((0, 0))
    disp1 = disp0 + np.array([desired_line_px, 0])
    data0 = ax.transData.inverted().transform(disp0)
    data1 = ax.transData.inverted().transform(disp1)
    desired_line_dx = data1[0] - data0[0]

    # if the gas/colour combination isn't a first time, plot a horizontal line at the slope value in overall_slopes
    for t in unique_times:
        gas = gas_per_dt[t]
        xcenter = x_pos_map[t]
        for color in ['red', 'blue', 'green', 'orange', 'black']:
            if color in overall_slopes.get(gas, {}):
                slope_value = overall_slopes[gas][color]
                xmin = xcenter - desired_line_dx / 2.0
                xmax = xcenter + desired_line_dx / 2.0
                ax.hlines(y=slope_value, xmin=xmin, xmax=xmax, colors=color, linestyles='dotted', alpha=0.5)
    
    # in the legend, show a dotted line with the label "Consensus Slope" for the horizontal lines
    plt.plot([], [], color='black', linestyle='dotted', label='Consensus Slope')
    plt.legend()

    plt.xticks(range(len(unique_times)), x_labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.xlabel("Depressurization Date — Gas")
    plt.ylabel("Initial Slope (nm/min)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

plot_recent_deflation_curve_slopes()

color_to_marker = {
    'black': 'x',
    'green': '+',
    'orange': 'o',
    'blue': '*',
    'red': '^'
}
unfilled_colors = {'orange', 'red'}

# plot the slopes versus kinetic diameters. Colour the points by the keys of the slope dictionaries. Make the orange and red point markers unfilled.
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('CO2', overall_slopes['CO2']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4']), ('N2', overall_slopes['N2']), ('C2H4', overall_slopes['C2H4']), ('C3H8', overall_slopes['C3H8'])]:
    kd = gc.kinetic_diameters[gas]
    for color, slope in slopes.items():
        marker = color_to_marker[color]
        if color in unfilled_colors:
            plt.scatter(kd, slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
        else:
            plt.scatter(kd, slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
plt.yscale('log')

# put labels for each gas kinematic diameters on the x axis
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('CO2', overall_slopes['CO2']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4']), ('N2', overall_slopes['N2']), ('C2H4', overall_slopes['C2H4']), ('C3H8', overall_slopes['C3H8'])]:
    kd = gc.kinetic_diameters[gas]
    plt.axvline(x=kd, color='gray', linestyle='--', linewidth=0.5)
    plt.text(kd, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
plt.xlabel('Kinetic Diameter (Å)')
plt.ylabel('Deflation Curve Slope (nm/min)')
plt.title('Deflation Curve Slopes vs Gas Kinetic Diameter')
plt.grid(True)
plt.show()

# do the same but plot slopes versus molecular weights
plt.figure()
delta_for_close_mws = 0.4
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('CO2', overall_slopes['CO2']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4']), ('N2', overall_slopes['N2']), ('C2H4', overall_slopes['C2H4']), ('C3H8', overall_slopes['C3H8'])]:
    mw = gc.molecular_weights[gas]
    if gas in ['N2', 'CO2']:
        mw -= delta_for_close_mws
    if gas in ['C2H4', 'C3H8']:
        mw += delta_for_close_mws
    for color, slope in slopes.items():
        marker = color_to_marker[color]
        if color in unfilled_colors:
            plt.scatter(mw, slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
        else:
            plt.scatter(mw, slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
plt.yscale('log')

# put labels for each gas molecular weights on the x axis
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4'])]:
    mw = gc.molecular_weights[gas]
    plt.axvline(x=mw, color='gray', linestyle='--', linewidth=0.5)
    plt.text(mw, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
# add a combined label for N2/C2H4, and another for C3H8/CO2
plt.axvline(x=(gc.molecular_weights['N2'] + gc.molecular_weights['C2H4'])/2, color='gray', linestyle='--', linewidth=0.5)
plt.text((gc.molecular_weights['N2'] + gc.molecular_weights['C2H4'])/2, plt.ylim()[0], 'N2/C2H4', verticalalignment='bottom', horizontalalignment='right')
plt.axvline(x=(gc.molecular_weights['C3H8'] + gc.molecular_weights['CO2'])/2, color='gray', linestyle='--', linewidth=0.5)
plt.text((gc.molecular_weights['C3H8'] + gc.molecular_weights['CO2'])/2, plt.ylim()[0], 'CO2/C3H8', verticalalignment='bottom', horizontalalignment='right')
plt.xlabel('Molecular Weight (amu)')
plt.xlim(left=0)
plt.ylabel('Deflation Curve Slope (nm/min)')
plt.title('Deflation Curve Slopes vs Gas Molecular Weight')
plt.grid(True)
plt.show()

# make a plot with diameter on the x axis and slope normalized by multiplying by the square root of (molar mass * 2 * π * R * T) on the y axis
plt.figure()
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('CO2', overall_slopes['CO2']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4']), ('N2', overall_slopes['N2']), ('C2H4', overall_slopes['C2H4']), ('C3H8', overall_slopes['C3H8'])]:
    kd = gc.kinetic_diameters[gas]
    mm = gc.molar_masses_kg_per_mol[gas]
    for color, slope in slopes.items():
        norm_slope = slope * ((mm * 2 * np.pi * gc.R * gc.T) ** 0.5)
        marker = color_to_marker[color]
        if color in unfilled_colors:
            plt.scatter(kd, norm_slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
        else:
            plt.scatter(kd, norm_slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
plt.yscale('log')

# put labels for each gas kinematic diameters on the x axis
for gas, slopes in [('H2', overall_slopes['H2']), ('He', overall_slopes['He']), ('CO2', overall_slopes['CO2']), ('Ar', overall_slopes['Ar']), ('CH4', overall_slopes['CH4']), ('N2', overall_slopes['N2']), ('C2H4', overall_slopes['C2H4']), ('C3H8', overall_slopes['C3H8'])]:
    kd = gc.kinetic_diameters[gas]
    plt.axvline(x=kd, color='gray', linestyle='--', linewidth=0.5)
    plt.text(kd, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
plt.xlabel('Kinetic Diameter (Å)')
plt.ylabel('Normalized Deflation Curve Slope (nm/min * sqrt(kg * J)/mol)')
plt.title('Normalized Deflation Curve Slopes vs Gas Kinetic Diameter')
plt.grid(True)
plt.show()