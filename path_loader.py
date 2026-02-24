editing_mode = True

depressurized_date = '20260221'
depressurized_time = '22:11:57'
end_hour = None # None to not give end limit (if end_hour given and end_day not given, uses only day of depressurization up to this hour)
end_day = None # None to not give end limit (use all files in specified folder)
sample_ID = '53'
transfer_location = 'o(5,1)'
if sample_ID == '37': transfer_location = '$(6,3)'
if sample_ID == '43': transfer_location = 'o(3,8)'
cavity_position = 'red'

# Additional depressurization date/time combinations can be added to this list to plot multiple datasets at once (see plot_curves.py)
plot_depressurizations = [
    # (depressurized_date, depressurized_time), # Plotted by default when no others specified

    # sample53 calibration curves
    ('20260117', '13:51:30'),
    ('20260221', '22:11:57'),

    # SF6 Inflations
    # ('20251227', '16:12:52'),
    # ('20260103', '14:24:35'),
    # ('20260108', '21:18:54'),
    # ('20260117', '12:59:23'),
    # ('20260203', '18:36:23'),
    # ('20260221', '21:03:55'),

    # CO2 curves
    # ('20251008', '14:56:36'),
    # ('20251014', '16:30:34'),
    # ('20251223', '16:26:48'),
    # ('20251223', '17:28:34'),
    # ('20251223', '19:32:13'),
    # ('20251223', '22:00:19'),
    # ('20251223', '22:55:36'),

    # He curves
    # ('20251007', '11:29:29'),
    # ('20251010', '19:32:20'),
    # ('20251010', '20:59:18'),
    # ('20251010', '22:23:34'),
    # ('20251010', '23:55:25'),
    # ('20251223', '23:47:42'),
    # ('20251224', '00:22:01'),
    # ('20251224', '00:55:54'),
    # ('20251224', '02:43:31'),

    # Red/blue H2
    # ('20250729', '17:18:16'),
    # ('20250918', '15:03:22'),
    # ('20250919', '15:45:32'),
    # ('20250929', '14:52:14'),
    # ('20251006', '14:09:05'),
    # ('20251007', '16:05:25'),
    # ('20251014', '13:15:45'),
    # ('20251017', '15:54:29'),
    # ('20251031', '13:47:46'),
    # ('20251105', '19:36:49'),
    # ('20251117', '15:24:00'),
    # ('20251119', '21:32:46'),
    # ('20251204', '16:01:37'),
    # ('20251205', '15:38:48'),
    # ('20251206', '19:35:33'),
    # ('20251210', '19:23:49'),
    # ('20251223', '13:54:19'),
    # End of red/blue H2


    ]

##############################################################################
import csv
import re
from datetime import datetime
from pathlib import Path

depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
depressurized_datetime = datetime.strptime(depressurized_date + depressurized_time, '%Y%m%d%H%M%S')

plot_depressurizations = [(date, time.replace(':', '')) for date, time in plot_depressurizations]

experiment_data_path = Path(r'C:\Users\chasg\OneDrive - The University of Western Ontario\MESc\Research\Experiments\expt005_250403_gas_inflations')

# raw data
excel_action_tracker_path = experiment_data_path / 'action_tracker.xlsx'

afm_images_path = experiment_data_path / 'raw_data' / depressurized_date / 'FlattenedData'

pressure_logs_path = experiment_data_path / 'raw_data' / 'pressure_logs'
# add function(s) to point to specific pressure log files

# well maps
well_maps_path = experiment_data_path / 'well_maps'

# processed data - deflation curves
deflation_curves_path = experiment_data_path / 'data_processing' / 'deflation_curves'

def get_deflation_curve_path(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position):
    deflation_curve_filename = f'deflation_curve_sample{sample_ID}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
    return deflation_curves_path / deflation_curve_filename

deflation_curve_path = get_deflation_curve_path(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position)

def get_all_deflation_curve_paths():
    paths = []
    for file_path in deflation_curves_path.glob('deflation_curve_*.csv'):
        paths.append(file_path)
    return paths

# processed data - deflation curve slopes
deflation_curve_slope_path = experiment_data_path / 'data_processing' / 'deflation_curve_slopes.csv'

def get_deflation_curve_slope_id(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position):
    depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
    return f'slope_sample{sample_ID}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}'

def get_slope_id_from_filename(csv_path):
    """Infer the slope ID for a CSV file based on its filename."""
    csv_path = Path(csv_path)
    filename = csv_path.name
    pattern = (
        r"deflation_curve_sample(?P<sample>[^_]+)"
        r"_depressurized(?P<date>\d{8})_(?P<time>\d+)"
        r"_loc(?P<loc>.+?)_cav(?P<cav>.+?)\.csv$"
    )
    match = re.match(pattern, filename)
    if not match:
        return None
    return get_deflation_curve_slope_id(
        match.group('sample'),
        match.group('date'),
        match.group('time'),
        match.group('loc'),
        match.group('cav'),
    )

def load_saved_slopes_intercepts():
    """Return a mapping of deflation curve slope IDs to slope/intercept values."""
    slope_file = deflation_curve_slope_path
    if not slope_file.exists():
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