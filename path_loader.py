editing_mode = True

depressurized_date = '20251215'
depressurized_time = '16:23:22'
end_hour = None # None to not give end limit (if end_hour given and end_day not given, uses only day of depressurization up to this hour)
end_day = None # None to not give end limit (use all files in specified folder)
sample_ID = '53'
transfer_location = 'o(5,1)' # 'o(3,8)'
if sample_ID == '37': transfer_location = '$(6,3)'
cavity_position = 'right'

# Additional depressurization date/time combinations can be added to this list to plot multiple datasets at once (see plot_curves.py)
plot_depressurizations = [
    (depressurized_date, depressurized_time), # Plotted by default when no others specified

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
    # End of red/blue H2


    ]

##############################################################################
import os
import csv
import re
from datetime import datetime
depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
depressurized_datetime = datetime.strptime(depressurized_date + depressurized_time, '%Y%m%d%H%M%S')

plot_depressurizations = [(date, time.replace(':', '')) for date, time in plot_depressurizations]

experiment_data_path = 'C:\\Users\\chasg\\OneDrive - The University of Western Ontario\\MESc\\Research\\Experiments\\expt005_250403_gas_inflations'

excel_action_tracker_path = experiment_data_path + '\\action_tracker.xlsx'

afm_images_path = experiment_data_path + '\\raw_data\\'+depressurized_date+'\\FlattenedData'

deflation_curves_path = experiment_data_path + '\\data_processing\\deflation_curves'

def get_deflation_curve_path(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position):
    deflation_curve_filename = f'deflation_curve_sample{sample_ID}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
    return deflation_curves_path + f'\\{deflation_curve_filename}'

deflation_curve_path = get_deflation_curve_path(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position)

def get_deflation_curve_slope_id(sample_ID, depressurized_date, depressurized_time, transfer_location, cavity_position):
    depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
    return f'slope_sample{sample_ID}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}'

def get_slope_id_from_filename(csv_path):
    """Infer the slope ID for a CSV file based on its filename."""
    filename = os.path.basename(csv_path)
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

deflation_curve_slope_path = experiment_data_path + '\\data_processing\\deflation_curve_slopes.csv'

def get_all_deflation_curve_paths():
    paths = []
    for file in os.listdir(deflation_curves_path):
        if file.endswith('.csv') and file.startswith('deflation_curve_'):
            paths.append(os.path.join(deflation_curves_path, file))
    return paths

def load_saved_slopes_intercepts():
    """Return a mapping of deflation curve slope IDs to slope/intercept values."""
    slope_file = deflation_curve_slope_path
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