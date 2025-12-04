editing_mode = True

depressurized_date = '20251120'
depressurized_time = '14:47:12'
end_hour = None # None to not give end limit (if end_hour given and end_day not given, uses only day of depressurization up to this hour)
end_day = None # None to not give end limit (use all files in specified folder)
sample_number = '53'
transfer_location = 'o(5,1)' # $(6,3)
cavity_position = 'red'

# Additional depressurization date/time combinations can be added to this list to plot multiple datasets at once (see plot_curves.py)
plot_depressurizations = [
    # (depressurized_date, depressurized_time),
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
    # ('20251001', '14:31:11'),
    # ('20251028', '16:57:19'),
    ('20251126', '14:07:40'),
    ('20251201', '14:15:20'),
    ]


##############################################################################
import os
from datetime import datetime
depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
depressurized_datetime = datetime.strptime(depressurized_date + depressurized_time, '%Y%m%d%H%M%S')

plot_depressurizations = [(date, time.replace(':', '')) for date, time in plot_depressurizations]

experiment_data_path = 'C:\\Users\\chasg\\OneDrive - The University of Western Ontario\\MESc\\Research\\Experiments\\expt005_250403_gas_inflations'

excel_action_tracker_path = experiment_data_path + '\\action_tracker.xlsx'

afm_images_path = experiment_data_path + '\\raw_data\\'+depressurized_date+'\\FlattenedData'

deflation_curves_path = experiment_data_path + '\\data_processing\\deflation_curves'

def get_deflation_curve_path(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position):
    deflation_curve_filename = f'deflation_curve_sample{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
    return deflation_curves_path + f'\\{deflation_curve_filename}'

deflation_curve_path = get_deflation_curve_path(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position)

def get_deflation_curve_slope_id(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position):
    depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
    return f'slope_sample{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}'

deflation_curve_slope_path = experiment_data_path + '\\data_processing\\deflation_curve_slopes.csv'

def get_all_deflation_curve_paths():
    paths = []
    for file in os.listdir(deflation_curves_path):
        if file.endswith('.csv') and file.startswith('deflation_curve_'):
            paths.append(os.path.join(deflation_curves_path, file))
    return paths