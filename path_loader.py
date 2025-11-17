depressurized_date = '20251109'
depressurized_time = '18:58:36'
sample_number = '37'
transfer_location = '$(6,3)'
cavity_position = 'orange'

# Additional depressurization date/time combinations can be added to this list to plot multiple datasets at once (see plot_curves.py)
plot_depressurizations = [
    ('20251109', '18:58:36'),
    ('20251106', '17:00:36')
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