depressurized_date = '20251016'
depressurized_time = '15:03:31'
sample_number = '37'
transfer_location = '$(6,3)'
cavity_position = 'green'



##############################################################################
from datetime import datetime
depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
depressurized_datetime = datetime.strptime(depressurized_date + depressurized_time, '%Y%m%d%H%M%S')

experiment_data_path = 'C:\\Users\\chasg\\OneDrive - The University of Western Ontario\\MESc\\Research\\Experiments\\expt005_250403_gas_inflations'

afm_images_path = experiment_data_path + '\\raw_data\\'+depressurized_date+'\\FlattenedData'

deflation_curves_path = experiment_data_path + '\\data_processing\\deflation_curves'
deflation_curve_filename = f'deflation_curve_sample{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
deflation_curve_path = deflation_curves_path + f'\\{deflation_curve_filename}'

def get_deflation_curve_slope_id(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position):
    depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format
    return f'slope_sample{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}'

deflation_curve_slope_path = experiment_data_path + '\\data_processing\\deflation_curve_slopes.csv'
deflation_curve_slope_id = get_deflation_curve_slope_id(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position)

def get_all_deflation_curve_paths():
    import os
    paths = []
    for file in os.listdir(deflation_curves_path):
        if file.endswith('.csv') and file.startswith('deflation_curve_'):
            paths.append(os.path.join(deflation_curves_path, file))
    return paths