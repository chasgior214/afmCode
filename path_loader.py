depressurized_date = '20250918'
depressurized_time = '15:03:22'
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
deflation_curve_filename = f'deflation_curve_{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
deflation_curve_path = deflation_curves_path + f'\\{deflation_curve_filename}'

deflation_curve_slope_path = experiment_data_path + '\\data_processing\\deflation_curve_slopes.csv'
deflation_curve_slope_id = f'slope_{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}'