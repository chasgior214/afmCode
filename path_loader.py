depressurized_day = '20251001'
depressurized_time = '14:31:11'
sample_number = '37'
transfer_location = '$(6,3)'
cavity_position = 'blue'



##############################################################################
depressurized_time = depressurized_time.replace(':', '')  # 'HHMMSS' format

afm_images_path = 'C:\\Users\\chasg\\OneDrive - The University of Western Ontario\\MESc\\Research\\Experiments\\expt005_250403_gas_inflations\\raw_data\\'+depressurized_day+'\\FlattenedData'

deflation_curves_path = 'C:\\Users\\chasg\\OneDrive - The University of Western Ontario\\MESc\\Research\\Experiments\\expt005_250403_gas_inflations\\data_processing\\deflation_curves'
deflation_curve_filename = f'deflation_curve_{sample_number}_depressurized{depressurized_day}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv'
deflation_curve_path = deflation_curves_path + f'\\{deflation_curve_filename}'