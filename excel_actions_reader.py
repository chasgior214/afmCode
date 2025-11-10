from path_loader import excel_action_tracker_path
import pandas as pd

"""
The experiments whose data are processed with this repo's code are done as follows:
- Microcavities covered in a graphene membrane are charged for an extended duration in a pressurized gas
- The sample is removed from the pressure cell and taken to an AFM to image how the membrane deflection changes over time as the gas leaks through the membrane
- The ibw files accessed are the AFM images
- An Excel file exists which tracks the following actions taken on the sample:
* The action number, chronicling the order the actions were taken in
* The date the action was taken on
* For pressurizations (and re-pressurizations to counter leaks: the gas species pressurized in, the time the sample was pressurized (/re-pressurized), and the pressure
* For depressurizations: the time the sample was depressurized
* For AFM images, the filename(s) of images taken, and at which location on the sample they were taken
* The different sheets titles in the Excel are the names of different samples
"""

def load_sample_action_tracker(sample_sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_action_tracker_path, sheet_name=sample_sheet_name)
    return df

class AFMImagePointer:
    """Class to represent a pointer to an AFM image and its metadata"""
    def __init__(self, sample: str, filename: str, date: str, location: str):
        self.sample = sample
        self.filename = filename
        self.date = date
        self.location = location


def list_depressurization_action_numbers(df: pd.DataFrame) -> list[int]:
    """List the action numbers where the gas is 'air' and the previous action is not also 'air'"""
    depressurization_actions = []
    for idx, row in df.iterrows():
        if row['Gas'] == 'air':
            if idx == 0 or df.at[idx - 1, 'Gas'] != 'air':
                depressurization_actions.append(row['Action Number'])
    return depressurization_actions


def list_AFM_images_from_a_depressurization(sample: str, df: pd.DataFrame, depressurization_action_number: int) -> list[AFMImagePointer]:
    """Given a DataFrame and a depressurization action number, list the AFM image pointers for images taken after that depressurization and before the next pressurization"""
    # first, find the action number of the next pressurization (first time Gas is not 'air' after the depressurization)
    next_pressurization_action_number = None
    for idx, row in df.iterrows():
        if row['Action Number'] > depressurization_action_number and row['Gas'] != 'air':
            next_pressurization_action_number = row['Action Number']
            break
    # now, for all actions between depressurization_action_number and next_pressurization_action_number, collect the non-NaN strings in the 'AFM Image(s)' column along with their dates and sample location ('Location on Sample' column)
    afm_image_range_strings = []
    for idx, row in df.iterrows():
        action_number = row['Action Number']
        if action_number > depressurization_action_number and (next_pressurization_action_number is None or action_number < next_pressurization_action_number):
            afm_image_str = row['AFM Image(s)']
            if isinstance(afm_image_str, str) and afm_image_str.strip():
                afm_image_range_strings.append((afm_image_str, row['Date'], row['Location on Sample']))
    # whenever the string contains a -, split it into multiple filenames for all the numbers in the range (ex Image0001-0010 -> Image0001, Image0002, ..., Image0010)
    afm_image_pointers = []
    for img_str, date, location in afm_image_range_strings:
        if '-' in img_str:
            prefix = ''.join(filter(str.isalpha, img_str))
            range_part = img_str[len(prefix):]
            start_str, end_str = range_part.split('-')
            start_num = int(start_str)
            end_num = int(end_str)
            for num in range(start_num, end_num + 1):
                filename = f"{prefix}{num:04d}"
                afm_image_pointers.append(AFMImagePointer(sample, filename, date, location))
        else:
            afm_image_pointers.append(AFMImagePointer(sample, img_str, date, location))

    return afm_image_pointers


if __name__ == "__main__":
    sample_sheet_name = 'sample37'
    df = load_sample_action_tracker(sample_sheet_name)
    print(df.head())
    depressurization_actions = list_depressurization_action_numbers(df)
    print("Depressurization action numbers:", depressurization_actions)
    # show AFM images taken after the last depressurization
    if depressurization_actions:
        afm_images = list_AFM_images_from_a_depressurization(sample_sheet_name, df, depressurization_actions[-1])
        print(f"AFM images taken after depressurization of action {depressurization_actions[-1]}:")
        for img_pointer in afm_images:
            print(f"  - {img_pointer.filename} on {img_pointer.date} at {img_pointer.location}")
    # find total number of AFM images listed in all actions for sample37
    total_afm_images = 0
    for idx, row in df.iterrows():
        afm_image_str = row['AFM Image(s)']
        if isinstance(afm_image_str, str) and afm_image_str.strip():
            if '-' in afm_image_str:
                prefix = ''.join(filter(str.isalpha, afm_image_str))
                range_part = afm_image_str[len(prefix):]
                start_str, end_str = range_part.split('-')
                start_num = int(start_str)
                end_num = int(end_str)
                total_afm_images += (end_num - start_num + 1)
            else:
                total_afm_images += 1
    print(f"Total number of AFM images listed in all actions for {sample_sheet_name}: {total_afm_images}")
    # find total number of AFM images listed in all actions for all samples in the Excel file
    excel_file = pd.ExcelFile(excel_action_tracker_path)
    total_afm_images_all_samples = 0
    for sheet_name in excel_file.sheet_names:
        df_sample = pd.read_excel(excel_action_tracker_path, sheet_name=sheet_name)
        for idx, row in df_sample.iterrows():
            afm_image_str = row['AFM Image(s)']
            try:
                if isinstance(afm_image_str, str) and afm_image_str.strip():
                    if '-' in afm_image_str:
                        prefix = ''.join(filter(str.isalpha, afm_image_str))
                        range_part = afm_image_str[len(prefix):]
                        start_str, end_str = range_part.split('-')
                        start_num = int(start_str)
                        end_num = int(end_str)
                        total_afm_images_all_samples += (end_num - start_num + 1)
                    else:
                        total_afm_images_all_samples += 1
            except Exception as e:
                print(f"Error processing AFM Image(s) entry '{afm_image_str}' in sheet '{sheet_name}': {e}")
    print(f"Total number of AFM images listed in all actions for all samples: {total_afm_images_all_samples}")