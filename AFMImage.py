import igor.binarywave
import numpy as np
from datetime import datetime

class AFMImage:
    """
    The AFMImage class represents an AFM image as numpy arrays and metadata. Channel 0-3 extraction is based on the default order for tapping mode, channels 4-5 are for postprocessed data.
    
    Attributes:
    ----------
    data : dict
        The raw data loaded from the .ibw file.
    wave_data : numpy.ndarray
        The 2D array of data points from the AFM image.
    note : str
        The note associated with the AFM image.
    labels : list
        The labels associated with the AFM image.
    """
    # Class initialization and loading of the .ibw file
    def __init__(self, file_path):
        self.data = self.load_ibw_file(file_path)
        if not self.data or 'wave' not in self.data:
            raise RuntimeError(f"Cannot load wave from {file_path}")
        wave = self.data['wave']
        self.hdr  = wave['wave_header']
        self.bname = self.hdr['bname']
        self.wave_data = wave['wData']
        self.note = wave['note'].decode('utf-8', 'replace')
        self.labels = wave['labels']

    def load_ibw_file(self, file_path):
        """Load an Igor Binary Wave (.ibw) file from the specified file path."""
        try:
            return igor.binarywave.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    # Retrace Data Methods
    def get_retrace_data(self, index, unit_conversion=1):
        """Get retrace data for a specific index with optional unit conversion."""
        if self.wave_data is not None:
            return np.rot90(self.wave_data[:, :, index], k=1) * unit_conversion
        return None

    # Note that the channel order for extractions below is for tapping mode after channel 1 (raw data channels 0-3, postprocessed with AFM software 4-5)
    def get_height_retrace(self):
        """Get retrace data for channel 0, which is the default channel for height in both tapping and contact modes."""
        return self.get_retrace_data(0, unit_conversion=1e9) # Units are in nm

    def get_contrast_retrace(self):
        """Get retrace data for channel 1, which is the default channel for amplitude in tapping mode and deflection in contact mode. Both give high contrast to surface topographies, so can be used to extract a high-contrast qualitative map of topographical features from images taken in either mode."""
        return self.get_retrace_data(1)

    def get_phase_retrace(self):
        return self.get_retrace_data(2)

    def get_ZSensorRetrace(self): # default for channel 2 in contact is z-sensor
        return self.get_retrace_data(3, unit_conversion=1e9) # Units are in nm

    def get_FlatHeight(self):
        return self.get_retrace_data(4, unit_conversion=1e9) # Units are in nm

    def get_FlatZtrace(self):
        return self.get_retrace_data(5, unit_conversion=1e9) # Units are in nm

    # Metadata Extraction Methods
    def get_scan_rate(self):
        return float(self._extract_parameter('ScanRate')) # Units are Hz

    def get_scan_size(self):
        return float(self._extract_parameter('ScanSize'))*1e6 # Units are um 

    def get_drive_amplitude(self):
        return float(self._extract_parameter('DriveAmplitude'))*1e3 # Units are in mV

    def get_setpoint(self):
        # Since Setpoint might have different keys, adjust as needed
        return float(self._extract_parameter('Setpoint', alternative_keys=['AmplitudeSetpointVolts', 'DeflectionSetpointVolts']))

    def get_date(self): # Assume this returns something like "2024-02-27"
        return self._extract_parameter('Date')

    def get_time(self): # Assume this returns something like "11:20:48 AM"
        return self._extract_parameter('Time')

    def get_datetime(self):
        combined_str = f"{self.get_date()} {self.get_time()}"
        # Parse the combined string into a datetime object
        return datetime.strptime(combined_str, "%Y-%m-%d %I:%M:%S %p")

    def get_pointsLines(self):
        return float(self._extract_parameter('PointsLines'))

    def get_filename(self):
        return self._extract_parameter('FileName')

    def _extract_parameter(self, key, alternative_keys=None):
        if self.note:
            for line in self.note.split('\r'):
                if ':' in line:
                    split_key, value = line.split(':', 1)
                    split_key = split_key.strip()
                    value = value.strip()

                    if split_key == key or (alternative_keys and split_key in alternative_keys):
                        return value
        return None

    # Methods for Altering Data
    def get_maximum_point(self, retrace_function):
        retrace_data = retrace_function()
        if retrace_data is not None:
            max_value = np.max(retrace_data)
            max_position = np.unravel_index(np.argmax(retrace_data), retrace_data.shape)
            return max_value, max_position
        return None, None

    def get_maximum_Zpoint(self):
        return self.get_maximum_point(self.get_FlatZtrace)

    def get_maximum_Hpoint(self):
        return self.get_maximum_point(self.get_FlatHeight)

    def get_trace(self, retrace_function, max_position):
        retrace_data = retrace_function()
        if retrace_data is not None and max_position is not None:
            trace_x = retrace_data[max_position[0], :]
            trace_y = retrace_data[:, max_position[1]]
            return trace_x, trace_y
        return None, None

    def get_trace_z(self):
        _, max_position = self.get_maximum_Zpoint()
        return self.get_trace(self.get_FlatZtrace, max_position)

    def get_trace_h(self):
        _, max_position = self.get_maximum_Hpoint()
        return self.get_trace(self.get_FlatHeight, max_position)

    def get_conversion_rate(self):
        return self.get_scan_size()/self.get_pointsLines()

    def get_x_4z(self):
        conversion_rate = self.get_conversion_rate()
        max_value, max_position = self.get_maximum_Zpoint()
        a, b = self.get_trace_z()
        z_x_x = [None] * len(a) 
        if max_position is not None:

            count = -1 * conversion_rate
            for i in range(0, max_position[1]):
                z_x_x[max_position[1]-1-i] = count
                count = count - conversion_rate        

            count2 = 0
            for i in range(max_position[1], len(a)):
                z_x_x[i] = count2
                count2 = count2 + conversion_rate
            
            return z_x_x
        return None

    def get_x_4H(self):
        conversion_rate = self.get_conversion_rate()
        max_value, max_position = self.get_maximum_Hpoint()
        a, b = self.get_trace_h()
        h_x_x = [None] * len(a)
        if max_position is not None:
            count = -1 * conversion_rate
            for i in range(0, max_position[0]):
                h_x_x[max_position[0]-1-i] = count
                count = count - conversion_rate 
            count2 = 0 
            for i in range(max_position[0], len(a)):
                h_x_x[i] = count2
                count2 = count2 + conversion_rate
            return h_x_x
        return None

    def get_trimmed_trace_z(self, length):
        zxx = self.get_x_4z()
        raw_trace, b = self.get_trace_z() 
        x, y = [], []

        for i in range(len(zxx)):
            if -length <= zxx[i] <= length:
                x.append(zxx[i])
                y.append(raw_trace[i])
        
        if y[0] < 0:
            shift = -y[0]
        else:
            shift = y[0]
        
        for i in range(len(y)):
            y[i] = y[i] + shift 

        return x, y, max(y), shift 
    
    def get_trimmed_trace_h(self, length):
        hxx = self.get_x_4H()
        raw_trace, b = self.get_trace_h()
        x, y = [], []

        for i in range(len(hxx)):
            if -length <= hxx[i] <= length:
                x.append(hxx[i])
                y.append(raw_trace[i])
        
        if y[0] < 0:
            shift = -y[0]
        else:
            shift = y[0]
        
        for i in range(len(y)):
            y[i] = y[i] + shift 

        return x, y, max(y), shift
