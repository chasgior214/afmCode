import igor.binarywave
import numpy as np
from datetime import datetime

class AFMImage:
    def __init__(self, file_path):
        self.data = self.load_ibw_file(file_path)
        self.wave_data = self.data.get('wave', {}).get('wData', None) if self.data else None
        self.note = self.data.get('wave', {}).get('note', b'').decode('utf-8', 'replace') if self.data else None
        self.labels = self.data.get('wave', {}).get('labels', []) if self.data else []
    
    def load_ibw_file(self, file_path):
        try:
            loaded_data = igor.binarywave.load(file_path)
            return loaded_data
        except Exception as e: 
            print(f"Error loading file {file_path}: {e}")
            return None 
    
    def get_height_retrace(self):
        return np.rot90(self.wave_data[:,:,0], k=1) * 1e9 #Units are in nm
    
    def get_amplitude_retrace(self):
        return np.rot90(self.wave_data[:,:,1], k=1)
    
    def get_phase_retrace(self):
        return np.rot90(self.wave_data[:,:,2], k=1)
    
    def get_ZSensorRetrace(self):
        return np.rot90(self.wave_data[:,:,3], k=1) * 1e9 #Units are in nm
    
    def get_FlatHeight(self):
        return np.rot90(self.wave_data[:,:,4], k=1) * 1e9 #Units are in nm
    
    def get_FlatZtrace(self):
        return np.rot90(self.wave_data[:,:,5], k=1) * 1e9 #Unites are in nm
    
    def get_scan_rate(self):
        return float(self._extract_parameter('ScanRate')) #Units are Hz

    def get_scan_size(self):
        return float(self._extract_parameter('ScanSize'))*1e6 #Units are um 

    def get_drive_amplitude(self):
        return float(self._extract_parameter('DriveAmplitude'))*1e3 #Units are in mV

    def get_setpoint(self):
        # Since Setpoint might have different keys, adjust as needed
        return float(self._extract_parameter('Setpoint', alternative_keys=['AmplitudeSetpointVolts', 'DeflectionSetpointVolts']))
    
    def get_date(self):
        return self._extract_parameter('Date')
    
    def get_time(self):
        return self._extract_parameter('Time')
    
    def get_pointsLines(self):
        return float(self._extract_parameter('PointsLines'))
    
    def get_fileName(self):
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
    
    def get_datetime(self):
        date_str = self.get_date()  # Assume this returns something like "2024-02-27"
        time_str = self.get_time()  # Assume this returns something like "11:20:48 AM"
        combined_str = f"{date_str} {time_str}"
        # Parse the combined string into a datetime object
        return datetime.strptime(combined_str, "%Y-%m-%d %I:%M:%S %p")
    
    def get_maximum_Zpoint(self):
        if self.wave_data is not None:
            max_value = np.max(self.get_FlatZtrace())
            max_position = np.unravel_index(np.argmax(self.get_FlatZtrace()), self.get_FlatZtrace().shape)
            return max_value, max_position
        return None, None
    
    def get_maximum_Hpoint(self):
        if self.wave_data is not None:
            max_value = np.max(self.get_FlatHeight())
            max_position = np.unravel_index(np.argmax(self.get_FlatHeight()), self.get_FlatHeight().shape)
            return max_value, max_position
        return None, None
    
    def get_trace_z(self):
        max_value, max_position = self.get_maximum_Zpoint()
        if max_position is not None:
            z_trace_x = self.get_FlatZtrace()[max_position[0],:]
            z_trace_y = self.get_FlatZtrace()[:,max_position[1]]
            return z_trace_x, z_trace_y
        return None, None 
    
    def get_trace_h(self):
        max_value, max_position = self.get_maximum_Hpoint()
        if max_position is not None:
            h_trace_x = self.get_FlatHeight()[max_position[0],:]
            h_trace_y = self.get_FlatHeight()[:,max_position[1]]
            return h_trace_x, h_trace_y
        return None, None 

    def get_conversion_rate(self):
        conv = self.get_scan_size()/self.get_pointsLines() 
        return conv
    
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
    
    def get_trimmed_trace_z(self,l):
        zxx = self.get_x_4z()
        raw_trace, b = self.get_trace_z() 
        x = []
        y = [] 
        shift = 0
        max_value = 0 

        for i in range(len(zxx)):
            if zxx[i] <= l and zxx[i] >= (-1*l):
                x.append(zxx[i])
                y.append(raw_trace[i])
        
        if y[0] < 0: 
            shift = -1 * y[0]
        else:
            shift = y[0]
        
        for i in range(len(y)):
            y[i] = y[i] + shift 

        max_value = max(y)

        return x, y, max_value, shift 
    
    def get_trimmed_trace_h(self,l):
        hxx = self.get_x_4H()
        raw_trace, b = self.get_trace_h()
        x = [] 
        y = []
        shift = 0 
        max_value = 0 

        for i in range(len(hxx)):
            if hxx[i] <= l and hxx[i] >= (-1*l):
                x.append(hxx[i])
                y.append(raw_trace[i])
        
        if y[0] < 0:
            shift = -1 * y[0]
        else:
            shift = y[0]
        
        for i in range(len(y)):
            y[i] = y[i] + shift 
        
        max_value = max(y)

        return x, y, max_value, shift
