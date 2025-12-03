import igor.binarywave
import numpy as np
from datetime import datetime, timedelta

class AFMImage:
    """
    The AFMImage class represents an AFM image as numpy arrays and metadata.
    
    Attributes:
    ----------
    data : dict
        The raw data loaded from the .ibw file.
    wave_data : numpy.ndarray
        The set of 2D arrays of data points from the AFM images.
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
        """Get retrace for a specific index with optional unit conversion."""
        if self.wave_data is not None:
            return np.rot90(self.wave_data[:, :, index], k=1) * unit_conversion
        return None

    # Tapping mode: raw data indexes 0-3, contact mode: raw data indexes 0-2
    def get_height_retrace(self):
        """Get retrace for index 0, which is the default index for height in both tapping and contact modes. Converts units to nm."""
        return self.get_retrace_data(0, unit_conversion=1e9)

    def get_contrast_retrace(self):
        """Get retrace for index 1, which is the default index for amplitude in tapping mode and deflection in contact mode. Both give high contrast to surface topographies, so can be used to extract a high-contrast qualitative map of topographical features from images taken in either mode."""
        return self.get_retrace_data(1)

    def get_phase_retrace(self):
        """If a tapping mode image, this will return the phase retrace (assuming it is stored at the default index, index 2). If a contact mode image, this will return None."""
        if self.get_imaging_mode() == 'AC Mode':
            return self.get_retrace_data(2)
        elif self.get_imaging_mode() == 'Contact':
            print("Phase retrace is not available for Contact mode images.")
            return None

    def get_ZSensorRetrace(self):
        """Get retrace for the Z sensor, pulling from index 3 for tapping mode images and index 2 for contact mode images (their default indexes). Converts units to nm."""
        if self.get_imaging_mode() == 'AC Mode':
            return self.get_retrace_data(3, unit_conversion=1e9)
        elif self.get_imaging_mode() == 'Contact':
            return self.get_retrace_data(2, unit_conversion=1e9)

    def get_FlatHeight(self):
        """Get the flattened height retrace (assuming postprocessing was done in Igor which put the flat height in the next free index after the Z retrace). Converts units to nm."""
        if self.get_imaging_mode() == 'AC Mode':
            return self.get_retrace_data(4, unit_conversion=1e9)
        elif self.get_imaging_mode() == 'Contact':
            return self.get_retrace_data(3, unit_conversion=1e9)

    def get_FlatZtrace(self):
        """Get the flattened Z retrace (assuming postprocessing was done in Igor which put the flat Z retrace in the second next free index after the Z retrace). Converts units to nm."""
        if self.get_imaging_mode() == 'AC Mode':
            return self.get_retrace_data(5, unit_conversion=1e9)
        elif self.get_imaging_mode() == 'Contact':
            return self.get_retrace_data(4, unit_conversion=1e9)

    # Metadata Extraction Methods
    ## Imaging parameters
    def get_imaging_mode(self):
        """Get the imaging mode from the note. Tapping is 'AC Mode' and contact is 'Contact'."""
        return self._extract_parameter('ImagingMode')

    def get_scan_rate(self):
        return float(self._extract_parameter('ScanRate')) # Units are Hz

    def get_initial_drive_amplitude(self):
        """Get the initial drive amplitude from the note (if drive amplitude not changed during imaging, this will be the drive amplitude for the entire image). Converts units to mV."""
        return float(self._extract_parameter('DriveAmplitude'))*1e3 # Units are in mV

    def get_tapping_setpoint(self):
        """Get the setpoint from the note for a tapping mode image. Units are in V."""
        return float(self._extract_parameter('AmplitudeSetpointVolts'))
    
    def get_contact_setpoint(self):
        """Get the setpoint from the note for a contact mode image. Units are in V."""
        return float(self._extract_parameter('DeflectionSetpointVolts'))

    ## Image geometry/position
    def get_scan_size(self):
        """Note that ScanSize is the size the image would be if the image was completed. For partial images, this is larger than the actual image size in the slow scan direction."""
        return float(self._extract_parameter('ScanSize'))*1e6 # Units are um
    
    def get_x_y_pixel_counts(self):
        image_shape = self.get_height_retrace().shape
        return image_shape[1], image_shape[0]  # (X pixels, Y pixels)

    def get_x_offset(self):
        """Get the X offset from the note. Converts units to um."""
        return float(self._extract_parameter('XOffset'))*1e6

    def get_y_offset(self):
        """Get the Y offset from the note. Converts units to um."""
        return float(self._extract_parameter('YOffset'))*1e6

    def get_pointsLines(self):
        return float(self._extract_parameter('PointsLines'))

    def get_scan_direction(self):
        """1 if scanning down, 0 if scanning up."""
        return int(self._extract_parameter('ScanDown'))

    def get_filename(self):
        return self._extract_parameter('FileName')
    
    def get_FastScanSize(self):
        return float(self._extract_parameter('FastScanSize')) * 1e6 # Units are in microns
    def get_SlowScanSize(self):
        """Note that SlowScanSize is the size the image would be in the slow scan direction if the image was completed. For partial images, this is larger than the actual image size."""
        return float(self._extract_parameter('SlowScanSize')) * 1e6 # Units are in microns
    def get_ScanPoints(self):
        """Get the number of scan points (pixels) in the fast scan direction."""
        return int(self._extract_parameter('ScanPoints'))
    def get_ScanLines(self):
        """Get the number of scan lines (pixels) that would be in the slow scan direction if the image was completed. For partial images, this is larger than the actual number of lines in the image."""
        return int(self._extract_parameter('ScanLines'))
    
    def get_x_y_size(self):
        """Calculate the physical size of the image in the X and Y directions, accounting for partial images. Returns a tuple (x_size in um, y_size in um)."""
        return self.get_FastScanSize()*self.get_x_y_pixel_counts()[0]/self.get_ScanPoints(), self.get_SlowScanSize()*self.get_x_y_pixel_counts()[1]/self.get_ScanLines() # Returns (x_size in um, y_size in um)

    def get_pixel_size(self):
        """
        Calculate the pixel size. Units are same as FastScanSize and SlowScanSize (um).
        Validates that the pixels are square by comparing FastScanSize/ScanPoints and SlowScanSize/ScanLines.
        Raises ValueError if pixels are not square.
        """
        fast_size = self.get_FastScanSize()
        slow_size = self.get_SlowScanSize()
        points = self.get_ScanPoints()
        lines = self.get_ScanLines()

        pixel_size_x = fast_size / points
        pixel_size_y = slow_size / lines

        # Check if pixel dimensions match within a small tolerance (relative tolerance of 1%)
        if not np.isclose(pixel_size_x, pixel_size_y, rtol=0.01):
            raise ValueError(f"Non-square pixels: X size ({pixel_size_x:.3e}) != Y size ({pixel_size_y:.3e})")

        return pixel_size_x

    ## Date and Time Methods
    def get_datetime(self):
        combined_str = f"{self._extract_parameter('Date')} {self._extract_parameter('Time')}"
        # Parse the combined string into a datetime object
        return datetime.strptime(combined_str, "%Y-%m-%d %I:%M:%S %p")

    def get_imaging_duration(self):
        """Calculate the total imaging duration in seconds."""
        return self.get_x_y_pixel_counts()[1]/self.get_scan_rate()

    def get_acquisition_start_time(self):
        """
        Get the time when the image acquisition started.
        Assumes get_datetime() returns the finish time.
        """
        return self.get_datetime() - timedelta(seconds=self.get_imaging_duration())

    def get_line_acquisition_time(self, line_index):
        """
        Calculate the acquisition time for a specific line index.
        
        Args:
            line_index (int): The 0-based index of the line.
            
        Returns:
            datetime: When the line was imaged.
        """
        start_time = self.get_acquisition_start_time()
        duration = self.get_imaging_duration()
        direction = self.get_scan_direction()
        _, total_lines = self.get_x_y_pixel_counts()
        
        # Clamp index
        line_index = max(0, min(line_index, total_lines - 1))
        
        ratio = line_index / total_lines
        
        if direction == 1: # Scan Down
            offset_seconds = ratio * duration
        else: # Scan Up
            offset_seconds = (1 - ratio) * duration
            
        return start_time + timedelta(seconds=offset_seconds)

    ## Helper method to extract parameters from the note
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

    ## Methods for extracting maximum points and traces
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
        _, max_position = self.get_maximum_Zpoint()
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
        _, max_position = self.get_maximum_Hpoint()
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
        
        shift = np.abs(y[0])
        
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
        
        shift = np.abs(y[0])
        
        for i in range(len(y)):
            y[i] = y[i] + shift 

        return x, y, max(y), shift
