import igor.binarywave
import numpy as np
from datetime import datetime, timedelta


# Pixel Coordinate Functions (module-level for reuse without AFMImage object)

def compute_x_pixel_coords(x_pixel_count, pixel_size):
    """
    Compute x-coordinates of pixel centers in microns.
    
    Pixel edges span (0, x_size). Centers are offset by half a pixel.
    """
    return (np.arange(x_pixel_count) + 0.5) * pixel_size


def compute_y_pixel_coords(y_pixel_count, pixel_size):
    """
    Compute y-coordinates of pixel centers in microns.
    
    Pixel edges span (0, y_size). Centers are offset by half a pixel.
    Row index 0 is at the TOP of the image (highest y-value).
    """
    return (y_pixel_count - np.arange(y_pixel_count) - 0.5) * pixel_size


def index_to_x_coord(x_index, x_pixel_count, pixel_size):
    """Convert a column index to the x-coordinate at the center of that pixel."""
    x_index = int(np.clip(x_index, 0, x_pixel_count - 1))
    return (x_index + 0.5) * pixel_size


def x_to_nearest_index(x_um, x_pixel_count, pixel_size):
    """Convert an x-coordinate (in microns) to the nearest column index."""
    idx_float = (x_um / pixel_size) - 0.5
    return int(np.clip(int(round(idx_float)), 0, x_pixel_count - 1))


def index_to_y_coord(y_index, y_pixel_count, pixel_size):
    """Convert a row index to the y-coordinate at the center of that pixel."""
    y_index = int(np.clip(y_index, 0, y_pixel_count - 1))
    return (y_pixel_count - (y_index + 0.5)) * pixel_size


def y_to_nearest_index(y_um, y_pixel_count, pixel_size):
    """Convert a y-coordinate (in microns) to the nearest row index."""
    idx_float = y_pixel_count - (y_um / pixel_size) - 0.5
    return int(np.clip(int(round(idx_float)), 0, y_pixel_count - 1))


# AFMImage Class

class AFMImage:
    """
    The AFMImage class represents an AFM image as numpy arrays and metadata.
    
    Attributes:
    ----------
    data : dict
        The raw data loaded from the .ibw file.
    wave_data : numpy.ndarray
        The set of 2D arrays of data points (channels/layers) from the AFM image.
    note : str
        The note associated with the AFM image.
    labels : list
        The labels associated with the AFM image.
    channel_names : list
        The names of the data channels/layers in the AFM image.
    """

    # Class initialization and loading of the .ibw file
    def __init__(self, file_path):
        self.data = self.load_ibw_file(file_path)
        if not self.data or 'wave' not in self.data:
            raise RuntimeError(f"Cannot load wave from {file_path}")
        wave = self.data['wave']
        self.hdr = wave['wave_header']
        self.bname = self.hdr['bname']
        self.wave_data = wave['wData']
        self.note = wave['note'].decode('latin-1', 'replace')
        self.labels = wave['labels']
        self.channel_names = [label.decode('latin-1', 'replace') for label in self.labels[2]][1:]

    def load_ibw_file(self, file_path):
        """Load an Igor Binary Wave (.ibw) file from the specified file path."""
        try:
            return igor.binarywave.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    # Channel Data Methods
    def get_channel_data(self, index=None, channel_name=None, unit_conversion=1):
        """Get channel data for a specific index or channel name with optional unit conversion."""
        if self.wave_data is not None:
            if index is not None:
                return np.rot90(self.wave_data[:, :, index], k=1) * unit_conversion
            elif channel_name is not None:
                if channel_name in self.channel_names:
                    index = self.channel_names.index(channel_name)
                return np.rot90(self.wave_data[:, :, index], k=1) * unit_conversion
        return None

    def get_number_of_arrays(self):
        """Get the number of data arrays (a.k.a. channels/layers in the image) in the wave data."""
        return self.hdr['nDim'][2]

    def get_height_retrace(self):
        """Returns the image's height retrace channel. Converts units to nm."""
        return self.get_channel_data(channel_name='HeightRetrace', unit_conversion=1e9)
    
    def get_height_trace(self):
        """Returns the image's height trace channel. Converts units to nm."""
        return self.get_channel_data(channel_name='HeightTrace', unit_conversion=1e9)

    def get_amplitude_retrace(self):
        """Returns the image's amplitude retrace channel."""
        return self.get_channel_data(channel_name='AmplitudeRetrace')
    
    def get_amplitude_trace(self):
        """Returns the image's amplitude trace channel."""
        return self.get_channel_data(channel_name='AmplitudeTrace')

    def get_deflection_retrace(self):
        """Returns the image's deflection retrace channel."""
        return self.get_channel_data(channel_name='DeflectionRetrace')
    
    def get_deflection_trace(self):
        """Returns the image's deflection trace channel."""
        return self.get_channel_data(channel_name='DeflectionTrace')

    def get_contrast_map(self):
        """If the image was taken in tapping mode and an amplitude channel is available, this will return AmplitudeRetrace or AmplitudeTrace. If the image was taken in contact mode and a deflection channel is available, this will return the deflection retrace or deflection trace. All give high contrast to surface topographies, so can be used to extract a high-contrast qualitative map of topographical features from images taken in either mode."""
        potential_contrast_maps = ['AmplitudeRetrace', 'AmplitudeTrace', 'DeflectionRetrace', 'DeflectionTrace']
        for channel_name in potential_contrast_maps:
            if channel_name in self.channel_names:
                return self.get_channel_data(channel_name=channel_name)

    def get_phase_retrace(self):
        """Returns the image's phase retrace channel. Units are in degrees."""
        return self.get_channel_data(channel_name='PhaseRetrace')

    def get_phase_trace(self):
        """Returns the image's phase trace channel. Units are in degrees."""
        return self.get_channel_data(channel_name='PhaseTrace')

    def get_z_sensor_retrace(self):
        """Returns the image's Z sensor retrace channel. Converts units to nm."""
        return self.get_channel_data(channel_name='ZSensorRetrace', unit_conversion=1e9)
    
    def get_z_sensor_trace(self):
        """Returns the image's Z sensor trace channel. Converts units to nm."""
        return self.get_channel_data(channel_name='ZSensorTrace', unit_conversion=1e9)

    def get_flat_height_retrace(self):
        """Get the flattened height retrace (assuming postprocessing was done in Igor which flattened the height retrace and put that flattened image in 'HeightRetraceMod0'). Converts units to nm."""
        return self.get_channel_data(channel_name='HeightRetraceMod0', unit_conversion=1e9)
    
    def get_flat_height_trace(self):
        """Get the flattened height trace (assuming postprocessing was done in Igor which flattened the height trace and put that flattened image in 'HeightTraceMod0'). Converts units to nm."""
        return self.get_channel_data(channel_name='HeightTraceMod0', unit_conversion=1e9)

    def get_flat_z_retrace(self):
        """Get the flattened Z retrace (assuming postprocessing was done in Igor which flattened the Z retrace and put that flattened image in 'ZSensorRetraceMod0'). Converts units to nm."""
        return self.get_channel_data(channel_name='ZSensorRetraceMod0', unit_conversion=1e9)
    
    def get_flat_z_trace(self):
        """Get the flattened Z trace (assuming postprocessing was done in Igor which flattened the Z trace and put that flattened image in 'ZSensorTraceMod0'). Converts units to nm."""
        return self.get_channel_data(channel_name='ZSensorTraceMod0', unit_conversion=1e9)


    # Metadata Extraction Methods
    ## Imaging parameters
    def get_imaging_mode(self):
        """Get the imaging mode from the note. Tapping is 'AC Mode' and contact is 'Contact'."""
        return self._extract_parameter('ImagingMode')

    def get_scan_rate(self):
        """Get the scan rate from the note. Units are in Hz (lines per second)."""
        return float(self._extract_parameter('ScanRate'))

    def get_initial_drive_amplitude(self):
        """Get the initial drive amplitude from the note (if drive amplitude not changed during imaging, this will be the drive amplitude for the entire image). Converts units to mV."""
        return float(self._extract_parameter('DriveAmplitude'))*1e3

    def get_tapping_setpoint(self):
        """Get the setpoint from the note for a tapping mode image. Units are in V."""
        return float(self._extract_parameter('AmplitudeSetpointVolts'))
    
    def get_contact_setpoint(self):
        """Get the setpoint from the note for a contact mode image. Units are in V."""
        return float(self._extract_parameter('DeflectionSetpointVolts'))

    ## Image geometry/position
    def get_scan_size(self):
        """Note that ScanSize is the size the image would be if the image was completed. For partial images, this is larger than the actual image size in the slow scan direction. Converts units to um."""
        return float(self._extract_parameter('ScanSize'))*1e6

    def get_x_y_pixel_counts(self):
        """Get the number of pixels in the X and Y directions as a tuple (X pixels, Y pixels).
        Equivalent to using self.hdr['nDim'][:2]"""
        image_shape = self.get_height_retrace().shape
        return image_shape[1], image_shape[0]

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
        return self.get_FastScanSize()*self.get_x_y_pixel_counts()[0]/self.get_ScanPoints(), self.get_SlowScanSize()*self.get_x_y_pixel_counts()[1]/self.get_ScanLines()

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

    def get_x_pixel_coords(self):
        """Get the x-coordinates of pixel centers in microns."""
        return compute_x_pixel_coords(self.get_x_y_pixel_counts()[0], self.get_pixel_size())

    def get_y_pixel_coords(self):
        """Get the y-coordinates of pixel centers in microns."""
        return compute_y_pixel_coords(self.get_x_y_pixel_counts()[1], self.get_pixel_size())

    def index_to_x_center(self, x_index):
        """Convert a column index to the x-coordinate at the center of that pixel."""
        return index_to_x_coord(x_index, self.get_x_y_pixel_counts()[0], self.get_pixel_size())

    def x_to_nearest_index(self, x_um):
        """Convert an x-coordinate (in microns) to the nearest column index."""
        return x_to_nearest_index(x_um, self.get_x_y_pixel_counts()[0], self.get_pixel_size())

    def index_to_y_center(self, y_index):
        """Convert a row index to the y-coordinate at the center of that pixel."""
        return index_to_y_coord(y_index, self.get_x_y_pixel_counts()[1], self.get_pixel_size())

    def y_to_nearest_index(self, y_um):
        """Convert a y-coordinate (in microns) to the nearest row index."""
        return y_to_nearest_index(y_um, self.get_x_y_pixel_counts()[1], self.get_pixel_size())

    ## Date and Time Methods
    def get_scan_end_datetime(self):
        """Get when the image scan finished as a datetime object."""
        combined_str = f"{self._extract_parameter('Date')} {self._extract_parameter('Time')}"
        # Parse the combined string into a datetime object
        return datetime.strptime(combined_str, "%Y-%m-%d %I:%M:%S %p")

    def get_imaging_duration(self):
        """Calculate the total imaging duration in seconds."""
        return self.get_x_y_pixel_counts()[1]/self.get_scan_rate()

    def get_scan_start_datetime(self):
        """
        Get when the image scan started as a datetime object.
        """
        return self.get_scan_end_datetime() - timedelta(seconds=self.get_imaging_duration())

    def get_line_acquisition_datetime(self, line_index):
        """
        Calculate the acquisition time for a specific line index.
        
        Args:
            line_index (int): The 0-based index of the line.

        Returns:
            datetime: When the line was imaged.
        """
        start_time = self.get_scan_start_datetime()
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

    ## Other Metadata Methods
    def get_filename(self):
        return self._extract_parameter('FileName')
    
    # If I have a need to check if a channel was saved raw or flattened, look for FlattenOrder {channel_index} in the note.

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

    ## Methods for extracting maximum points and retraces
    def get_maximum_point(self, retrace_function):
        retrace_data = retrace_function()
        if retrace_data is not None:
            max_value = np.max(retrace_data)
            max_position = np.unravel_index(np.argmax(retrace_data), retrace_data.shape)
            return max_value, max_position
        return None, None

    def get_maximum_Zpoint(self):
        return self.get_maximum_point(self.get_flat_z_retrace)

    def get_maximum_Hpoint(self):
        return self.get_maximum_point(self.get_flat_height_retrace)

    def get_trace(self, retrace_function, max_position):
        retrace_data = retrace_function()
        if retrace_data is not None and max_position is not None:
            trace_x = retrace_data[max_position[0], :]
            trace_y = retrace_data[:, max_position[1]]
            return trace_x, trace_y
        return None, None

    def get_trace_z(self):
        _, max_position = self.get_maximum_Zpoint()
        return self.get_trace(self.get_flat_z_retrace, max_position)

    def get_trace_h(self):
        _, max_position = self.get_maximum_Hpoint()
        return self.get_trace(self.get_flat_height_retrace, max_position)

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
