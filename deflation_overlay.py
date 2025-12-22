"""
Deflation Overlay Visualization Tool

Overlays deflation height data onto an optical microscope image of
a well network, with patches colored according to well height at a
given time (controlled via slider).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import os
import glob

import path_loader as pl


def load_deflation_curves(folder=None, sample_id=None, depressurized_date=None, depressurized_time=None, cutoff_time=None):
    """
    Load deflation curve data from CSV files.
    
    Returns a dictionary mapping well names to DataFrames with columns:
    'Time (minutes)', 'Deflection (nm)'
    """
    if folder is None:
        folder = pl.deflation_curves_path
    if sample_id is None:
        sample_id = pl.sample_ID
    if depressurized_date is None:
        depressurized_date = pl.depressurized_date
    if depressurized_time is None:
        depressurized_time = pl.depressurized_time
    
    curves = {}
    
    # Find all CSV files matching the pattern
    pattern = f"deflation_curve_sample{sample_id}_depressurized{depressurized_date}_{depressurized_time}*.csv"
    for csv_file in glob.glob(os.path.join(folder, pattern)):
        df = pd.read_csv(csv_file)
        
        # Extract well name from filename
        # Pattern: deflation_curve_sample{}_depressurized{}_{}_{}_loc{}_{well}.csv
        basename = os.path.basename(csv_file)
        # The well name is typically encoded in the filename
        # For sample37 style: ...cav{well}.csv
        # For sample53 style: ...cav{coords}.csv
        
        # Try to extract well identifier - it's the last segment before .csv after 'cav'
        if '_cav' in basename:
            well_name = basename.split('_cav')[-1].replace('.csv', '')
        else:
            # Fallback: use filename without extension
            well_name = basename.replace('.csv', '')
        
        if 'Time (minutes)' in df.columns and 'Deflection (nm)' in df.columns:
            # Filter out data after cutoff_time if set
            if cutoff_time is not None:
                df = df[df['Time (minutes)'] <= cutoff_time]
            curves[well_name] = df[['Time (minutes)', 'Deflection (nm)']].copy()
    
    return curves


def interpolate_height(curve_df, time_minutes):
    """
    Interpolate height (deflection) at a given time from a deflection curve.
    
    Args:
        curve_df: DataFrame with 'Time (minutes)' and 'Deflection (nm)' columns
        time_minutes: Time to interpolate at
    
    Returns:
        Interpolated height in nm, or None if time is before first data point
    """
    if curve_df.empty:
        return None
    
    times = curve_df['Time (minutes)'].values
    heights = curve_df['Deflection (nm)'].values
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    heights = heights[sort_idx]
    
    min_time = times[0]
    max_time = times[-1]
    
    # Before first data point: return None (will be white)
    if time_minutes < min_time:
        return None
    
    # At or after last data point: return last value
    if time_minutes >= max_time:
        return heights[-1]
    
    # Linear interpolation between surrounding points
    right_idx = np.searchsorted(times, time_minutes)
    left_idx = right_idx - 1
    
    t_left, t_right = times[left_idx], times[right_idx]
    h_left, h_right = heights[left_idx], heights[right_idx]
    
    # Interpolation factor
    if t_right == t_left:
        return h_left
    
    frac = (time_minutes - t_left) / (t_right - t_left)
    return h_left + frac * (h_right - h_left)


class DeflationOverlay:
    """
    Interactive visualization overlaying deflation data on an optical microscope image.
    """
    
    def __init__(
        self,
        image_path,
        deflation_curves,
        well_map,
        anchor_well,
        anchor_pixel_x,
        anchor_pixel_y,
        patch_size_px,
        x_translation_px,
        y_translation_px,
        x_spacing=7.63,
        y_spacing=4.6,
        time_in_hours=False,
        failure_threshold=None,
        whiteout_after_last=False,
    ):
        """
        Args:
            image_path: Path to the optical microscope image
            deflation_curves: Dict mapping well names to DataFrames with deflation data
            well_map: Dict mapping well names to (x_coord, y_coord) grid coordinates
            anchor_well: Name of the well at the anchor pixel position
            anchor_pixel_x: X pixel position of the anchor well patch center
            anchor_pixel_y: Y pixel position of the anchor well patch center
            patch_size_px: Size of the square patches in pixels
            x_translation_px: Pixels per x_spacing movement in the grid
            y_translation_px: Pixels per y_spacing movement in the grid
            x_spacing: Physical x spacing between grid positions (default from membrane_relative_positions)
            y_spacing: Physical y spacing between grid positions (default from membrane_relative_positions)
            time_in_hours: If True, display time in hours; otherwise in minutes
            failure_threshold: If set, wells that fall below this height become black permanently
            whiteout_after_last: If True, wells show as white after their last data point
        """
        self.image_path = image_path
        self.deflation_curves = deflation_curves
        self.well_map = well_map
        self.anchor_well = anchor_well
        self.anchor_pixel_x = anchor_pixel_x
        self.anchor_pixel_y = anchor_pixel_y
        self.patch_size_px = patch_size_px
        self.x_translation_px = x_translation_px
        self.y_translation_px = y_translation_px
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.time_in_hours = time_in_hours
        self.failure_threshold = failure_threshold
        self.whiteout_after_last = whiteout_after_last
        
        # Precompute max times for each well if whiteout_after_last is enabled
        self.well_max_times = {}
        if whiteout_after_last:
            for well_name, df in deflation_curves.items():
                if not df.empty:
                    self.well_max_times[well_name] = df['Time (minutes)'].max()
        
        # Precompute failure times for each well (time when it first crosses below threshold)
        # Uses linear interpolation between the last point above and first point below
        self.failure_times = {}
        if failure_threshold is not None:
            for well_name, df in deflation_curves.items():
                if df.empty:
                    continue
                times = df['Time (minutes)'].values
                heights = df['Deflection (nm)'].values
                # Sort by time
                sort_idx = np.argsort(times)
                sorted_times = times[sort_idx]
                sorted_heights = heights[sort_idx]
                
                # Find where height crosses below threshold
                for i in range(len(sorted_heights)):
                    if sorted_heights[i] < failure_threshold:
                        if i == 0:
                            # First point is already below threshold
                            self.failure_times[well_name] = sorted_times[0]
                        else:
                            # Interpolate between previous point and this one
                            t0, t1 = sorted_times[i-1], sorted_times[i]
                            h0, h1 = sorted_heights[i-1], sorted_heights[i]
                            # Linear interpolation to find exact crossing time
                            # h0 + frac * (h1 - h0) = threshold
                            # frac = (threshold - h0) / (h1 - h0)
                            if h1 != h0:
                                frac = (failure_threshold - h0) / (h1 - h0)
                                self.failure_times[well_name] = t0 + frac * (t1 - t0)
                            else:
                                self.failure_times[well_name] = sorted_times[i]
                        break
        
        # Load image
        self.image = plt.imread(image_path)
        
        # Determine time range from all curves
        self.min_time = float('inf')
        self.max_time = float('-inf')
        for curve_df in deflation_curves.values():
            if not curve_df.empty:
                self.min_time = min(self.min_time, curve_df['Time (minutes)'].min())
                self.max_time = max(self.max_time, curve_df['Time (minutes)'].max())
        
        if self.min_time == float('inf'):
            self.min_time = 0
        if self.max_time == float('-inf'):
            self.max_time = 1
        
        # Determine height range for colorbar
        self.min_height = float('inf')
        self.max_height = float('-inf')
        for curve_df in deflation_curves.values():
            if not curve_df.empty:
                self.min_height = min(self.min_height, curve_df['Deflection (nm)'].min())
                self.max_height = max(self.max_height, curve_df['Deflection (nm)'].max())
        
        if self.min_height == float('inf'):
            self.min_height = 0
        if self.max_height == float('-inf'):
            self.max_height = 1
        
        # Get anchor coordinates
        if anchor_well not in well_map:
            raise ValueError(f"Anchor well '{anchor_well}' not found in well_map")
        self.anchor_coords = well_map[anchor_well]
        
        # Precompute patch positions for all wells in well_map
        self.patch_positions = {}  # well_name -> (center_x, center_y)
        for well_name, coords in well_map.items():
            dx = coords[0] - self.anchor_coords[0]
            dy = coords[1] - self.anchor_coords[1]
            
            # Convert grid coordinate difference to pixel offset
            # Note: Y is negated because image Y increases downward, but grid Y increases upward
            px_offset_x = dx * self.x_translation_px
            px_offset_y = -dy * self.y_translation_px
            
            center_x = self.anchor_pixel_x + px_offset_x
            center_y = self.anchor_pixel_y + px_offset_y
            
            self.patch_positions[well_name] = (center_x, center_y)
        
        self.fig = None
        self.ax = None
        self.patches = {}  # well_name -> Rectangle patch
        self.colorbar = None
        self.slider = None
        self.time_text = None
        # If failure threshold is set, use it as the min for the colorbar
        colorbar_min = self.failure_threshold if self.failure_threshold is not None else self.min_height
        self.norm = Normalize(vmin=colorbar_min, vmax=self.max_height)
        self.cmap = plt.cm.turbo
    
    def get_well_pixel_position(self, well_name):
        """Get the center pixel position for a well."""
        return self.patch_positions.get(well_name)
    
    def update_patches(self, time_minutes):
        """Update patch colors based on the current time."""
        for well_name, patch in self.patches.items():
            if well_name in self.deflation_curves:
                # Check if well has failed (dropped below threshold)
                if self.failure_threshold is not None and well_name in self.failure_times:
                    if time_minutes >= self.failure_times[well_name]:
                        # Well has failed - show as black
                        patch.set_facecolor('black')
                        patch.set_edgecolor('black')
                        continue
                
                # Check if should whiteout after last data point
                if self.whiteout_after_last and well_name in self.well_max_times:
                    if time_minutes > self.well_max_times[well_name]:
                        patch.set_facecolor('white')
                        patch.set_edgecolor('lightgray')
                        continue

                height = interpolate_height(self.deflation_curves[well_name], time_minutes)
                if height is None:
                    # Before data exists: white
                    patch.set_facecolor('white')
                    patch.set_edgecolor('gray')
                else:
                    # Color based on height
                    color = self.cmap(self.norm(height))
                    patch.set_facecolor(color)
                    patch.set_edgecolor('black')
            else:
                # No data for this well: white with dashed edge
                patch.set_facecolor('white')
                patch.set_edgecolor('lightgray')
                patch.set_linestyle('--')
        
        # Update time display
        if self.time_text is not None:
            if self.time_in_hours:
                hours = time_minutes / 60
                self.time_text.set_text(f'Time: {hours:.2f} h')
            else:
                self.time_text.set_text(f'Time: {time_minutes:.1f} min')
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive visualization."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15, right=0.85)
        
        # Display image
        self.ax.imshow(self.image)
        self.ax.set_title('Deflation Overlay')
        
        # Create patches for all wells
        half_size = self.patch_size_px / 2
        for well_name, (cx, cy) in self.patch_positions.items():
            patch = Rectangle(
                (cx - half_size, cy - half_size),
                self.patch_size_px,
                self.patch_size_px,
                facecolor='white',
                edgecolor='gray',
                linewidth=1.5,
                alpha=0.8,
            )
            self.ax.add_patch(patch)
            self.patches[well_name] = patch
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        cax = self.fig.add_axes([0.88, 0.15, 0.03, 0.7])
        self.colorbar = self.fig.colorbar(sm, cax=cax)
        self.colorbar.set_label('Deflection (nm)')
        
        # Add black section below colorbar for failure threshold
        if self.failure_threshold is not None:
            # Create a small axes below the colorbar for the black section
            black_ax = self.fig.add_axes([0.88, 0.10, 0.03, 0.04])
            black_ax.set_facecolor('black')
            black_ax.set_xticks([])
            black_ax.set_yticks([])
            # Add "Failed" label
            black_ax.text(0.5, 0.5, f'< {self.failure_threshold}', ha='center', va='center', 
                         color='white', fontsize=7, transform=black_ax.transAxes)
        
        # Time text
        # Initial time text
        if self.time_in_hours:
            initial_time_str = f'Time: {self.min_time / 60:.2f} h'
        else:
            initial_time_str = f'Time: {self.min_time:.1f} min'
        
        self.time_text = self.ax.text(
            0.02, 0.98,
            initial_time_str,
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
        
        # Slider for time
        ax_slider = self.fig.add_axes([0.15, 0.05, 0.65, 0.03])
        
        if self.time_in_hours:
            slider_label = 'Time (h)'
            slider_min = self.min_time / 60
            slider_max = self.max_time / 60
            slider_step = max(0.01, (slider_max - slider_min) / 500)
        else:
            slider_label = 'Time (min)'
            slider_min = self.min_time
            slider_max = self.max_time
            slider_step = max(0.1, (slider_max - slider_min) / 500)
        
        self.slider = Slider(
            ax_slider,
            slider_label,
            slider_min,
            slider_max,
            valinit=slider_min,
            valstep=slider_step,
        )
        
        def on_slider(val):
            # Convert slider value back to minutes if displaying in hours
            time_minutes = val * 60 if self.time_in_hours else val
            self.update_patches(time_minutes)
        
        self.slider.on_changed(on_slider)
        
        # Initial update
        self.update_patches(self.min_time)
        
        # Axis settings
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        
        plt.show()
    
    def export_html(self, output_path='deflation_overlay.html'):
        """
        Export the visualization as an interactive HTML file.
        
        The HTML file embeds all data (image, deflection curves, well positions)
        and uses JavaScript for smooth client-side interpolation. No pre-computed
        frames are needed, making the slider completely smooth.
        
        Args:
            output_path: Path to save the HTML file
        """
        import base64
        import json
        import io
        from PIL import Image
        
        # Encode image as base64
        img = Image.open(self.image_path)
        img_width, img_height = img.size
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare deflation curves data as JSON
        curves_data = {}
        for well_name, df in self.deflation_curves.items():
            times = df['Time (minutes)'].tolist()
            heights = df['Deflection (nm)'].tolist()
            # Sort by time
            sorted_pairs = sorted(zip(times, heights), key=lambda x: x[0])
            curves_data[well_name] = {
                'times': [p[0] for p in sorted_pairs],
                'heights': [p[1] for p in sorted_pairs]
            }
        
        # Prepare patch positions data
        patches_data = {
            well_name: {'cx': pos[0], 'cy': pos[1]}
            for well_name, pos in self.patch_positions.items()
        }
        
        # Generate turbo colormap as array (256 colors)
        turbo_colors = []
        for i in range(256):
            rgba = plt.cm.turbo(i / 255)
            turbo_colors.append([int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)])
        
        # Colorbar min is threshold when set, otherwise min height
        colorbar_min = self.failure_threshold if self.failure_threshold is not None else self.min_height
        
        # Configuration
        config = {
            'minTime': self.min_time,
            'maxTime': self.max_time,
            'minHeight': colorbar_min,
            'maxHeight': self.max_height,
            'patchSize': self.patch_size_px,
            'timeInHours': self.time_in_hours,
            'imgWidth': img_width,
            'imgHeight': img_height,
            'failureThreshold': self.failure_threshold,
            'failureTimes': self.failure_times,
            'whiteoutAfterLast': self.whiteout_after_last,
            'wellMaxTimes': self.well_max_times,
        }
        
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>Deflation Overlay</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5;
            min-height: 100vh;
        }
        .container { 
            max-width: 95vw; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1 { color: #333; margin: 0 0 15px 0; font-size: 24px; }
        #canvas-container { 
            position: relative; 
            width: 100%;
            display: flex;
            justify-content: center;
        }
        #overlay { 
            border: 1px solid #ccc; 
            max-width: 100%;
            max-height: 75vh;
            object-fit: contain;
        }
        .controls { 
            margin-top: 15px; 
            padding: 15px; 
            background: #f9f9f9; 
            border-radius: 4px; 
        }
        .slider-container { 
            display: flex; 
            align-items: center; 
            gap: 15px; 
        }
        .slider-container span:first-child { font-size: 16px; }
        #timeSlider { 
            flex: 1; 
            height: 24px; 
            cursor: pointer;
        }
        #timeDisplay { 
            min-width: 100px; 
            font-weight: bold; 
            font-size: 16px;
            text-align: right;
        }
        .colorbar { 
            display: flex; 
            align-items: center; 
            margin-top: 12px; 
            gap: 10px; 
            flex-wrap: wrap;
        }
        .colorbar span { font-size: 14px; }
        #colorbar-canvas { border: 1px solid #ccc; flex-shrink: 0; }
        .colorbar-label { margin-left: 5px; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Deflation Overlay</h1>
        <div id="canvas-container">
            <canvas id="overlay"></canvas>
        </div>
        <div class="controls">
            <div class="slider-container">
                <span>Time:</span>
                <input type="range" id="timeSlider" min="0" max="100" value="0" step="0.1">
                <span id="timeDisplay">0.00 h</span>
            </div>
            <div class="colorbar">
                <span id="failedLabel" style="display: none; background: black; color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;"></span>
                <span id="minHeight">0 nm</span>
                <canvas id="colorbar-canvas" width="200" height="20"></canvas>
                <span id="maxHeight">100 nm</span>
                <span class="colorbar-label">Deflection (nm)</span>
            </div>
        </div>
    </div>
    
    <script>
        // Embedded data
        const IMAGE_DATA = "IMAGE_DATA_PLACEHOLDER";
        const CURVES = CURVES_PLACEHOLDER;
        const PATCHES = PATCHES_PLACEHOLDER;
        const TURBO = TURBO_PLACEHOLDER;
        const CONFIG = CONFIG_PLACEHOLDER;
        
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const slider = document.getElementById('timeSlider');
        const timeDisplay = document.getElementById('timeDisplay');
        
        let bgImage = new Image();
        let displayScale = 1;
        
        function interpolateHeight(curveData, timeMinutes) {
            if (!curveData || curveData.times.length === 0) return null;
            
            const times = curveData.times;
            const heights = curveData.heights;
            
            if (timeMinutes < times[0]) return null;
            if (timeMinutes >= times[times.length - 1]) return heights[heights.length - 1];
            
            for (let i = 1; i < times.length; i++) {
                if (timeMinutes < times[i]) {
                    const t0 = times[i-1], t1 = times[i];
                    const h0 = heights[i-1], h1 = heights[i];
                    const frac = (timeMinutes - t0) / (t1 - t0);
                    return h0 + frac * (h1 - h0);
                }
            }
            return heights[heights.length - 1];
        }
        
        function heightToColor(height) {
            if (height === null) return 'rgba(255, 255, 255, 0.8)';
            
            const norm = (height - CONFIG.minHeight) / (CONFIG.maxHeight - CONFIG.minHeight + 1e-9);
            const idx = Math.max(0, Math.min(255, Math.floor(norm * 255)));
            const c = TURBO[idx];
            return `rgba(${c[0]}, ${c[1]}, ${c[2]}, 0.8)`;
        }
        
        function draw(timeMinutes) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
            
            const halfSize = (CONFIG.patchSize * displayScale) / 2;
            const patchSize = CONFIG.patchSize * displayScale;
            
            for (const [wellName, pos] of Object.entries(PATCHES)) {
                const curveData = CURVES[wellName];
                
                // Check if well has failed
                const failureTime = CONFIG.failureTimes ? CONFIG.failureTimes[wellName] : null;
                const hasFailed = CONFIG.failureThreshold !== null && failureTime !== null && failureTime !== undefined && timeMinutes >= failureTime;
                
                // Check if should whiteout (after last data point)
                const maxTime = CONFIG.wellMaxTimes ? CONFIG.wellMaxTimes[wellName] : null;
                const isAfterLast = CONFIG.whiteoutAfterLast && maxTime !== null && maxTime !== undefined && timeMinutes > maxTime;
                
                const cx = pos.cx * displayScale;
                const cy = pos.cy * displayScale;
                
                if (hasFailed) {
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
                    ctx.strokeStyle = 'black';
                } else if (isAfterLast) {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.strokeStyle = 'rgba(200, 200, 200, 1)';
                } else {
                    const height = interpolateHeight(curveData, timeMinutes);
                    ctx.fillStyle = heightToColor(height);
                    ctx.strokeStyle = height === null ? 'gray' : 'black';
                }
                ctx.lineWidth = 1.5;
                
                ctx.beginPath();
                ctx.rect(cx - halfSize, cy - halfSize, patchSize, patchSize);
                ctx.fill();
                ctx.stroke();
            }
            
            // Time label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.strokeStyle = 'gray';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(10, 10, 130, 30, 5);
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = 'black';
            ctx.font = '14px Arial';
            let timeText;
            if (CONFIG.timeInHours) {
                timeText = `Time: ${(timeMinutes / 60).toFixed(2)} h`;
            } else {
                timeText = `Time: ${timeMinutes.toFixed(1)} min`;
            }
            ctx.fillText(timeText, 20, 30);
        }
        
        function updateSlider() {
            const sliderVal = parseFloat(slider.value);
            let timeMinutes;
            if (CONFIG.timeInHours) {
                const hours = CONFIG.minTime/60 + (sliderVal / 100) * (CONFIG.maxTime/60 - CONFIG.minTime/60);
                timeMinutes = hours * 60;
                timeDisplay.textContent = `${hours.toFixed(2)} h`;
            } else {
                timeMinutes = CONFIG.minTime + (sliderVal / 100) * (CONFIG.maxTime - CONFIG.minTime);
                timeDisplay.textContent = `${timeMinutes.toFixed(1)} min`;
            }
            draw(timeMinutes);
        }
        
        function drawColorbar() {
            const cbCanvas = document.getElementById('colorbar-canvas');
            const cbCtx = cbCanvas.getContext('2d');
            for (let i = 0; i < 200; i++) {
                const idx = Math.floor((i / 199) * 255);
                const c = TURBO[idx];
                cbCtx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
                cbCtx.fillRect(i, 0, 1, 20);
            }
            document.getElementById('minHeight').textContent = `${CONFIG.minHeight.toFixed(1)} nm`;
            document.getElementById('maxHeight').textContent = `${CONFIG.maxHeight.toFixed(1)} nm`;
            
            // Show failed label if failure threshold is set
            if (CONFIG.failureThreshold !== null) {
                document.getElementById('failedLabel').textContent = `< ${CONFIG.failureThreshold}`;
                document.getElementById('failedLabel').style.display = 'inline';
            }
        }
        
        function resizeCanvas() {
            const container = document.getElementById('canvas-container');
            const maxWidth = container.clientWidth - 20;
            const maxHeight = window.innerHeight * 0.75;
            
            const aspectRatio = CONFIG.imgWidth / CONFIG.imgHeight;
            
            let displayWidth, displayHeight;
            if (maxWidth / aspectRatio <= maxHeight) {
                displayWidth = maxWidth;
                displayHeight = maxWidth / aspectRatio;
            } else {
                displayHeight = maxHeight;
                displayWidth = maxHeight * aspectRatio;
            }
            
            displayScale = displayWidth / CONFIG.imgWidth;
            canvas.width = displayWidth;
            canvas.height = displayHeight;
            
            updateSlider();
        }
        
        bgImage.onload = function() {
            drawColorbar();
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        };
        
        slider.addEventListener('input', updateSlider);
        bgImage.src = 'data:image/png;base64,' + IMAGE_DATA;
    </script>
</body>
</html>'''
        
        # Replace placeholders
        html_content = html_template.replace('IMAGE_DATA_PLACEHOLDER', img_base64)
        html_content = html_content.replace('CURVES_PLACEHOLDER', json.dumps(curves_data))
        html_content = html_content.replace('PATCHES_PLACEHOLDER', json.dumps(patches_data))
        html_content = html_content.replace('TURBO_PLACEHOLDER', json.dumps(turbo_colors))
        html_content = html_content.replace('CONFIG_PLACEHOLDER', json.dumps(config))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Exported interactive HTML to: {output_path}")


# Example usage / script entry point
if __name__ == "__main__":
    import membrane_relative_positions as mrp
    
    # Get well map based on path_loader configuration
    sample_ID = pl.sample_ID
    location = pl.transfer_location
    
    # Select well map based on sample/location
    well_map = mrp.sample_ID_and_location_to_well_map.get((sample_ID, location))
    if well_map is None:
        print(f"No well map configured for sample {sample_ID} at location {location}")
        exit(1)
    
    # Load deflation curves
    curves = load_deflation_curves(cutoff_time=1000)
    
    if not curves:
        print("No deflation curves found. Please check path_loader configuration.")
        exit(1)
    
    print(f"Loaded {len(curves)} deflation curves: {list(curves.keys())}")
    
    # Configuration
    image_path = 'sample53_optical.jpg'
    
    anchor_well = '(8, 8)'
    anchor_x = 1250
    anchor_y = 1177
    patch_size = 60
    x_trans = 144
    y_trans = 85

    
    # Create and show overlay
    overlay = DeflationOverlay(
        image_path=image_path,
        deflation_curves=curves,
        well_map=well_map,
        anchor_well=anchor_well,
        anchor_pixel_x=anchor_x,
        anchor_pixel_y=anchor_y,
        patch_size_px=patch_size,
        x_translation_px=x_trans,
        y_translation_px=y_trans,
        time_in_hours=True,
        # failure_threshold=20,
        whiteout_after_last=True,
    )
    
    overlay.show()
    overlay.export_html('deflation_overlay.html')
