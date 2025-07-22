import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def select_heights(image, initial_line_height=0):
    # Extract data from the image
    image_bname = str(image.bname)
    image_save_date_time = image.get_datetime().strftime("%Y-%m-%d %H:%M:%S")

    scan_size = image.get_scan_size()
    scan_direction = image.get_scan_direction()
    scan_rate = image.get_scan_rate()

    height_map = image.get_height_retrace()
    contrast_map = image.get_contrast_retrace()
    phase_map = image.get_phase_retrace()

    # Calculate pixel size
    x_pixel_count = height_map.shape[1]
    y_pixel_count = height_map.shape[0]
    pixel_size = scan_size / x_pixel_count  # microns per pixel

    imaging_duration = y_pixel_count / scan_rate

    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(0, scan_size, y_pixel_count)  # y-coordinates in microns
    extent = (0, scan_size, 0, scan_size * y_pixel_count / x_pixel_count)

    line_height = initial_line_height
    y_pixels = np.arange(0, y_pixel_count)
    nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

    selected_points = []  # To store the two right-clicked points
    cumulative_adjusted_height = None  # To store the cumulative adjusted height retrace
    selected_heights = []  # List to store the selected height values
    selected_heights_info = []  # (height, x, y)
    time_since_start = None  # To store the time since the start of the scan

    def update_stats_display():
        """Refresh the text panel summarizing selection info."""
        ax4.cla()
        ax4.axis('off')
        lines = []
        if time_since_start is not None:
            lines.append(f"Line imaged {time_since_start:.2f} seconds after imaging began")

        for idx, (h, xh, yh) in enumerate(selected_heights_info[-2:][::-1], 1):
            lines.append(f"Selected {idx}: {h:.3f} nm at ({xh:.3f}, {yh:.3f}) μm")

        if len(selected_heights_info) == 2:
            diff = selected_heights_info[-1][0] - selected_heights_info[-2][0]
            lines.append(f"Difference: {diff:.3f} nm")

        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        max_idx = np.argmax(ydata)
        min_idx = np.argmin(ydata)
        lines.append(f"Max cross-section: {ydata[max_idx]:.3f} nm at x={x[max_idx]:.3f} μm")
        lines.append(f"Min cross-section: {ydata[min_idx]:.3f} nm at x={x[min_idx]:.3f} μm")

        ax4.text(0.05, 0.95, '\n'.join(lines), va='top')
        fig.canvas.draw_idle()

    def update_plots(event):
        nonlocal line_height, nearest_y_to_plot, selected_points, cumulative_adjusted_height
        nonlocal selected_heights, selected_heights_info, time_since_start
        nonlocal im, im_phase, hline_contrast, hline_phase, cross_line

        # Check if the event is triggered during zooming or panning
        if plt.get_current_fig_manager().toolbar.mode != '':
            return  # Ignore events during zoom or pan

        if event.inaxes in (ax1, ax3) and event.button == 1:  # Left-click on either image
            line_height = float(event.ydata)

            if line_height < 0 or line_height >= contrast_map.shape[0]:
                return  # Ignore clicks outside the valid range

            nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

            # Calculate time between start of scan and when the selected line was imaged
            ratio = nearest_y_to_plot / y_pixel_count
            if scan_direction == 1: # Scanning down
                time_since_start = ratio * imaging_duration
            else: # Scanning up
                time_since_start = (1 - ratio) * imaging_duration

            # Reset cumulative_adjusted_height for the new y location
            cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()

            # Update the horizontal indicator lines
            hline_contrast.set_ydata([line_height, line_height])
            if hline_phase is not None:
                hline_phase.set_ydata([line_height, line_height])

            # Update the lower plot without clearing the axis
            cross_line.set_ydata(cumulative_adjusted_height)
            ax2.relim()
            ax2.autoscale_view()
            ax2.set_title(f"Height at y = {round(line_height, 3)} μm")
            update_stats_display()

            # Refresh the figure
            fig.canvas.draw_idle()
            return

        if event.inaxes == ax2 and event.button == 3:  # Right-click on the bottom plot
            selected_points.append((event.xdata, event.ydata))
            if len(selected_points) == 2:
                # Calculate the slope based on the two selected points
                x1, y1 = selected_points[0]
                x2, y2 = selected_points[1]
                slope = (y2 - y1) / (x2 - x1)

                # Adjust the displayed height retrace using the slope
                if cumulative_adjusted_height is None:
                    cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()
                cumulative_adjusted_height -= slope * x

                # Update the lower plot without clearing
                cross_line.set_ydata(cumulative_adjusted_height)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_title(f"Height at y = {round(line_height, 3)} μm (Slope Corrected)")

                update_stats_display()
                fig.canvas.draw_idle()

                # Reset selected_points to allow further adjustments
                selected_points = []
            return

        if event.inaxes == ax2 and event.button == 1:  # Left-click on the bottom plot
            selected_heights.append(event.ydata)  # Store the height value
            selected_heights_info.append((event.ydata, event.xdata, line_height))
            if len(selected_heights) > 2:
                selected_heights.pop(0)  # Keep only the two most recent heights
            if len(selected_heights_info) > 2:
                selected_heights_info.pop(0)
            print(f"Height value at x = {event.xdata:.3f} μm: {event.ydata:.3f} nm")
            update_stats_display()
            return

    # Create the figure and axes in a 2x2 grid
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for sliders

    # Maximize the window if possible
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except Exception:
        try:
            manager.window.state('zoomed')
        except Exception:
            pass

    # Initial plots
    im = ax1.imshow(contrast_map, cmap='grey', extent=extent)
    hline_contrast = ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title("Contrast Map")
    ax1.set_ylabel("y (μm)")

    if phase_map is not None:
        im_phase = ax3.imshow(phase_map, cmap='grey', extent=extent,
                              vmin=np.min(phase_map), vmax=np.max(phase_map))
        hline_phase = ax3.axhline(y=line_height, color='r', linestyle='--')
        ax3.set_title("Phase Retrace")
        ax3.set_ylabel("y (μm)")
    else:
        im_phase = None
        hline_phase = None
        ax3.axis('off')

    # Calculate the initial aspect ratio of ax1
    height, width = contrast_map.shape
    aspect_ratio = height / width

    cross_line, = ax2.plot(x, height_map[nearest_y_to_plot, :])
    ax2.set_xlim(0, scan_size)  # Ensure x-axis matches upper plot
    ax2.set_title(f"Height at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Set the same fixed aspect ratio for image axes
    ax1.set_box_aspect(aspect_ratio)
    ax2.set_box_aspect(aspect_ratio)
    if phase_map is not None:
        ax3.set_box_aspect(aspect_ratio)

    ax4.axis('off')
    update_stats_display()

    # Create sliders for vmin and vmax of contrast map
    ax_vmin = plt.axes([0.1, 0.15, 0.35, 0.03], facecolor='lightgrey')
    ax_vmax = plt.axes([0.1, 0.1, 0.35, 0.03], facecolor='lightgrey')
    slider_vmin = Slider(ax_vmin, 'vmin (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=0)
    slider_vmax = Slider(ax_vmax, 'vmax (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=1e9)

    if phase_map is not None:
        ax_phase_vmin = plt.axes([0.55, 0.15, 0.35, 0.03], facecolor='lightgrey')
        ax_phase_vmax = plt.axes([0.55, 0.1, 0.35, 0.03], facecolor='lightgrey')
        slider_phase_vmin = Slider(ax_phase_vmin, 'phase vmin', np.min(phase_map), np.max(phase_map), valinit=np.min(phase_map))
        slider_phase_vmax = Slider(ax_phase_vmax, 'phase vmax', np.min(phase_map), np.max(phase_map), valinit=np.max(phase_map))

    # Update function for the sliders
    def update_slider(val):
        im.set_clim(vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
        if phase_map is not None and im_phase is not None:
            im_phase.set_clim(vmin=slider_phase_vmin.val, vmax=slider_phase_vmax.val)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    slider_vmin.on_changed(update_slider)
    slider_vmax.on_changed(update_slider)
    if phase_map is not None:
        slider_phase_vmin.on_changed(update_slider)
        slider_phase_vmax.on_changed(update_slider)

    # Ensure sliders remain functional after height selection
    fig.canvas.mpl_connect('button_press_event', update_plots)

    from matplotlib.widgets import Button
    ax_button = plt.axes([0.82, 0.05, 0.15, 0.05])
    btn_set_max = Button(ax_button, 'Set Max Height')

    # Add Set Min Height button
    ax_button_min = plt.axes([0.65, 0.05, 0.15, 0.05])
    btn_set_min = Button(ax_button_min, 'Set Min Height')

    def set_max_height(event):
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        max_idx = np.argmax(ydata)
        max_x = x[max_idx]
        max_y = ydata[max_idx]
        selected_heights.append(max_y)
        selected_heights_info.append((max_y, max_x, line_height))
        if len(selected_heights) > 2:
            selected_heights.pop(0)
        if len(selected_heights_info) > 2:
            selected_heights_info.pop(0)
        print(f"Max height at x = {max_x:.3f} μm: {max_y:.3f} nm (Set by button)")
        update_stats_display()

    def set_min_height(event):
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        min_idx = np.argmin(ydata)
        min_x = x[min_idx]
        min_y = ydata[min_idx]
        selected_heights.append(min_y)
        selected_heights_info.append((min_y, min_x, line_height))
        if len(selected_heights) > 2:
            selected_heights.pop(0)
        if len(selected_heights_info) > 2:
            selected_heights_info.pop(0)
        print(f"Min height at x = {min_x:.3f} μm: {min_y:.3f} nm (Set by button)")
        update_stats_display()

    btn_set_max.on_clicked(set_max_height)
    btn_set_min.on_clicked(set_min_height)

    # Override toolbar home to always rescale the cross-section
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar is not None and hasattr(toolbar, "home"):
        orig_home = toolbar.home

        def _home(*args, **kwargs):
            orig_home(*args, **kwargs)
            ax2.relim()
            ax2.autoscale_view()
            fig.canvas.draw_idle()

        toolbar.home = _home

    fig.canvas.manager.set_window_title(image_bname + " - " + image_save_date_time)

    plt.show()

    if len(selected_heights) == 1:
        return selected_heights[0], time_since_start  # Return the single selected height and time since start
    elif len(selected_heights) == 2:
        return selected_heights[0], selected_heights[1], time_since_start  # Return the two most recent selected heights and time since start

def export_heightmap_3d_surface(image):
    scan_size = image.get_scan_size()
    height_map = image.get_height_retrace()

    x_pixel_count = height_map.shape[1]
    y_pixel_count = height_map.shape[0]
    
    # Calculate y dimension based on aspect ratio
    y_dimension = scan_size * y_pixel_count / x_pixel_count

    # Generate x and y coordinates with correct dimensions
    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(0, y_dimension, y_pixel_count)  # y-coordinates in microns
    X, Y = np.meshgrid(x, y)

    # Use Plotly for the interactive 3D plot
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Surface(z=height_map, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title="Height Retrace (Entire Image)",
        scene=dict(
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            zaxis_title="Height (nm)",
            aspectmode='data'  # Maintain the data's aspect ratio
        )
    )
    
    # Save as an interactive HTML file
    fig.write_html("3d_plot.html")
