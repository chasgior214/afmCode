import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def height_and_defln(image, line_height, slope = None):
    scan_size = image.get_scan_size()

    height_retrace = image.get_height_retrace()
    contrast_map = image.get_amplitude_retrace()

    x_pixel_count = height_retrace.shape[1]
    y_pixel_count = height_retrace.shape[0]
    pixel_size = scan_size / x_pixel_count # microns per pixel

    x = np.linspace(0, scan_size, x_pixel_count) # x-coordinates in microns
    y = np.linspace(0, scan_size, y_pixel_count) # y-coordinates in microns
    extent = (0, scan_size, 0, scan_size * y_pixel_count / x_pixel_count)

    y_pixels= np.arange(0, y_pixel_count)
    # find the nearest y pixel in y_pixels to the line height, remembering that the y values decrease to zero as the pixel number increases to y_pixel_count
    nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for sliders

    # Initial plots
    im = ax1.imshow(contrast_map, cmap='grey', extent=extent)
    ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title(f"Contrast Map")
    ax1.set_ylabel("y (μm)")

    # Calculate the initial aspect ratio of ax1
    height, width = contrast_map.shape
    aspect_ratio = height / width
    toplot = height_retrace[nearest_y_to_plot, :]
    # If height_retrace is slanted correct for this by subtracting the slope of the line from the height_retrace
    if slope:
        toplot = toplot - slope * x

    ax2.plot(x, toplot)
    ax2.set_title(f"Height at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Set the same fixed aspect ratio for both axes
    ax1.set_box_aspect(aspect_ratio)
    ax2.set_box_aspect(aspect_ratio)
    
    # Create sliders for vmin and vmax
    ax_vmin = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor='lightgrey')
    ax_vmax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgrey')
    slider_vmin = Slider(ax_vmin, 'vmin (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=0)
    slider_vmax = Slider(ax_vmax, 'vmax (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=1e9)

    # Update function for the sliders
    def update(val):
        im.set_clim(vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)

    plt.show()

def height_and_defln_row_selector(image, initial_line_height=0):
    scan_size = image.get_scan_size()
    height_retrace = image.get_height_retrace()
    contrast_map = image.get_amplitude_retrace()

    x_pixel_count = height_retrace.shape[1]
    y_pixel_count = height_retrace.shape[0]
    pixel_size = scan_size / x_pixel_count  # microns per pixel

    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(0, scan_size, y_pixel_count)  # y-coordinates in microns
    extent = (0, scan_size, 0, scan_size * y_pixel_count / x_pixel_count)

    line_height = initial_line_height
    y_pixels = np.arange(0, y_pixel_count)
    nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

    selected_points = []  # To store the two right-clicked points
    cumulative_adjusted_height = None  # To store the cumulative adjusted height retrace
    selected_heights = []  # List to store the selected height values

    def update_plots(event):
        nonlocal line_height, nearest_y_to_plot, selected_points, cumulative_adjusted_height, selected_heights

        # Check if the event is triggered during zooming or panning
        if plt.get_current_fig_manager().toolbar.mode != '':
            return  # Ignore events during zoom or pan

        if event.inaxes == ax1 and event.button == 1:  # Left-click on the top plot
            line_height = float(event.ydata)
            if line_height < 0 or line_height >= contrast_map.shape[0]:
                return  # Ignore clicks outside the valid range
            nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

            # Reset cumulative_adjusted_height for the new y location
            cumulative_adjusted_height = height_retrace[nearest_y_to_plot, :].copy()

            # Clear the existing plots
            ax1.cla()
            ax2.cla()

            # Redraw the upper plot
            ax1.imshow(contrast_map, cmap='grey', extent=extent, vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
            ax1.axhline(y=line_height, color='r', linestyle='--')
            ax1.set_title("Contrast Map")
            ax1.set_ylabel("y (μm)")

            # Redraw the lower plot
            ax2.plot(x, cumulative_adjusted_height)
            ax2.set_xlim(0, scan_size)  # Ensure x-axis matches upper plot
            ax2.set_title(f"Height at y = {round(line_height, 3)} μm")
            ax2.set_xlabel("x (μm)")
            ax2.set_ylabel("Height (nm)")

            # Refresh the figure
            fig.canvas.draw()
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
                    cumulative_adjusted_height = height_retrace[nearest_y_to_plot, :].copy()
                cumulative_adjusted_height -= slope * x

                # Update the lower plot
                ax2.cla()
                ax2.plot(x, cumulative_adjusted_height)
                ax2.set_xlim(0, scan_size)  # Ensure x-axis matches upper plot
                ax2.set_title(f"Height at y = {round(line_height, 3)} μm (Slope Corrected)")
                ax2.set_xlabel("x (μm)")
                ax2.set_ylabel("Height (nm)")

                fig.canvas.draw()

                # Reset selected_points to allow further adjustments
                selected_points = []
            return

        if event.inaxes == ax2 and event.button == 1:  # Left-click on the bottom plot
            selected_heights.append(event.ydata)  # Store the height value
            if len(selected_heights) > 2:
                selected_heights.pop(0)  # Keep only the two most recent heights
            print(f"Height value at x = {event.xdata:.3f} μm: {event.ydata:.3f} nm")
            return

    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for sliders

    # Initial plots
    im = ax1.imshow(contrast_map, cmap='grey', extent=extent)
    ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title("Contrast Map")
    ax1.set_ylabel("y (μm)")

    # Calculate the initial aspect ratio of ax1
    height, width = contrast_map.shape
    aspect_ratio = height / width

    ax2.plot(x, height_retrace[nearest_y_to_plot, :])
    ax2.set_xlim(0, scan_size)  # Ensure x-axis matches upper plot
    ax2.set_title(f"Height at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Set the same fixed aspect ratio for both axes
    ax1.set_box_aspect(aspect_ratio)
    ax2.set_box_aspect(aspect_ratio)

    # Create sliders for vmin and vmax
    ax_vmin = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor='lightgrey')
    ax_vmax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgrey')
    slider_vmin = Slider(ax_vmin, 'vmin (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=0)
    slider_vmax = Slider(ax_vmax, 'vmax (x10^9)', np.min(contrast_map) * 1e9, np.max(contrast_map) * 1e9, valinit=1e9)

    # Update function for the sliders
    def update_slider(val):
        # Update the contrast of the contrast map
        im.set_clim(vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    slider_vmin.on_changed(update_slider)
    slider_vmax.on_changed(update_slider)

    # Ensure sliders remain functional after height selection
    fig.canvas.mpl_connect('button_press_event', update_plots)

    from matplotlib.widgets import Button
    ax_button = plt.axes([0.82, 0.05, 0.15, 0.05])
    btn_set_max = Button(ax_button, 'Set Max Height')

    # Add Set Min Height button
    ax_button_min = plt.axes([0.65, 0.05, 0.15, 0.05])
    btn_set_min = Button(ax_button_min, 'Set Min Height')

    def set_max_height(event):
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_retrace[nearest_y_to_plot, :]
        max_idx = np.argmax(ydata)
        max_x = x[max_idx]
        max_y = ydata[max_idx]
        selected_heights.append(max_y)  # Store the height value
        if len(selected_heights) > 2:
            selected_heights.pop(0)  # Keep only the two most recent heights
        print(f"Max height at x = {max_x:.3f} μm: {max_y:.3f} nm (Set by button)")

    def set_min_height(event):
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_retrace[nearest_y_to_plot, :]
        min_idx = np.argmin(ydata)
        min_x = x[min_idx]
        min_y = ydata[min_idx]
        selected_heights.append(min_y)  # Store the height value
        if len(selected_heights) > 2:
            selected_heights.pop(0)  # Keep only the two most recent heights
        print(f"Min height at x = {min_x:.3f} μm: {min_y:.3f} nm (Set by button)")

    btn_set_max.on_clicked(set_max_height)
    btn_set_min.on_clicked(set_min_height)
    fig.canvas.manager.set_window_title(str(image.bname) + image.get_datetime().strftime(" %Y-%m-%d %H:%M:%S"))

    plt.show()

    if len(selected_heights) == 1:
        return selected_heights[0]  # Return the single selected height
    elif len(selected_heights) == 2:
        return selected_heights  # Return the two most recent selected heights

def export_heightmap_3d_surface(image):
    scan_size = image.get_scan_size()
    height_retrace = image.get_height_retrace()

    x_pixel_count = height_retrace.shape[1]
    y_pixel_count = height_retrace.shape[0]
    
    # Calculate y dimension based on aspect ratio
    y_dimension = scan_size * y_pixel_count / x_pixel_count

    # Generate x and y coordinates with correct dimensions
    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(0, y_dimension, y_pixel_count)  # y-coordinates in microns
    X, Y = np.meshgrid(x, y)

    # Use Plotly for the interactive 3D plot
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Surface(z=height_retrace, x=X, y=Y, colorscale='Viridis')])
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
