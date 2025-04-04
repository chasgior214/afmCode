import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def height_and_defln(image, line_height, slope = (-4113.2 - (-4036.1)) / (9.08 - 0.76)):
    scan_size = image.get_scan_size()

    height_retrace = image.get_height_retrace()
    deflection_retrace = image.get_amplitude_retrace()

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
    im = ax1.imshow(deflection_retrace, cmap='grey', extent=extent)
    ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title(f"Deflection Retrace")
    ax1.set_ylabel("y (μm)")

    # Calculate the initial aspect ratio of ax1
    height, width = deflection_retrace.shape
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
    slider_vmin = Slider(ax_vmin, 'vmin (x10^9)', np.min(deflection_retrace) * 1e9, np.max(deflection_retrace) * 1e9, valinit=0)
    slider_vmax = Slider(ax_vmax, 'vmax (x10^9)', np.min(deflection_retrace) * 1e9, np.max(deflection_retrace) * 1e9, valinit=1e9)

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
    deflection_retrace = image.get_amplitude_retrace()

    x_pixel_count = height_retrace.shape[1]
    y_pixel_count = height_retrace.shape[0]
    pixel_size = scan_size / x_pixel_count  # microns per pixel

    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(0, scan_size, y_pixel_count)  # y-coordinates in microns
    extent = (0, scan_size, 0, scan_size * y_pixel_count / x_pixel_count)

    line_height = initial_line_height
    y_pixels = np.arange(0, y_pixel_count)
    # Find the nearest y pixel in y_pixels to the line height
    nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

    def update_plots(event):
        nonlocal line_height, nearest_y_to_plot

        # Check if the click event occurred within the bounds of ax1
        if event.inaxes != ax1:
            return  # Ignore clicks outside ax1

        # Check if zoom or pan mode is active
        if fig.canvas.toolbar.mode:
            # Synchronize the x-axis of ax2 with ax1.
            xlim = ax1.get_xlim()  # Get the x-axis limits from ax1
            ax2.set_xlim(xlim)  # Apply the same x-axis limits to ax2

            # Dynamically scale the y-axis of ax2 based on the new x-axis limits
            x_min, x_max = xlim
            indices = np.where((x >= x_min) & (x <= x_max))[0]
            if len(indices) > 0:
                y_min = np.min(height_retrace[nearest_y_to_plot, indices]) * 1.1
                y_max = np.max(height_retrace[nearest_y_to_plot, indices]) * 1.1
                ax2.set_ylim(y_min, y_max)  # Adjust the y-axis limits

            fig.canvas.draw_idle()  # Redraw the figure
            return  # Ignore updates during zoom or pan

        # Get the y-coordinate from the click event and update the line_height variable
        line_height = float(event.ydata)
        if line_height < 0 or line_height >= deflection_retrace.shape[0]:
            return  # Ignore clicks outside the valid range
        nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))

        # Clear the existing plots
        ax1.cla()
        ax2.cla()

        # Redraw the upper plot
        ax1.imshow(deflection_retrace, cmap='grey', extent=extent, vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
        ax1.axhline(y=line_height, color='r', linestyle='--')
        ax1.set_title("Deflection Retrace")
        ax1.set_ylabel("y (μm)")

        # Redraw the lower plot
        ax2.plot(x, height_retrace[nearest_y_to_plot, :])
        ax2.set_title(f"Height at y = {round(line_height, 3)} μm")
        ax2.set_xlabel("x (μm)")
        ax2.set_ylabel("Height (nm)")

        # Refresh the figure
        fig.canvas.draw()

    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to make space for sliders

    # Initial plots
    im = ax1.imshow(deflection_retrace, cmap='grey', extent=extent)
    ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title("Deflection Retrace")
    ax1.set_ylabel("y (μm)")

    # Calculate the initial aspect ratio of ax1
    height, width = deflection_retrace.shape
    aspect_ratio = height / width

    ax2.plot(x, height_retrace[nearest_y_to_plot, :])
    ax2.set_title(f"Height at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Set the same fixed aspect ratio for both axes
    ax1.set_box_aspect(aspect_ratio)
    ax2.set_box_aspect(aspect_ratio)

    # Create sliders for vmin and vmax
    ax_vmin = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor='lightgrey')
    ax_vmax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgrey')
    slider_vmin = Slider(ax_vmin, 'vmin (x10^9)', np.min(deflection_retrace) * 1e9, np.max(deflection_retrace) * 1e9, valinit=0)
    slider_vmax = Slider(ax_vmax, 'vmax (x10^9)', np.min(deflection_retrace) * 1e9, np.max(deflection_retrace) * 1e9, valinit=1e9)

    # Update function for the sliders
    def update_slider(val):
        im.set_clim(vmin=slider_vmin.val / 1e9, vmax=slider_vmax.val / 1e9)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    slider_vmin.on_changed(update_slider)
    slider_vmax.on_changed(update_slider)

    # Connect the click event to the update function
    fig.canvas.mpl_connect('button_press_event', update_plots)

    plt.show()

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

if __name__ == '__main__':
    from AFMImageCollection import AFMImageCollection
    import os
    folder_path = "C:/Users/chasg/afmCode/DataFolder" # Erfan's is D:/afmCode/DataFolder
    collection = AFMImageCollection(folder_path)

    for i in range(collection.get_number_of_images()):
        image = collection.get_image(i)
        # height_and_defln(image, 6.5)
        height_and_defln_row_selector(image)
        # break