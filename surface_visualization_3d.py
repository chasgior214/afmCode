import numpy as np
import plotly.graph_objects as go

def make_heightmap_3d_surface(image, view = False, save = False, cmap = 'turbo', points_list = None):
    imaging_mode = image.get_imaging_mode()
    scan_size = image.get_scan_size()
    if imaging_mode == 'AC Mode' and image.wave_data.shape[2] > 4:
        height_map = image.get_flat_height_retrace()
        title_prefix = "Flattened Height"
    elif imaging_mode == 'Contact' and image.wave_data.shape[2] > 3:
        height_map = image.get_flat_height_retrace()
        title_prefix = "Flattened Height"
    else:
        height_map = image.get_height_retrace()
        title_prefix = "Height Map"

    x_pixel_count, y_pixel_count = image.get_x_y_pixel_counts()

    # Calculate y dimension based on aspect ratio
    _, y_dimension = image.get_x_y_size()

    # Generate x and y coordinates with correct dimensions
    x = np.linspace(0, scan_size, x_pixel_count)  # x-coordinates in microns
    y = np.linspace(y_dimension, 0, y_pixel_count)  # y-coordinates in microns
    X, Y = np.meshgrid(x, y)

    # Use Plotly for the interactive 3D plot
    fig = go.Figure(data=[go.Surface(z=height_map, x=X, y=Y, colorscale=cmap)])
    
    # Scale axes
    x_extent = x.max() - x.min()
    y_extent = y.max() - y.min()
    xy_diagonal = np.sqrt(x_extent**2 + y_extent**2)

    camera = dict(
        eye=dict(
            x = -xy_diagonal,
            y = -xy_diagonal,
            z = 0.8 * xy_diagonal
        )
    )

    fig.update_layout(
        title=f"{title_prefix}",
        scene=dict(
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            zaxis_title="Height (nm)",
            aspectmode='manual',
            aspectratio=dict(x=x_extent,y=y_extent,z=1)
        ),
        scene_camera=camera
    )

    if points_list:
        for point in points_list:
            fig.add_trace(
                go.Scatter3d(
                    x=point[0],
                    y=point[1],
                    z=point[2],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="red",
                        symbol="circle"
                    ),
                    name="points"
                )
            )

    if save:
        fig.write_html("3d_plot.html")
    if view:
        fig.show()