import numpy as np
import plotly.graph_objects as go

def make_heightmap_3d_surface(image, view = False, save = False, save_filename = '3d_plot.html', cmap = 'turbo', points_list = None, paraboloids = None):
    scan_size = image.get_scan_size()
    height_map = image.get_flat_height_retrace()
    title_prefix = "Flattened Height"
    if height_map is None:
        height_map = image.get_height_retrace()
        title_prefix = "Height Map"

    z_min = np.nanmin(height_map)
    z_max = np.nanmax(height_map) * 1.01 if np.nanmax(height_map) > 0 else np.nanmax(height_map) * 0.99

    x_pixel_count, y_pixel_count = image.get_x_y_pixel_counts()

    # Calculate y dimension based on aspect ratio
    _, y_dimension = image.get_x_y_size()

    # Generate x and y coordinates with correct dimensions
    x = image.get_x_pixel_coords()  # x-coordinates in microns
    y = image.get_y_pixel_coords()  # y-coordinates in microns
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
            zaxis=dict(title="Height (nm)", range=[z_min, z_max]),
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

    if paraboloids:
        for i, coeffs in enumerate(paraboloids):
            # z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
            a = coeffs['a']
            b = coeffs['b']
            c = coeffs['c']
            d = coeffs['d']
            e = coeffs['e']
            f = coeffs['f']
            
            Z_paraboloid = a * X**2 + b * Y**2 + c * X * Y + d * X + e * Y + f
            
            fig.add_trace(
                go.Surface(
                    z=Z_paraboloid,
                    x=X,
                    y=Y,
                    colorscale='Viridis', 
                    opacity=0.5,
                    showscale=False,
                    name=f"Paraboloid {i+1}"
                )
            )

    if save:
        fig.write_html(save_filename)
    if view:
        fig.show()


if __name__ == "__main__":
    import AFMImage
    import os
    import visualizations as vis
    import surface_analysis as sa
    current_path = os.path.dirname(os.path.abspath(__file__))
    image = AFMImage.AFMImage('Image0008.ibw')

    res = vis.select_heights(image)
    slots = res.get('selected_slots', [None, None])
    extremum = slots[1]
    extremum_z = extremum[0]
    extremum_x = extremum[1]
    extremum_y = extremum[2]
    extremum_x_idx = extremum[3]
    extremum_y_idx = extremum[4]

    paraboloid_fit_result = sa.fit_paraboloid(image.get_flat_height_retrace(), extremum_x_idx, extremum_y_idx, 1, image.get_pixel_size())
    coefficients = paraboloid_fit_result['coefficients']

    print(f"Intersection area: {sa.paraboloid_substrate_intersection_area(coefficients['a'], coefficients['b'], coefficients['c'], coefficients['d'], coefficients['e'], coefficients['f'], slots[0][0])} um^2")

    res2 = vis.select_heights(image)
    slots2 = res2.get('selected_slots', [None, None])
    extremum2 = slots2[1]
    extremum2_z = extremum2[0]
    extremum2_x = extremum2[1]
    extremum2_y = extremum2[2]

    paraboloid_fit_result2 = sa.fit_paraboloid(image.get_flat_height_retrace(), extremum2[3], extremum2[4], 1, image.get_pixel_size())
    coefficients2 = paraboloid_fit_result2['coefficients']

    print(f"Intersection area: {sa.paraboloid_substrate_intersection_area(coefficients2['a'], coefficients2['b'], coefficients2['c'], coefficients2['d'], coefficients2['e'], coefficients2['f'], slots2[0][0])} um^2")

    make_heightmap_3d_surface(image, view = True, save = False, points_list=[[[extremum_x],[extremum_y],[extremum_z]], [[extremum2_x],[extremum2_y],[extremum2_z]]], paraboloids=[coefficients, coefficients2])