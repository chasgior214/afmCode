import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import patches
from gui_components import DualHandleSlider
from surface_analysis import fit_paraboloid, compute_extremum_square_info, iterative_paraboloid_fit, calculate_substrate_height

def select_heights(image, initial_line_height=0, initial_selected_slots=None):
    # Extract data from the image
    image_bname = str(image.bname)
    scan_end_date_time = image.get_scan_end_datetime().strftime("%Y-%m-%d %H:%M:%S")

    scan_rate = image.get_scan_rate()

    # Determine which height data to use. If flattened data is available use it
    # and update the title prefix accordingly.
    height_map = image.get_flat_height_retrace()
    title_prefix = "Flattened Height"
    if height_map is None:
        height_map = image.get_height_retrace()
        title_prefix = "Raw Height"
    contrast_map = image.get_contrast_map()
    if image.get_imaging_mode() == 'AC Mode':
        phase_map = image.get_phase_retrace()
    else:
        phase_map = None

    # Calculate pixel size
    x_pixel_count, y_pixel_count = image.get_x_y_pixel_counts()
    pixel_size = image.get_pixel_size()  # microns per pixel

    imaging_duration = image.get_imaging_duration()

    x_dimension, y_dimension = image.get_x_y_size() # x, y dimensions in microns
    extent = (0, x_dimension, 0, y_dimension)

    x = image.get_x_pixel_coords()  # x-coordinates of pixel centers in microns

    line_height = initial_line_height
    y_pixels = np.arange(0, y_pixel_count)
    nearest_y_to_plot = y_pixel_count - 1 - min(y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height))


    selected_points = []  # To store the two right-clicked points
    cumulative_adjusted_height = None  # To store the cumulative adjusted height cross-section

    # Slots for the two selectable height values
    selected_slots = [None, None]  # [(height, x, y), ...]
    slot_labels = ['Substrate', 'Extremum']

    # Handles for vertical and horizontal indicator lines for selected heights
    selected_vlines = [[], []]  # [[ax1_line, ax2_line, ax3_line?], ...]
    selected_cross_hlines = [None, None]  # [ax2_line, ...]
    selected_image_hlines = [[], []]  # [[image axis lines...], ...]
    square_marker_artists = []  # Markers highlighting the local max near the extremum
    paraboloid_marker_artists = []  # White + markers for paraboloid vertices
    paraboloid_circle_artists = []  # Dotted circles showing fit regions
    next_slot = 0  # Which slot to overwrite next
    locked_slot = None  # Slot index that should not be overwritten

    last_input = None  # Track the last user action for double-press logic
    last_zoom_button = None # Track which mouse button was used for the last zoom action

    time_since_start = None  # To store the time since the start of the scan
    aborted = False  # Set True when user presses Tab to cancel/exit

    image_axes = []  # Filled after axes are created; used for syncing/indicators
    paraboloid_window_um = 1.0  # Default diameter in microns for circular fit region
    paraboloid_fit_info = None
    paraboloid_slider = None
    paraboloid_vertex_text = None
    paraboloid_r2_text = None
    paraboloid_button = None

    def _clear_paraboloid_artists():
        nonlocal paraboloid_marker_artists, paraboloid_circle_artists
        for artist in paraboloid_marker_artists + paraboloid_circle_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        paraboloid_marker_artists = []
        paraboloid_circle_artists = []

    def _update_paraboloid_panel(info):
        if paraboloid_vertex_text is None or paraboloid_r2_text is None:
            return
        if info is None:
            paraboloid_vertex_text.set_text("Paraboloid fit vertex: --, --, --")
            paraboloid_r2_text.set_text("Paraboloid fit R^2: --")
        else:
            vx = info['vertex_x']
            vy = info['vertex_y']
            vz = info['vertex_z']
            paraboloid_vertex_text.set_text(
                f"Paraboloid fit vertex: {vx:.3f} μm, {vy:.3f} μm, {vz:.3f} nm"
            )
            paraboloid_r2_text.set_text(f"Paraboloid fit R^2: {info['r2']:.4f}")

    def _update_paraboloid_artists(info):
        _clear_paraboloid_artists()
        if info is None:
            return
        for target_ax in image_axes:
            marker, = target_ax.plot(
                info['vertex_x'],
                info['vertex_y'],
                marker='+',
                color='white',
                markersize=8,
                mew=2,
            )
            paraboloid_marker_artists.append(marker)
            circle = patches.Circle(
                (info['center_x'], info['center_y']),
                info['radius'],
                linewidth=1.2,
                edgecolor='white',
                linestyle='--',
                fill=False,
            )
            target_ax.add_patch(circle)
            paraboloid_circle_artists.append(circle)

    def _update_paraboloid_fit():
        nonlocal paraboloid_fit_info, selected_slots
        slot_info = selected_slots[1]
        if slot_info is None:
            paraboloid_fit_info = None
            _update_paraboloid_panel(None)
            _update_paraboloid_artists(None)
            fig.canvas.draw_idle()
            return
        if len(slot_info) >= 5:
            x_idx, y_idx = slot_info[3], slot_info[4]
        else:
            x_idx = y_idx = None
        _, x_val, y_val = slot_info[:3]
        if x_idx is None and x_val is not None:
            x_idx = int(np.argmin(np.abs(x - x_val)))
        if y_idx is None and y_val is not None:
            try:
                y_idx = _y_to_index(y_val)
            except Exception:
                y_idx = None
        if x_idx is None or y_idx is None:
            paraboloid_fit_info = None
            _update_paraboloid_panel(None)
            _update_paraboloid_artists(None)
            fig.canvas.draw_idle()
            return
        slot_tuple = (slot_info[0], x_val, y_val, x_idx, y_idx)
        selected_slots[1] = slot_tuple
        result = fit_paraboloid(
            height_map, x_idx, y_idx, paraboloid_window_um, pixel_size
        )
        paraboloid_fit_info = result
        _update_paraboloid_panel(result)
        _update_paraboloid_artists(result)
        fig.canvas.draw_idle()

    def _apply_line_colors():
        """Update indicator line colors based on selection order."""
        colors = ['purple', 'orange']
        for idx, vls in enumerate(selected_vlines):
            for ln in vls:
                ln.set_color(colors[idx])
        for idx, ln in enumerate(selected_cross_hlines):
            if ln is not None:
                ln.set_color(colors[idx])

    def _visible_x_slice():
        """Return slice of indices within the visible x-range of the cross-section plot."""
        xmin, xmax = sorted(ax2.get_xlim())
        start = np.searchsorted(x, xmin, side="left")
        end = np.searchsorted(x, xmax, side="right")
        return slice(start, end)

    # autoscale ax2 y-limits to the currently visible x-range
    def _autoscale_ax2_y_to_visible():
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        sl = _visible_x_slice()
        yd = ydata[sl] if (sl.stop - sl.start) > 0 else ydata
        if yd.size == 0 or not np.isfinite(yd).any():
            return
        ymin = float(np.nanmin(yd))
        ymax = float(np.nanmax(yd))
        if ymin == ymax:
            pad = 1.0 if ymin == 0.0 else abs(ymin) * 0.05
            ymin -= pad
            ymax += pad

        # Add an extra 5% padding to the computed span (both sides)
        span = ymax - ymin
        if span <= 0:
            # Fallback small padding if something went wrong
            extra = 1.0
        else:
            extra = span * 0.05
        ymin -= extra
        ymax += extra

        ax2.set_ylim(ymin, ymax)

    def _y_to_index(y_val):
        """Convert a y-axis value to a row index in height_map."""
        bottom_idx = min(y_pixels, key=lambda yp: abs(yp * pixel_size - y_val))
        return y_pixel_count - 1 - bottom_idx

    def _visible_matrix_slices():
        """Return slices for the currently visible region of the image axes."""
        ref_ax = ax_height if image_axes else ax1
        xlim = sorted(ref_ax.get_xlim())
        ylim = sorted(ref_ax.get_ylim())
        xs = slice(np.searchsorted(x, xlim[0], side="left"),
                   np.searchsorted(x, xlim[1], side="right"))
        y_start = _y_to_index(ylim[1])  # lower index
        y_end = _y_to_index(ylim[0]) + 1  # upper index inclusive
        if y_start > y_end:
            y_start, y_end = y_end, y_start
        ys = slice(y_start, y_end)
        return ys, xs

    def _update_cross_section(new_height):
        """Change which row is used for the cross-section plots."""
        nonlocal line_height, nearest_y_to_plot, cumulative_adjusted_height, time_since_start
        # Clamp to valid y-range in microns (0 .. y_dimension)
        line_height = float(new_height)
        if line_height < 0:
            line_height = 0.0
        elif line_height > y_dimension:
            line_height = y_dimension

        nearest_y_to_plot = y_pixel_count - 1 - min(
            y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height)
        )

        line_datetime = image.get_line_acquisition_datetime(nearest_y_to_plot)
        time_since_start = (line_datetime - image.get_scan_start_datetime()).total_seconds()

        cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()
        hline_height.set_ydata([line_height, line_height])
        hline_contrast.set_ydata([line_height, line_height])
        if hline_phase is not None:
            hline_phase.set_ydata([line_height, line_height])

        cross_line.set_ydata(cumulative_adjusted_height)
        _autoscale_ax2_y_to_visible()
        ax2.set_title(f"{title_prefix} at y = {round(line_height, 3)} μm")
        update_stats_display()
        fig.canvas.draw_idle()

    def _record_selection(x_val, h_val, x_idx=None, y_idx=None, slot_override=None, *, advance=True):
        nonlocal next_slot, locked_slot
        colors = ['purple', 'orange']

        if x_idx is None:
            if x_val is None:
                return False
            x_idx = int(np.argmin(np.abs(x - x_val)))
        else:
            x_idx = int(np.clip(x_idx, 0, len(x) - 1))
        x_val = float(x[x_idx])

        if y_idx is None:
            y_idx = int(nearest_y_to_plot)
        else:
            y_idx = int(np.clip(y_idx, 0, y_pixel_count - 1))
        y_val = float(line_height)

        if slot_override is None:
            slot = next_slot if locked_slot is None else 1 - locked_slot
        else:
            slot = slot_override
            if slot not in (0, 1):
                raise ValueError("slot_override must be 0 or 1")
            if locked_slot is not None and slot == locked_slot:
                return False

        # Remove any existing lines in this slot
        for ln in selected_vlines[slot]:
            ln.remove()
        selected_vlines[slot] = []
        if selected_cross_hlines[slot] is not None:
            selected_cross_hlines[slot].remove()
            selected_cross_hlines[slot] = None
        for ln in selected_image_hlines[slot]:
            ln.remove()
        selected_image_hlines[slot] = []

        # Draw new indicator lines
        vls = []
        for img_ax in image_axes:
            vls.append(img_ax.axvline(x_val, linestyle=':', color=colors[slot]))
        vls.append(ax2.axvline(x_val, linestyle=':', color=colors[slot]))
        selected_vlines[slot] = vls
        selected_cross_hlines[slot] = ax2.axhline(h_val, linestyle=':', color=colors[slot])

        y_val = line_height
        img_lines = [img_ax.axhline(y_val, linestyle=':', color='r') for img_ax in image_axes]
        selected_image_hlines[slot] = img_lines

        selected_slots[slot] = (float(h_val), x_val, y_val, x_idx, y_idx)

        if slot == 1:
            _update_paraboloid_fit()

        if advance:
            if locked_slot is None:
                next_slot = 1 - slot
            else:
                next_slot = 1 - locked_slot

        _apply_line_colors()
        return True

    def _compute_extremum_square_info():
        """Return info about the 4 μm square around the extremum selection."""
        if len(selected_slots) < 2:
            return None
        extremum_info = selected_slots[1]
        if extremum_info is None or len(extremum_info) < 5:
            return None
        _, _, _, x_idx, y_idx = extremum_info[:5]
        if x_idx is None or y_idx is None:
            return None

        return compute_extremum_square_info(
            height_map, x_idx, y_idx, pixel_size
        )

    def update_stats_display():
        """Refresh the text panel summarizing selection info."""
        nonlocal square_marker_artists
        ax4.cla()
        ax4.axis('off')

        # Clear existing extremum markers from the images.
        for artist in square_marker_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        square_marker_artists = []

        entries = []
        if time_since_start is not None:
            entries.append((f"Line imaged {time_since_start:.2f} seconds after imaging began", 'black'))

        entries.append((f"Scan Rate: {scan_rate:.2f} Hz", 'black'))
        entries.append((f"Drive Amplitude: {image.get_initial_drive_amplitude():.2f} mV", 'black'))
        entries.append((f"X-offset: {image.get_x_offset():.2f} μm", 'black'))
        entries.append((f"Y-offset: {image.get_y_offset():.2f} μm", 'black'))

        # Display selected heights for both slots
        for idx, info in enumerate(selected_slots, 1):
            if info is not None:
                if len(info) >= 5:
                    h, xh, yh, x_idx, y_idx = info[:5]
                else:
                    h, xh, yh = info[:3]
                    x_idx = y_idx = None
                color = 'purple' if idx == 1 else 'orange'
                label = f"{slot_labels[idx - 1]}: {h:.2f} nm at ({xh:.3f}, {yh:.3f}) μm"
                if x_idx is not None and y_idx is not None:
                    label += f" [px ({x_idx}, {y_idx})]"
                if locked_slot == idx - 1:
                    label += " (locked)"
                entries.append((label, color))

        if all(info is not None for info in selected_slots):
            diff = selected_slots[1][0] - selected_slots[0][0]
            xdiff = selected_slots[1][1] - selected_slots[0][1]
            entries.append((f"Δz: {diff:.3f} nm", 'black'))
            entries.append((f"Δx: {xdiff:.3f} μm", 'black'))

        square_info = _compute_extremum_square_info()
        if square_info is not None:
            entries.append((
                "Max Δz within 4 μm square of selected extremum: "
                f"{square_info['delta_nm']:.3f} nm at "
                f"({square_info['x']:.3f}, {square_info['y']:.3f}) μm "
                f"[px ({square_info['x_idx']}, {square_info['y_idx']})]",
                'green'
            ))

            for target_ax in image_axes:
                marker, = target_ax.plot(
                    square_info['x'],
                    square_info['y'],
                    marker='x',
                    color='green',
                    markersize=8,
                    mew=2,
                )
                square_marker_artists.append(marker)

        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        max_idx = np.argmax(ydata)
        min_idx = np.argmin(ydata)
        entries.append((f"Max z cross-section: {ydata[max_idx]:.3f} nm at x={x[max_idx]:.3f} μm", 'black'))
        entries.append((f"Min z cross-section: {ydata[min_idx]:.3f} nm at x={x[min_idx]:.3f} μm", 'black'))

        ypos = 0.95
        for text, color in entries:
            ax4.text(0.05, ypos, text, va='top', color=color)
            ypos -= 0.05

        fig.canvas.draw_idle()

    dragging = False  # True when left mouse is held on an image

    def _on_press(event):
        nonlocal line_height, nearest_y_to_plot, selected_points, cumulative_adjusted_height
        nonlocal time_since_start, next_slot
        nonlocal im_height, im_contrast, im_phase
        nonlocal hline_height, hline_contrast, hline_phase, cross_line
        nonlocal selected_vlines, selected_cross_hlines, selected_image_hlines, selected_slots
        nonlocal last_input, dragging

        last_input = 'mouse'

        # If any toolbar mode is active (e.g. zoom), let the tool handle it.
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar is not None and toolbar.mode != '':
            return

        # right-click on image to set y, slot 1, and mode in slot 0
        if event.button == 3 and event.inaxes in image_axes:
            if event.xdata is None or event.ydata is None:
                return
            # 1) Set the cross-section to this y
            _update_cross_section(event.ydata)

            # 2) Put the height at clicked x into slot 1
            ydata_line = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
            idx = int(np.argmin(np.abs(x - event.xdata)))
            h_val = ydata_line[idx]
            _record_selection(x[idx], h_val, x_idx=idx, y_idx=nearest_y_to_plot, slot_override=1, advance=False)

            # 3) Put mode for this line into slot 0
            set_mode_height(slot=0, advance=False, silent=True)

            update_stats_display()
            fig.canvas.draw_idle()
            return

        if event.inaxes in image_axes and event.button == 1:  # Left-click drag to change y
            dragging = True
            _update_cross_section(event.ydata)
            return

        if event.inaxes == ax2 and event.button == 3:  # Right-click on the bottom plot
            selected_points.append((event.xdata, event.ydata))
            if len(selected_points) == 2:
                # Calculate the slope based on the two selected points
                x1, y1 = selected_points[0]
                x2, y2 = selected_points[1]
                slope = (y2 - y1) / (x2 - x1)

                # Adjust the displayed height cross-section using the slope
                if cumulative_adjusted_height is None:
                    cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()
                cumulative_adjusted_height -= slope * x

                # Update the lower plot without clearing
                cross_line.set_ydata(cumulative_adjusted_height)
                _autoscale_ax2_y_to_visible()
                ax2.set_title(f"{title_prefix} at y = {round(line_height, 3)} μm (Slope Corrected)")

                update_stats_display()
                fig.canvas.draw_idle()

                # Reset selected_points to allow further adjustments
                selected_points = []
            return

        if event.inaxes == ax2 and event.button == 1:  # Left-click on the bottom plot
            if event.xdata is None:
                return
            x_idx = int(np.argmin(np.abs(x - event.xdata)))
            if event.key == 'control':
                if event.ydata is None:
                    return
                height_val = event.ydata
            else:
                ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
                height_val = ydata[x_idx]
            if _record_selection(event.xdata, height_val, x_idx=x_idx, y_idx=nearest_y_to_plot):
                print(f"Height value at x = {event.xdata:.3f} μm: {height_val:.3f} nm")
                update_stats_display()
                fig.canvas.draw_idle()
            return

    def _on_move(event):
        nonlocal dragging
        if dragging and event.inaxes in image_axes and event.ydata is not None:
            _update_cross_section(event.ydata)

    def _on_release(event):
        nonlocal dragging
        if event.button == 1:
            dragging = False

    # Create the figure and axes in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_height, ax1, ax3), (ax2, ax_placeholder, ax4) = axes
    plt.subplots_adjust(bottom=0.32)  # Adjust layout to make space for sliders

    ax_placeholder.axis('off')
    paraboloid_vertex_text = ax_placeholder.text(
        0.05, 0.9, "Paraboloid fit vertex: --, --, --", transform=ax_placeholder.transAxes
    )
    paraboloid_r2_text = ax_placeholder.text(
        0.05, 0.75, "Paraboloid fit R^2: --", transform=ax_placeholder.transAxes
    )
    slider_min = max(pixel_size, 0.1)
    slider_max_candidate = max(x_dimension, y_dimension)
    if slider_max_candidate <= 0:
        slider_max_candidate = slider_min
    slider_max = max(slider_min, slider_max_candidate)
    paraboloid_window_um = min(max(paraboloid_window_um, slider_min), slider_max)
    slider_ax = ax_placeholder.inset_axes([0.15, 0.45, 0.7, 0.12])
    paraboloid_slider = Slider(
        slider_ax,
        'Fit diameter (μm)',
        valmin=slider_min,
        valmax=slider_max,
        valinit=paraboloid_window_um,
    )
    button_ax = ax_placeholder.inset_axes([0.3, 0.15, 0.4, 0.18])
    paraboloid_button = Button(button_ax, 'Paraboloid Vertex (6)')

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
    image_axes[:] = [ax_height, ax1]

    def _build_cmap():
        cmap = plt.cm.get_cmap('turbo').copy()
        cmap.set_under('black')
        cmap.set_over('white')
        return cmap

    cmap_height = _build_cmap()
    im_height = ax_height.imshow(height_map, cmap=cmap_height, extent=extent)
    hline_height = ax_height.axhline(y=line_height, color='r', linestyle='--')
    ax_height.set_title(f"{title_prefix} Map")
    ax_height.set_ylabel("y (μm)")

    cmap_contrast = _build_cmap()
    im_contrast = ax1.imshow(contrast_map, cmap=cmap_contrast, extent=extent)
    hline_contrast = ax1.axhline(y=line_height, color='r', linestyle='--')
    ax1.set_title("Contrast Map")
    ax1.set_ylabel("y (μm)")

    if phase_map is not None:
        cmap_phase = _build_cmap()
        im_phase = ax3.imshow(
            phase_map,
            cmap=cmap_phase,
            extent=extent,
            vmin=np.min(phase_map),
            vmax=np.max(phase_map),
        )
        hline_phase = ax3.axhline(y=line_height, color='r', linestyle='--')
        ax3.set_title("Phase Map")
        ax3.set_ylabel("y (μm)")
        image_axes.append(ax3)
    else:
        im_phase = None
        hline_phase = None
        ax3.axis('off')


    cross_line, = ax2.plot(x, height_map[nearest_y_to_plot, :])
    ax2.set_xlim(0, x_dimension)  # Ensure x-axis matches upper plot
    ax2.set_title(f"{title_prefix} at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Ensure 1 μm in x corresponds to 1 μm in y regardless of zoom
    for axis in image_axes:
        axis.set_aspect('equal', adjustable='box')

    # Match the cross-section width with the image plots while allowing the
    # y-axis to occupy the full subplot height.
    aligning_ax2 = False

    def _align_cross_section_axes(event=None):
        nonlocal aligning_ax2
        if aligning_ax2:
            return
        aligning_ax2 = True
        try:
            bbox_image = ax_height.get_position()
            bbox_cross = ax2.get_position()
            new_bounds = [bbox_image.x0, bbox_cross.y0, bbox_image.width, bbox_cross.height]
            current = ax2.get_position()
            if (
                abs(current.x0 - new_bounds[0]) > 1e-5
                or abs(current.y0 - new_bounds[1]) > 1e-5
                or abs(current.width - new_bounds[2]) > 1e-5
                or abs(current.height - new_bounds[3]) > 1e-5
            ):
                ax2.set_position(new_bounds)
                if event is not None:
                    fig.canvas.draw_idle()
        finally:
            aligning_ax2 = False

    _align_cross_section_axes()
    fig.canvas.mpl_connect('resize_event', _align_cross_section_axes)
    fig.canvas.mpl_connect('draw_event', _align_cross_section_axes)

    ax4.axis('off')
    update_stats_display()

    # Create sliders for the three map panels
    slider_height = 0.1
    slider_bottom = 0.16
    slider_cmap = plt.cm.get_cmap('turbo')

    def _make_slider_axis(target_axis):
        bbox = target_axis.get_position()
        return plt.axes([bbox.x0, slider_bottom, bbox.width, slider_height], facecolor='none')

    ax_height_slider = _make_slider_axis(ax_height)
    height_min = float(np.min(height_map))
    height_max = float(np.max(height_map))
    height_slider = DualHandleSlider(
        ax_height_slider,
        'Height (nm)',
        height_min,
        height_max,
        (height_min, height_max),
        slider_cmap,
        value_format="{:.3f}",
    )

    ax_contrast_slider = _make_slider_axis(ax1)
    contrast_min = float(np.min(contrast_map) * 1e9)
    contrast_max = float(np.max(contrast_map) * 1e9)
    contrast_slider = DualHandleSlider(
        ax_contrast_slider,
        'Contrast (x10^9)',
        contrast_min,
        contrast_max,
        (
            np.clip(0.0, contrast_min, contrast_max),
            np.clip(1e9, contrast_min, contrast_max),
        ),
        slider_cmap,
        value_format="{:.2f}",
    )

    phase_slider = None
    if phase_map is not None:
        ax_phase_slider = _make_slider_axis(ax3)
        phase_min = float(np.min(phase_map))
        phase_max = float(np.max(phase_map))
        phase_slider = DualHandleSlider(
            ax_phase_slider,
            'Phase',
            phase_min,
            phase_max,
            (phase_min, phase_max),
            slider_cmap,
            value_format="{:.2f}",
        )

    def _update_visible_ranges():
        """Reset slider ranges to the data limits of the visible region."""
        ys, xs = _visible_matrix_slices()
        sub_height = height_map[ys, xs]
        hmin = float(np.min(sub_height))
        hmax = float(np.max(sub_height))
        height_slider.set_bounds(hmin, hmax, update_values=False)
        height_slider.set_val((hmin, hmax))

        sub_contrast = contrast_map[ys, xs]
        cmin = float(np.min(sub_contrast)) * 1e9
        cmax = float(np.max(sub_contrast)) * 1e9
        contrast_slider.set_bounds(cmin, cmax, update_values=False)
        contrast_slider.set_val((cmin, cmax))
        if phase_map is not None and phase_slider is not None:
            sub_phase = phase_map[ys, xs]
            pmin = float(np.min(sub_phase))
            pmax = float(np.max(sub_phase))
            phase_slider.set_bounds(pmin, pmax, update_values=False)
            phase_slider.set_val((pmin, pmax))

    syncing_zoom = False
    pending_zoom_autoset = False
    zoom_autoset_timer = None

    def _sync_zoom(ax):
        nonlocal syncing_zoom, pending_zoom_autoset
        if ax not in image_axes:
            return
        if syncing_zoom:
            return
        syncing_zoom = True
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for other_ax in image_axes:
            if other_ax is ax:
                continue
            other_ax.set_xlim(xlim)
            other_ax.set_ylim(ylim)
        ax2.set_xlim(xlim)
        _autoscale_ax2_y_to_visible()
        # Ensure the cross-section subplot keeps the same width as the image/phase
        # plot after interactive zooming.
        _align_cross_section_axes()
        _update_visible_ranges()
        # Defer auto-select until after all limit change callbacks have fired
        # to ensure we use the final zoomed x/y ranges.
        if pending_zoom_autoset:
            nonlocal zoom_autoset_timer
            def _run_autoset_from_timer():
                nonlocal pending_zoom_autoset, zoom_autoset_timer
                if pending_zoom_autoset:
                    pending_zoom_autoset = False
                    _auto_select_zoom_points()
                zoom_autoset_timer = None

            # Use a short single-shot timer to run after current UI updates.
            if zoom_autoset_timer is not None:
                try:
                    zoom_autoset_timer.stop()
                except Exception:
                    pass
                zoom_autoset_timer = None
            try:
                zoom_autoset_timer = fig.canvas.new_timer(interval=50)
                zoom_autoset_timer.single_shot = True
                zoom_autoset_timer.add_callback(_run_autoset_from_timer)
                zoom_autoset_timer.start()
            except Exception:
                # Fallback: run immediately if timer creation fails
                pending_zoom_autoset = False
                _auto_select_zoom_points()
        fig.canvas.draw_idle()
        syncing_zoom = False

    def _zoom_to_square(center_x, center_y, size_um=4.0):
        """Programmatically zoom image axes to a square region in microns."""
        if not image_axes:
            return False
        if size_um <= 0:
            return False

        half = size_um / 2.0

        def _calc_limits(center, total_span):
            start = center - half
            end = center + half
            if start < 0:
                end += -start
                start = 0
            if end > total_span:
                start -= (end - total_span)
                end = total_span
            start = max(0.0, start)
            end = min(total_span, end)
            desired = size_um
            current = end - start
            if total_span >= desired and current < desired:
                deficit = desired - current
                start = max(0.0, start - deficit / 2)
                end = min(total_span, end + deficit / 2)
                # Final clamp in case shifting hit a boundary.
                if end - start < desired:
                    if start <= 0:
                        end = min(total_span, start + desired)
                    elif end >= total_span:
                        start = max(0.0, end - desired)
            return start, end

        xmin, xmax = _calc_limits(center_x, x_dimension)
        ymin, ymax = _calc_limits(center_y, y_dimension)
        if xmin >= xmax or ymin >= ymax:
            return False

        ref_ax = image_axes[0]
        ref_ax.set_xlim(xmin, xmax)
        ref_ax.set_ylim(ymin, ymax)
        _sync_zoom(ref_ax)
        return True

    def _zoom_to_full_extent():
        """Reset all synchronized axes to show the full scan area."""
        if not image_axes:
            return False
        ref_ax = image_axes[0]
        ref_ax.set_xlim(0, x_dimension)
        ref_ax.set_ylim(0, y_dimension)
        _sync_zoom(ref_ax)
        return True

    # Update function for the sliders
    def update_slider(_=None):
        hmin, hmax = height_slider.val
        im_height.set_clim(vmin=hmin, vmax=hmax)
        cmin, cmax = contrast_slider.val
        im_contrast.set_clim(vmin=cmin / 1e9, vmax=cmax / 1e9)
        if phase_map is not None and im_phase is not None and phase_slider is not None:
            pmin, pmax = phase_slider.val
            im_phase.set_clim(vmin=pmin, vmax=pmax)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    height_slider.on_changed(update_slider)
    contrast_slider.on_changed(update_slider)
    if phase_slider is not None:
        phase_slider.on_changed(update_slider)

    _update_visible_ranges()

    for axis in image_axes:
        axis.callbacks.connect('xlim_changed', lambda evt, ax=axis: _sync_zoom(ax))
        axis.callbacks.connect('ylim_changed', lambda evt, ax=axis: _sync_zoom(ax))
    # Also adapt y-axis when user zooms/pans directly on the cross-section
    ax2.callbacks.connect('xlim_changed', lambda evt: _autoscale_ax2_y_to_visible())

    # Ensure sliders remain functional after height selection
    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_move)
    fig.canvas.mpl_connect('button_release_event', _on_release)

    # Make buttons smaller and fit on the same line
    btn_width = 0.12
    btn_height = 0.045
    btn_y = 0.05
    btn_spacing = 0.01

    ax_button_lock = plt.axes([0.08, btn_y, btn_width, btn_height])
    btn_lock = Button(ax_button_lock, 'Lock Point (w)')

    ax_button = plt.axes([0.08 + (btn_width + btn_spacing) * 1, btn_y, btn_width, btn_height])
    btn_set_max = Button(ax_button, 'Max Cross-Section (1)')

    ax_button_min = plt.axes([0.08 + (btn_width + btn_spacing) * 2, btn_y, btn_width, btn_height])
    btn_set_min = Button(ax_button_min, 'Min Cross-Section (2)')

    ax_button_gmax = plt.axes([0.08 + (btn_width + btn_spacing) * 3, btn_y, btn_width, btn_height])
    btn_global_max = Button(ax_button_gmax, 'Max Global (3)')

    ax_button_gmin = plt.axes([0.08 + (btn_width + btn_spacing) * 4, btn_y, btn_width, btn_height])
    btn_global_min = Button(ax_button_gmin, 'Min Global (4)')

    ax_button_mode = plt.axes([0.08 + (btn_width + btn_spacing) * 5, btn_y, btn_width, btn_height])
    btn_mode_height = Button(ax_button_mode, 'Mode, 0.5nm Bins (5)')

    def set_max_height(event=None, *, slot=None, advance=True, silent=False):
        """Select the maximum value within the visible part of the cross-section."""
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        sl = _visible_x_slice()
        if sl.stop - sl.start > 0:
            sub_y = ydata[sl]
            local_idx = sl.start + int(np.argmax(sub_y))
        else:
            local_idx = int(np.argmax(ydata))
        max_x = x[local_idx]
        max_y = ydata[local_idx]
        if _record_selection(
            max_x,
            max_y,
            x_idx=local_idx,
            y_idx=nearest_y_to_plot,
            slot_override=slot,
            advance=advance,
        ):
            if not silent:
                print(f"Max height at x = {max_x:.3f} μm: {max_y:.3f} nm (Set by button)")
            update_stats_display()

    def set_min_height(event=None):
        """Select the minimum value within the visible part of the cross-section."""
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        sl = _visible_x_slice()
        if sl.stop - sl.start > 0:
            sub_y = ydata[sl]
            local_idx = sl.start + int(np.argmin(sub_y))
        else:
            local_idx = int(np.argmin(ydata))
        min_x = x[local_idx]
        min_y = ydata[local_idx]
        if _record_selection(min_x, min_y, x_idx=local_idx, y_idx=nearest_y_to_plot):
            print(f"Min height at x = {min_x:.3f} μm: {min_y:.3f} nm (Set by button)")
            update_stats_display()

    def set_global_max(event=None, *, slot=None, advance=True, silent=False):
        """Select the maximum height within the visible image region."""
        ys, xs = _visible_matrix_slices()
        sub = height_map[ys, xs]
        if sub.size == 0:
            return False
        iy_local, ix_local = np.unravel_index(np.argmax(sub), sub.shape)
        iy = ys.start + iy_local
        ix = xs.start + ix_local
        x_val = x[ix]
        height_val = height_map[iy, ix]
        y_val = image.index_to_y_center(iy)
        _update_cross_section(y_val)
        success = _record_selection(
            x_val,
            height_val,
            x_idx=ix,
            y_idx=iy,
            slot_override=slot,
            advance=advance,
        )
        if success:
            if not silent:
                print(f"Global max height {height_val:.3f} nm at ({x_val:.3f}, {y_val:.3f}) μm")
            update_stats_display()
        return success

    def set_global_min(event=None, *, slot=None, advance=True, silent=False):
        """Select the minimum height within the visible image region."""
        ys, xs = _visible_matrix_slices()
        sub = height_map[ys, xs]
        if sub.size == 0:
            return False
        iy_local, ix_local = np.unravel_index(np.argmin(sub), sub.shape)
        iy = ys.start + iy_local
        ix = xs.start + ix_local
        x_val = x[ix]
        height_val = height_map[iy, ix]
        y_val = image.index_to_y_center(iy)
        _update_cross_section(y_val)
        success = _record_selection(
            x_val,
            height_val,
            x_idx=ix,
            y_idx=iy,
            slot_override=slot,
            advance=advance,
        )
        if success:
            if not silent:
                print(f"Global min height {height_val:.3f} nm at ({x_val:.3f}, {y_val:.3f}) μm")
            update_stats_display()
        return success

    def set_mode_height(event=None, *, slot=None, advance=True, silent=False):
        """Select the mode height from the current cross-section using 0.5 nm bins."""
        ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
        
        mode_val = calculate_substrate_height(ydata)
        if mode_val is None:
            return False

        idx = int(np.argmin(np.abs(ydata - mode_val)))
        success = _record_selection(
            x[idx],
            ydata[idx],
            x_idx=idx,
            y_idx=nearest_y_to_plot,
            slot_override=slot,
            advance=advance,
        )
        if success:
            if not silent:
                print(
                    f"Mode height ~{mode_val:.3f} nm at x = {x[idx]:.3f} μm (Set by button)"
                )
            update_stats_display()
        return success

    def _refine_extremum_with_paraboloid():
        """
        Iteratively fit a paraboloid to the data around the current extremum.
        Updates the extremum to the fitted vertex and repeats until convergence
        or a maximum number of iterations is reached.
        """
        # Check for an initial extremum to start with
        if selected_slots[1] is None:
            return

        # Initial center comes from the current extremum
        current_x, current_y = selected_slots[1][1], selected_slots[1][2]

        best_result = iterative_paraboloid_fit(
            height_map,
            current_x,
            current_y,
            pixel_size,
            paraboloid_window_um
        )

        if best_result is None:
            return

        # Update the extremum selection to the best result
        final_vx = best_result['vx']
        final_vy = best_result['vy']
        final_vz = best_result['vz']
        
        # Update cross-section to the new vertex y
        _update_cross_section(final_vy)
        
        # Record the new extremum
        # Find the closest indices for the record
        final_x_idx = int(np.argmin(np.abs(x - final_vx)))
        final_y_idx = _y_to_index(final_vy)
        
        _record_selection(
            final_vx,
            final_vz,
            x_idx=final_x_idx,
            y_idx=final_y_idx,
            slot_override=1,
            advance=False
        )
        
        # Update the paraboloid display with the best fit info
        nonlocal paraboloid_fit_info
        paraboloid_fit_info = best_result['fit_result']
        _update_paraboloid_panel(paraboloid_fit_info)
        _update_paraboloid_artists(paraboloid_fit_info)


    def _auto_select_zoom_points():
        """Populate default selections after a zoom interaction."""
        if locked_slot == 1:
            return
        
        # If the last zoom was a right-click (button 3), find the min.
        # Otherwise (left-click or default), find the max.
        if last_zoom_button == 3:
            success_extremum = set_global_min(slot=1, silent=True)
        else:
            success_extremum = set_global_max(slot=1, silent=True)

        if not success_extremum:
            return
            
        # Refine the extremum using iterative paraboloid fitting
        _refine_extremum_with_paraboloid()

        if locked_slot == 0:
            return
        # Update substrate to be the mode on the y-line of the (possibly moved) extremum
        set_mode_height(slot=0, silent=True)

    def toggle_lock(event=None):
        """Lock or unlock selections. Double 'w' locks the opposite slot."""
        nonlocal locked_slot, next_slot, last_input
        if locked_slot is not None and last_input == 'w':
            locked_slot = 1 - locked_slot
            next_slot = 1 - locked_slot
        elif locked_slot is None:
            target = 1 - next_slot
            if selected_slots[target] is not None:
                locked_slot = target
                next_slot = 1 - locked_slot
        else:
            locked_slot = None
        last_input = 'w'
        update_stats_display()

    def set_paraboloid_vertex(event=None, *, slot=None, advance=True, silent=False):
        if paraboloid_fit_info is None:
            return False
        vx = paraboloid_fit_info['vertex_x']
        vy = paraboloid_fit_info['vertex_y']
        vz = paraboloid_fit_info['vertex_z']
        x_idx = int(np.argmin(np.abs(x - vx)))
        y_idx = _y_to_index(vy)
        _update_cross_section(vy)
        target_slot = 1 if slot is None else slot
        success = _record_selection(
            vx,
            vz,
            x_idx=x_idx,
            y_idx=y_idx,
            slot_override=target_slot,
            advance=advance,
        )
        if success and target_slot == 1:
            set_mode_height(slot=0, advance=False, silent=True)
            if not silent:
                print(
                    f"Paraboloid vertex height {vz:.3f} nm at ({vx:.3f}, {vy:.3f}) μm "
                    "(Extremum) | Substrate set to 0.5 nm mode"
                )
            update_stats_display()
        elif success and not silent:
            print(
                f"Paraboloid vertex height {vz:.3f} nm at ({vx:.3f}, {vy:.3f}) μm (Set by button)"
            )
            update_stats_display()
        elif success:
            update_stats_display()
        return success

    btn_set_max.on_clicked(set_max_height)
    btn_set_min.on_clicked(set_min_height)
    btn_global_max.on_clicked(set_global_max)
    btn_global_min.on_clicked(set_global_min)
    btn_mode_height.on_clicked(set_mode_height)
    btn_lock.on_clicked(toggle_lock)
    paraboloid_button.on_clicked(set_paraboloid_vertex)

    def _on_paraboloid_slider(val):
        nonlocal paraboloid_window_um
        paraboloid_window_um = float(val)
        _update_paraboloid_fit()

    paraboloid_slider.on_changed(_on_paraboloid_slider)

    def _hotkey(event):
        nonlocal last_input, aborted
        if event.key == '1':
            set_max_height()
        elif event.key == '2':
            set_min_height()
        elif event.key == '3':
            set_global_max()
        elif event.key == '4':
            set_global_min()
        elif event.key == '5':
            set_mode_height()
        elif event.key == '6':
            set_paraboloid_vertex()
        elif event.key == 'w':
            toggle_lock()
        elif event.key == 'r':
            _zoom_to_full_extent()
        elif event.key == 'z':
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar is not None and hasattr(toolbar, "zoom"):
                toolbar.zoom()
        elif event.key == 'up':
            # Move up by one pixel (decrease index = increase y-value)
            new_y_idx = max(nearest_y_to_plot - 1, 0)
            new_y_val = image.index_to_y_center(new_y_idx)
            _update_cross_section(new_y_val)
            # If Control is held, auto-select max and mode
            if event.key == 'ctrl+up':
                set_max_height()
                set_mode_height()
        elif event.key == 'down':
            # Move down by one pixel (increase index = decrease y-value)
            new_y_idx = min(nearest_y_to_plot + 1, y_pixel_count - 1)
            new_y_val = image.index_to_y_center(new_y_idx)
            _update_cross_section(new_y_val)
            # If Control is held, auto-select max and mode
            if event.key == 'ctrl+down':
                set_max_height()
                set_mode_height()
        elif event.key == 'ctrl+up':
            # Move up by one pixel and auto-select with a fixed slot order
            new_y_idx = max(nearest_y_to_plot - 1, 0)
            new_y_val = image.index_to_y_center(new_y_idx)
            _update_cross_section(new_y_val)
            set_max_height(slot=1, silent=True)
            set_mode_height(slot=0, silent=True)
        elif event.key == 'ctrl+down':
            # Move down by one pixel and auto-select with a fixed slot order
            new_y_idx = min(nearest_y_to_plot + 1, y_pixel_count - 1)
            new_y_val = image.index_to_y_center(new_y_idx)
            _update_cross_section(new_y_val)
            set_max_height(slot=1, silent=True)
            set_mode_height(slot=0, silent=True)
        elif event.key == 'tab':  # abort and close
            aborted = True
            plt.close(fig)
            return
        last_input = event.key

    fig.canvas.mpl_connect('key_press_event', _hotkey)

    # --- Zoom click handling ---
    zoom_press_xy = None

    def _zoom_press(event):
        nonlocal zoom_press_xy
        tb = plt.get_current_fig_manager().toolbar
        # Allow button 1 (left) or 3 (right) for zoom rectangle
        if tb is not None and tb.mode == 'zoom rect' and event.button in (1, 3) and event.inaxes in image_axes:
            zoom_press_xy = (event.x, event.y)
        else:
            zoom_press_xy = None

    def _zoom_release(event):
        nonlocal zoom_press_xy, pending_zoom_autoset, last_zoom_button
        tb = plt.get_current_fig_manager().toolbar
        if (
            tb is not None
            and tb.mode == 'zoom rect'
            and zoom_press_xy is not None
            and event.button in (1, 3)
            and event.inaxes in image_axes
        ):
            last_zoom_button = event.button
            pending_zoom_autoset = True
            dx = event.x - zoom_press_xy[0]
            dy = event.y - zoom_press_xy[1]
            if abs(dx) < 5 and abs(dy) < 5 and event.xdata is not None and event.ydata is not None:
                x0 = event.xdata
                y0 = event.ydata
                half = 3
                xlim = (x0 - half, x0 + half)
                ylim = (y0 - half, y0 + half)
                event.inaxes.set_xlim(*xlim)
                event.inaxes.set_ylim(*ylim)
            try:
                tb.zoom()
            except Exception:
                pass
            zoom_press_xy = None

    fig.canvas.mpl_connect('button_press_event', _zoom_press)
    fig.canvas.mpl_connect('button_release_event', _zoom_release)

    # Override toolbar home to always rescale the cross-section
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar is not None:
        if hasattr(toolbar, "home"):
            orig_home = toolbar.home

            def _home(*args, **kwargs):
                if not _zoom_to_full_extent():
                    orig_home(*args, **kwargs)
                    ref_ax = ax_height if image_axes else ax1
                    ax2.set_xlim(ref_ax.get_xlim())
                    _autoscale_ax2_y_to_visible()
                    fig.canvas.draw_idle()

            toolbar.home = _home
        # Only auto-enter zoom mode for sufficiently large scans in both axes
        if (x_dimension >= 8 and y_dimension >= 5) and hasattr(toolbar, "zoom"):
            toolbar.zoom()

    # Determine whether there are selections to restore before running
    # the automatic small-scan defaults. If prior selections exist, keep them
    # instead of replacing them with auto-selected points.
    has_initial_preload = bool(
        initial_selected_slots
        and any(slot is not None for slot in initial_selected_slots)
    )

    # Auto-select when x < 8 μm OR y < 5 μm unless restoring previously
    # saved selections.
    if ((x_dimension < 8) or (y_dimension < 5)) and not has_initial_preload:
        # Global max in slot 1 (orange), mode in slot 0 (purple)
        set_global_max(slot=1, silent=True)
        _refine_extremum_with_paraboloid()
        set_mode_height(slot=0, silent=True)
        update_stats_display()

    fig.canvas.manager.set_window_title(image_bname + " - " + scan_end_date_time)

    # If initial selections were provided, apply them after the plots are created
    # initial_selected_slots expected as a list like [(h,x,y), (h,x,y)] or None
    if initial_selected_slots:
        try:
            for idx, info in enumerate(initial_selected_slots):
                if info is None:
                    continue
                # info expected as (h, x, y) or similar; record into slot idx
                h_val, x_val, y_val = info[:3]
                x_idx = info[3] if len(info) > 3 else None
                y_idx = info[4] if len(info) > 4 else None
                _update_cross_section(y_val)
                _record_selection(x_val, h_val, x_idx=x_idx, y_idx=y_idx, slot_override=idx, advance=False)
        except Exception:
            # ignore any malformed initial selection data
            pass
        else:
            # Ensure the restored selections are reflected in the UI summary.
            update_stats_display()
            # When both substrate and extremum points already exist, zoom in on
            # a 4 μm square centered on the extremum to make it easy to resume
            # work without re-navigating.
            if (
                selected_slots[0] is not None
                and selected_slots[1] is not None
                and len(selected_slots[1]) >= 3
            ):
                x_center = selected_slots[1][1]
                y_center = selected_slots[1][2]
                if x_center is not None and y_center is not None:
                    _zoom_to_square(float(x_center), float(y_center))

    # Use a manual event loop to ensure blocking behavior, especially when called
    # from within another matplotlib callback (nested event loops).
    def _on_close(event):
        fig.canvas.stop_event_loop()
    fig.canvas.mpl_connect('close_event', _on_close)

    plt.show(block=False)
    fig.canvas.start_event_loop()

    if aborted:
        return None  # Canceled by Tab key

    # Return a structured result so callers can store positions and reuse them
    time_offset = None
    if time_since_start is not None:
        time_offset = time_since_start - imaging_duration

    return {
        'selected_slots': selected_slots,  # list [ (h,x,y) | None, (h,x,y) | None ]
        'time_offset': time_offset
    }
