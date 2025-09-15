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

    # Determine which height data to use. If flattened data is available use it
    # and update the title prefix accordingly.
    imaging_mode = image.get_imaging_mode()
    title_prefix = "Height"
    if imaging_mode == 'AC Mode' and image.wave_data.shape[2] > 4:
        height_map = image.get_FlatHeight()
        title_prefix = "Flattened Height"
    elif imaging_mode == 'Contact' and image.wave_data.shape[2] > 3:
        height_map = image.get_FlatHeight()
        title_prefix = "Flattened Height"
    else:
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

    # Slots for the two selectable height values
    selected_slots = [None, None]  # [(height, x, y), ...]

    # Handles for vertical and horizontal indicator lines for selected heights
    selected_vlines = [[], []]  # [[ax1_line, ax2_line, ax3_line?], ...]
    selected_cross_hlines = [None, None]  # [ax2_line, ...]
    selected_image_hlines = [[], []]  # [[ax1_line, ax3_line?], ...]
    next_slot = 0  # Which slot to overwrite next
    locked_slot = None  # Slot index that should not be overwritten

    last_input = None  # Track the last user action for double-press logic

    time_since_start = None  # To store the time since the start of the scan

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

    def _y_to_index(y_val):
        """Convert a y-axis value to a row index in height_map."""
        bottom_idx = min(y_pixels, key=lambda yp: abs(yp * pixel_size - y_val))
        return y_pixel_count - 1 - bottom_idx

    def _visible_matrix_slices():
        """Return slices for the currently visible region of the image axes."""
        xlim = sorted(ax1.get_xlim())
        ylim = sorted(ax1.get_ylim())
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
        line_height = float(new_height)
        if line_height < 0 or line_height >= contrast_map.shape[0]:
            return

        nearest_y_to_plot = y_pixel_count - 1 - min(
            y_pixels, key=lambda y_pixel: abs(y_pixel * pixel_size - line_height)
        )

        ratio = nearest_y_to_plot / y_pixel_count
        if scan_direction == 1:  # Scanning down
            time_since_start = ratio * imaging_duration
        else:
            time_since_start = (1 - ratio) * imaging_duration

        cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()
        hline_contrast.set_ydata([line_height, line_height])
        if hline_phase is not None:
            hline_phase.set_ydata([line_height, line_height])

        cross_line.set_ydata(cumulative_adjusted_height)
        ax2.set_autoscaley_on(True)
        ax2.relim()
        ax2.autoscale_view(scalex=False)
        ax2.set_title(f"{title_prefix} at y = {round(line_height, 3)} μm")
        update_stats_display()
        fig.canvas.draw_idle()

    def _record_selection(x_val, h_val, slot_override=None, *, advance=True):
        """Store a selected height and draw indicator lines."""
        nonlocal next_slot, locked_slot
        colors = ['purple', 'orange']

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
        vls = [ax1.axvline(x_val, linestyle=':', color=colors[slot]),
               ax2.axvline(x_val, linestyle=':', color=colors[slot])]
        if phase_map is not None:
            vls.append(ax3.axvline(x_val, linestyle=':', color=colors[slot]))
        selected_vlines[slot] = vls
        selected_cross_hlines[slot] = ax2.axhline(h_val, linestyle=':', color=colors[slot])

        y_val = line_height
        img_lines = [ax1.axhline(y_val, linestyle=':', color='r')]
        if phase_map is not None:
            img_lines.append(ax3.axhline(y_val, linestyle=':', color='r'))
        selected_image_hlines[slot] = img_lines

        selected_slots[slot] = (h_val, x_val, line_height)

        if advance:
            if locked_slot is None:
                next_slot = 1 - slot
            else:
                next_slot = 1 - locked_slot

        _apply_line_colors()
        return True

    def update_stats_display():
        """Refresh the text panel summarizing selection info."""
        ax4.cla()
        ax4.axis('off')

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
                h, xh, yh = info
                color = 'purple' if idx == 1 else 'orange'
                label = f"Point {idx}: {h:.3f} nm at ({xh:.3f}, {yh:.3f}) μm"
                if locked_slot == idx - 1:
                    label += " (locked)"
                entries.append((label, color))

        if all(info is not None for info in selected_slots):
            diff = selected_slots[1][0] - selected_slots[0][0]
            xdiff = selected_slots[1][1] - selected_slots[0][1]
            entries.append((f"Δz: {diff:.3f} nm", 'black'))
            entries.append((f"Δx: {xdiff:.3f} μm", 'black'))

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
        nonlocal im, im_phase, hline_contrast, hline_phase, cross_line
        nonlocal selected_vlines, selected_cross_hlines, selected_image_hlines, selected_slots
        nonlocal last_input, dragging

        last_input = 'mouse'

        # Check if the event is triggered during zooming or panning
        if plt.get_current_fig_manager().toolbar.mode != '':
            return  # Ignore events during zoom or pan

        if event.inaxes in (ax1, ax3) and event.button == 1:  # Left-click on either image
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

                # Adjust the displayed height retrace using the slope
                if cumulative_adjusted_height is None:
                    cumulative_adjusted_height = height_map[nearest_y_to_plot, :].copy()
                cumulative_adjusted_height -= slope * x

                # Update the lower plot without clearing
                cross_line.set_ydata(cumulative_adjusted_height)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_title(f"{title_prefix} at y = {round(line_height, 3)} μm (Slope Corrected)")

                update_stats_display()
                fig.canvas.draw_idle()

                # Reset selected_points to allow further adjustments
                selected_points = []
            return

        if event.inaxes == ax2 and event.button == 1:  # Left-click on the bottom plot
            if event.key == 'control':
                height_val = event.ydata
            else:
                ydata = cumulative_adjusted_height if cumulative_adjusted_height is not None else height_map[nearest_y_to_plot, :]
                idx = np.argmin(np.abs(x - event.xdata))
                height_val = ydata[idx]

            if _record_selection(event.xdata, height_val):
                print(f"Height value at x = {event.xdata:.3f} μm: {height_val:.3f} nm")
                update_stats_display()
                fig.canvas.draw_idle()
            return

    def _on_move(event):
        nonlocal dragging
        if dragging and event.inaxes in (ax1, ax3) and event.ydata is not None:
            _update_cross_section(event.ydata)

    def _on_release(event):
        nonlocal dragging
        if event.button == 1:
            dragging = False

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


    cross_line, = ax2.plot(x, height_map[nearest_y_to_plot, :])
    ax2.set_xlim(0, scan_size)  # Ensure x-axis matches upper plot
    ax2.set_title(f"{title_prefix} at y = {line_height} μm")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("Height (nm)")

    # Ensure 1 μm in x corresponds to 1 μm in y regardless of zoom
    ax1.set_aspect('equal', adjustable='box')
    if phase_map is not None:
        ax3.set_aspect('equal', adjustable='box')

    # Match the cross-section width with the image plots
    height, width = contrast_map.shape
    aspect_ratio = height / width
    ax2.set_box_aspect(aspect_ratio)

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

    def _update_visible_ranges():
        """Reset slider ranges to the data limits of the visible region."""
        ys, xs = _visible_matrix_slices()
        sub_contrast = contrast_map[ys, xs]
        cmin = float(np.min(sub_contrast))
        cmax = float(np.max(sub_contrast))
        for sl in (slider_vmin, slider_vmax):
            sl.valmin = cmin * 1e9
            sl.valmax = cmax * 1e9
            sl.ax.set_xlim(sl.valmin, sl.valmax)
        slider_vmin.set_val(slider_vmin.valmin)
        slider_vmax.set_val(slider_vmax.valmax)
        if phase_map is not None:
            sub_phase = phase_map[ys, xs]
            pmin = float(np.min(sub_phase))
            pmax = float(np.max(sub_phase))
            for sl in (slider_phase_vmin, slider_phase_vmax):
                sl.valmin = pmin
                sl.valmax = pmax
                sl.ax.set_xlim(sl.valmin, sl.valmax)
            slider_phase_vmin.set_val(pmin)
            slider_phase_vmax.set_val(pmax)

    syncing_zoom = False
    pending_zoom_autoset = False

    def _sync_zoom(ax):
        nonlocal syncing_zoom, pending_zoom_autoset
        if syncing_zoom:
            return
        syncing_zoom = True
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ax is ax1 and phase_map is not None:
            ax3.set_xlim(xlim)
            ax3.set_ylim(ylim)
        elif ax is ax3:
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        _update_visible_ranges()
        if pending_zoom_autoset:
            pending_zoom_autoset = False
            _auto_select_zoom_points()
        fig.canvas.draw_idle()
        syncing_zoom = False

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

    _update_visible_ranges()

    ax1.callbacks.connect('xlim_changed', lambda evt: _sync_zoom(ax1))
    ax1.callbacks.connect('ylim_changed', lambda evt: _sync_zoom(ax1))
    if phase_map is not None:
        ax3.callbacks.connect('xlim_changed', lambda evt: _sync_zoom(ax3))
        ax3.callbacks.connect('ylim_changed', lambda evt: _sync_zoom(ax3))

    # Ensure sliders remain functional after height selection
    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_move)
    fig.canvas.mpl_connect('button_release_event', _on_release)

    from matplotlib.widgets import Button
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

    def set_max_height(event=None):
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
        if _record_selection(max_x, max_y):
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
        if _record_selection(min_x, min_y):
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
        y_val = (y_pixel_count - 1 - iy) * pixel_size
        _update_cross_section(y_val)
        success = _record_selection(
            x_val,
            height_val,
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
        y_val = (y_pixel_count - 1 - iy) * pixel_size
        _update_cross_section(y_val)
        success = _record_selection(
            x_val,
            height_val,
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
        if ydata.size == 0:
            return False
        bins = np.arange(np.min(ydata), np.max(ydata) + 0.5, 0.5)
        if bins.size < 2:
            return False
        hist, edges = np.histogram(ydata, bins=bins)
        mode_idx = int(np.argmax(hist))
        mode_val = (edges[mode_idx] + edges[mode_idx + 1]) / 2
        idx = int(np.argmin(np.abs(ydata - mode_val)))
        success = _record_selection(
            x[idx],
            ydata[idx],
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

    def _auto_select_zoom_points():
        """Populate default selections after a zoom interaction."""
        if locked_slot == 1:
            return
        success_max = set_global_max(slot=1, silent=True)
        if not success_max:
            return
        if locked_slot == 0:
            return
        set_mode_height(slot=0, silent=True)

    def toggle_lock(event=None):
        """Lock or unlock selections. Double 'w' locks the opposite slot."""
        nonlocal locked_slot, next_slot, last_input
        if locked_slot is not None and last_input == 'w':
            locked_slot = 1 - locked_slot
            next_slot = 1 - locked_slot
            print(f"Locked selection {locked_slot + 1}")
        elif locked_slot is None:
            target = 1 - next_slot
            if selected_slots[target] is not None:
                locked_slot = target
                next_slot = 1 - locked_slot
                print(f"Locked selection {locked_slot + 1}")
        else:
            print("Unlocked selection")
            locked_slot = None
        last_input = 'w'
        update_stats_display()

    btn_set_max.on_clicked(set_max_height)
    btn_set_min.on_clicked(set_min_height)
    btn_global_max.on_clicked(set_global_max)
    btn_global_min.on_clicked(set_global_min)
    btn_mode_height.on_clicked(set_mode_height)
    btn_lock.on_clicked(toggle_lock)

    def _hotkey(event):
        nonlocal last_input
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
        elif event.key == 'w':
            toggle_lock()
        elif event.key == 'z':
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar is not None and hasattr(toolbar, "zoom"):
                toolbar.zoom()
        last_input = event.key

    fig.canvas.mpl_connect('key_press_event', _hotkey)

    # --- Zoom click handling ---
    zoom_press_xy = None

    def _zoom_press(event):
        nonlocal zoom_press_xy
        tb = plt.get_current_fig_manager().toolbar
        if tb is not None and tb.mode == 'zoom rect' and event.button == 1 and event.inaxes in (ax1, ax3):
            zoom_press_xy = (event.x, event.y)
        else:
            zoom_press_xy = None

    def _zoom_release(event):
        nonlocal zoom_press_xy, pending_zoom_autoset
        tb = plt.get_current_fig_manager().toolbar
        if (
            tb is not None
            and tb.mode == 'zoom rect'
            and zoom_press_xy is not None
            and event.button == 1
            and event.inaxes in (ax1, ax3)
        ):
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
                orig_home(*args, **kwargs)
                ax2.set_autoscaley_on(True)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_xlim(ax1.get_xlim())
                fig.canvas.draw_idle()

            toolbar.home = _home
        if hasattr(toolbar, "zoom"):
            toolbar.zoom()

    fig.canvas.manager.set_window_title(image_bname + " - " + image_save_date_time)

    plt.show()

    final_heights = [info[0] for info in selected_slots if info is not None]
    if len(final_heights) == 1:
        return final_heights[0], time_since_start - imaging_duration  # Return the single selected height and time before end that line was imaged
    elif len(final_heights) == 2:
        return final_heights[0], final_heights[1], time_since_start - imaging_duration  # Return both selected heights and time before end that line was imaged

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
