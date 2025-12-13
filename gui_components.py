import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

class DualHandleSlider:
    """A two-ended slider widget for selecting a numeric range.

    The widget renders a colored band between two draggable handles to represent the
    active interval. The left handle is controlled with the primary mouse button and
    the right handle with the secondary button, allowing users to adjust either bound
    without modifier keys. Listeners registered through :meth:`on_changed` receive
    updates whenever the selected range changes.
    """

    def __init__(
        self,
        ax,
        label,
        valmin,
        valmax,
        valinit,
        cmap,
        *,
        value_format="{:.3f}",
        bounds_format=None,
    ):
        """Create a dual-handle slider.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which the slider should be drawn. Tick labels and spines are
            hidden during initialization to create a minimalist widget.
        label : str
            Text label displayed above the slider.
        valmin, valmax : float
            Minimum and maximum allowable values for the handles. Non-finite values
            are replaced with sensible defaults to avoid matplotlib warnings.
        valinit : tuple[float, float]
            Initial (lower, upper) values. The values are sorted and clamped into the
            provided bounds so the slider always starts in a valid state.
        cmap : matplotlib.colors.Colormap
            Colormap used to render the gradient band between the handles.
        value_format : str, optional
            Format string applied to the handle labels. Defaults to three decimal
            places.
        bounds_format : str, optional
            Format string for rendering the bounds; falls back to ``value_format``
            when omitted.
        """
        self.ax = ax
        self.label = label
        self.cmap = cmap
        self.value_format = value_format
        self.bounds_format = bounds_format or value_format
        self.ax.set_facecolor('none')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_ylim(-0.65, 1.1)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self._gradient_points = 256
        self._gradient_data = np.linspace(0, 1, self._gradient_points).reshape(1, -1)
        self._observers = []
        self._active_handle = None
        self._active_button = None
        valmin = float(valmin)
        valmax = float(valmax)
        if not np.isfinite(valmin):
            valmin = 0.0
        if not np.isfinite(valmax):
            valmax = valmin + 1.0
        if valmin == valmax:
            valmax = valmin + 1e-9
        self.lower_bound = min(valmin, valmax)
        self.upper_bound = max(valmin, valmax)
        init_low, init_high = valinit
        init_low = float(init_low)
        init_high = float(init_high)
        if init_low > init_high:
            init_low, init_high = init_high, init_low
        self.val = (
            float(np.clip(init_low, self.lower_bound, self.upper_bound)),
            float(np.clip(init_high, self.lower_bound, self.upper_bound)),
        )
        self.ax.set_xlim(self.lower_bound, self.upper_bound)
        self.label_artist = self.ax.text(
            0.5,
            1.02,
            self.label,
            transform=self.ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=9,
        )
        self.gradient_im = self.ax.imshow(
            self._gradient_data,
            extent=(self.val[0], self.val[1], 0, 1),
            aspect='auto',
            cmap=self.cmap,
            vmin=0.0,
            vmax=1.0,
            zorder=1,
            interpolation='nearest',
        )
        self.left_rect = patches.Rectangle(
            (self.lower_bound, 0),
            max(0.0, self.val[0] - self.lower_bound),
            1,
            color='black',
            zorder=2,
        )
        self.ax.add_patch(self.left_rect)
        self.right_rect = patches.Rectangle(
            (self.val[1], 0),
            max(0.0, self.upper_bound - self.val[1]),
            1,
            color='white',
            zorder=2,
        )
        self.ax.add_patch(self.right_rect)
        handle_color = 'white'
        self.handle_lines = [
            self.ax.axvline(self.val[0], color=handle_color, linewidth=2, zorder=3),
            self.ax.axvline(self.val[1], color=handle_color, linewidth=2, zorder=3),
        ]
        self.value_texts = [
            self.ax.text(
                self.val[0],
                -0.12,
                self.value_format.format(self.val[0]),
                ha='center',
                va='top',
                color='white',
                fontsize=9,
                zorder=4,
                bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'),
            ),
            self.ax.text(
                self.val[1],
                -0.12,
                self.value_format.format(self.val[1]),
                ha='center',
                va='top',
                color='black',
                fontsize=9,
                zorder=4,
                bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'),
            ),
        ]
        canvas = self.ax.figure.canvas
        self._cids = [
            canvas.mpl_connect('button_press_event', self._on_press),
            canvas.mpl_connect('button_release_event', self._on_release),
            canvas.mpl_connect('motion_notify_event', self._on_move),
        ]
        self._update_visuals(redraw=False)

    def _update_visuals(self, redraw=True):
        self.gradient_im.set_extent((self.val[0], self.val[1], 0, 1))
        self.left_rect.set_x(self.lower_bound)
        self.left_rect.set_width(max(0.0, self.val[0] - self.lower_bound))
        self.right_rect.set_x(self.val[1])
        self.right_rect.set_width(max(0.0, self.upper_bound - self.val[1]))
        for line, value in zip(self.handle_lines, self.val):
            line.set_xdata([value, value])
        for text, value in zip(self.value_texts, self.val):
            text.set_position((value, -0.12))
            text.set_text(self.value_format.format(value))
        if redraw:
            self.ax.figure.canvas.draw_idle()

    def _snap_to_bounds(self, pos):
        span = self.upper_bound - self.lower_bound
        if span <= 0:
            return pos
        threshold = span * 0.01
        if pos - self.lower_bound <= threshold:
            return self.lower_bound
        if self.upper_bound - pos <= threshold:
            return self.upper_bound
        return pos

    def _set_value_from_position(self, handle_idx, pos):
        if pos is None:
            return
        pos = float(np.clip(self._snap_to_bounds(pos), self.lower_bound, self.upper_bound))
        if handle_idx == 0:
            self.set_val((pos, self.val[1]))
        else:
            self.set_val((self.val[0], pos))

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button not in (1, 3):
            return
        self._active_handle = 0 if event.button == 1 else 1
        self._active_button = event.button
        self._set_value_from_position(self._active_handle, event.xdata)

    def _on_move(self, event):
        if self._active_handle is None:
            return
        if event.xdata is None:
            return
        self._set_value_from_position(self._active_handle, event.xdata)

    def _on_release(self, event):
        if event.button == self._active_button:
            self._active_handle = None
            self._active_button = None

    def set_bounds(self, valmin, valmax, *, update_values=True):
        """Update the allowed range and optionally clamp current values.

        Parameters
        ----------
        valmin, valmax : float
            New minimum and maximum allowed values.
        update_values : bool, optional
            When ``True`` (default) the active handles are clamped into the updated
            bounds and visuals are refreshed immediately.
        """
        valmin = float(valmin)
        valmax = float(valmax)
        if not np.isfinite(valmin):
            valmin = 0.0
        if not np.isfinite(valmax):
            valmax = valmin + 1.0
        if valmin == valmax:
            valmax = valmin + 1e-9
        self.lower_bound = min(valmin, valmax)
        self.upper_bound = max(valmin, valmax)
        self.ax.set_xlim(self.lower_bound, self.upper_bound)
        if update_values:
            clamped = (
                float(np.clip(self.val[0], self.lower_bound, self.upper_bound)),
                float(np.clip(self.val[1], self.lower_bound, self.upper_bound)),
            )
            if clamped[0] > clamped[1]:
                clamped = (clamped[0], clamped[0])
            self.val = clamped
        self._update_visuals()

    def set_val(self, values, *, notify=True):
        """Set the slider to a new (min, max) pair.

        Values are clamped to the configured bounds, and the upper value is coerced
        to be at least the lower value. Observers are notified if the range actually
        changes and ``notify`` is ``True``.
        """
        new_min, new_max = values
        new_min = float(np.clip(new_min, self.lower_bound, self.upper_bound))
        new_max = float(np.clip(new_max, self.lower_bound, self.upper_bound))
        if new_min > new_max:
            new_max = new_min
        changed = not np.isclose([new_min, new_max], list(self.val)).all()
        self.val = (new_min, new_max)
        self._update_visuals()
        if notify and changed:
            self._notify_observers()

    def on_changed(self, func):
        """Register a callback to be called when the slider range changes.

        Parameters
        ----------
        func : callable
            Function to be called with the new (min, max) values whenever the slider
            range changes. The function should accept a single argument: the new
            range as a tuple of two floats.
        """
        self._observers.append(func)

    def _notify_observers(self):
        """Notify all registered observers about the current value."""
        for callback in self._observers:
            callback(self.val)