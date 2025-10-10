import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import path_loader as pl
csv_path = pl.deflation_curve_path


def load_csv(path):
	times = []
	defs = []
	with open(path, 'r', newline='') as fh:
		reader = csv.reader(fh)
		header = next(reader, None)
		for row in reader:
			if not row:
				continue
			try:
				t = float(row[0])
				d = float(row[1])
			except Exception:
				# skip malformed rows
				continue
			times.append(t)
			defs.append(d)
	return np.array(times), np.array(defs)


def cumulative_linear_fit(x, y):
	"""Compute cumulative slopes and R^2 for linear fits using points up to each index.

	Returns arrays slope_up_to_i, r2_up_to_i (same length as x)
	"""
	n = len(x)
	slopes = np.full(n, np.nan)
	r2s = np.full(n, np.nan)

	for i in range(1, n):
		xi = x[: i + 1]
		yi = y[: i + 1]
		# fit linear model y = m*x + b
		A = np.vstack([xi, np.ones_like(xi)]).T
		try:
			m, b = np.linalg.lstsq(A, yi, rcond=None)[0]
		except Exception:
			slopes[i] = np.nan
			r2s[i] = np.nan
			continue
		slopes[i] = m
		# compute r^2
		y_pred = m * xi + b
		ss_res = np.sum((yi - y_pred) ** 2)
		ss_tot = np.sum((yi - np.mean(yi)) ** 2)
		r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0
		r2s[i] = r2

	# For the first point (index 0) slope is undefined; keep nan or set 0
	return slopes, r2s


def plot_results(times, defs, slopes, r2s):
	fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9), gridspec_kw={'height_ratios': [3, 1, 1]})

	# Scatter plot of deflection vs time
	axes[0].scatter(times, defs, s=20, color='tab:blue', alpha=0.7)
	axes[0].set_ylabel('Deflection (nm)')
	axes[0].set_title('Deflection vs Time')

	# R^2 plot
	axes[1].scatter(times, r2s, s=10, color='tab:orange')
	axes[1].set_ylabel(r'$R^2$')
	# Adapt y-limits for R^2: if R^2 occupies a narrow band, zoom in to improve resolution
	r2_valid = r2s[~np.isnan(r2s)]
	if r2_valid.size > 0:
		r2_min = float(np.min(r2_valid))
		r2_max = float(np.max(r2_valid))
		r2_span = r2_max - r2_min
		# If the span is small (doesn't cover most of 0-1), zoom into the data
		if r2_span < 0.6:
			# Add a small padding relative to the span (or a minimum absolute padding)
			pad = max(0.02, 0.05 * max(r2_span, 0.001))
			ymin = max(0.0, r2_min - pad)
			ymax = min(1.0, r2_max + pad)
			axes[1].set_ylim(ymin, ymax)
		else:
			axes[1].set_ylim(-0.05, 1.05)
	else:
		axes[1].set_ylim(-0.05, 1.05)

	# Slope plot
	axes[2].scatter(times, slopes, s=10, color='tab:green')
	axes[2].set_ylabel('Slope (nm/min)')
	axes[2].set_xlabel('Time (minutes)')

	try:
		x_left = -0.1
		# Apply left limit to the top axes (sharex=True will propagate)
		axes[0].set_xlim(left=x_left)
	except Exception:
		# if times invalid, skip
		pass

	# Interactive click handling: highlight selected index across all subplots and show annotation.
	prev_artists = []   # previously drawn markers/annotations to remove
	def clear_prev():
		nonlocal prev_artists
		for art in prev_artists:
			try:
				art.remove()
			except Exception:
				pass
		prev_artists = []

	def on_click(event):
		# Only respond to clicks inside axes and with valid xdata
		if event.inaxes is None or event.xdata is None:
			return
		# find nearest index by time
		try:
			idx = int(np.argmin(np.abs(times - event.xdata)))
		except Exception:
			return
		# ensure idx in range
		if idx < 0 or idx >= len(times):
			return

		t = float(times[idx])
		d = float(defs[idx])
		s = float(slopes[idx]) if not np.isnan(slopes[idx]) else float('nan')
		r2 = float(r2s[idx]) if not np.isnan(r2s[idx]) else float('nan')

		# remove previous highlights
		clear_prev()

		# highlight on deflection plot
		m0, = axes[0].plot(t, d, marker='o', color='red', markersize=9, markeredgecolor='white', zorder=11)
		prev_artists.append(m0)

		# highlight on R^2 plot (may be nan)
		if not np.isnan(r2):
			m1, = axes[1].plot(t, r2, marker='o', color='red', markersize=7, markeredgecolor='white', zorder=11)
			prev_artists.append(m1)

		# highlight on slope plot (may be nan)
		if not np.isnan(s):
			m2, = axes[2].plot(t, s, marker='o', color='red', markersize=7, markeredgecolor='white', zorder=11)
			prev_artists.append(m2)

		# Create annotation text and attach to deflection axes
		txt = f"t={t:.6g} min\ndef={d:.6g} nm\nslope={s:.6g} nm/min\nR²={r2:.6g}"
		# position annotation slightly above the point (adaptive offset)
		yspan = np.nanmax(defs) - np.nanmin(defs) if defs.size > 1 else 1.0
		offset = max(0.05 * (yspan if yspan != 0 else 1.0), 1e-6)
		ann = axes[0].annotate(txt, xy=(t, d), xytext=(t, d + offset),
		                       arrowprops=dict(arrowstyle="->", color='red'),
		                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
		                       fontsize=9, zorder=12)
		prev_artists.append(ann)

		# Print to console
		print(f"Selected index {idx}: time={t:.6g}, deflection={d:.6g}, slope={s:.6g}, R^2={r2:.6g}")

		# redraw
		try:
			fig.canvas.draw_idle()
		except Exception:
			pass

	# connect the click event
	fig.canvas.mpl_connect('button_press_event', on_click)

	plt.tight_layout()
	plt.show()


print(f"Loading data from {csv_path}")
if not os.path.exists(csv_path):
	print(f"CSV file not found: {csv_path}")
	sys.exit(1)

times, defs = load_csv(csv_path)
if len(times) == 0:
	print("No data found in CSV")
	sys.exit(1)

slopes, r2s = cumulative_linear_fit(times, defs)
plot_results(times, defs, slopes, r2s)