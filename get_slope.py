import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import tkinter as tk
from tkinter import ttk

import path_loader as pl
csv_path = pl.deflation_curve_path
deflation_curve_slope_path = pl.deflation_curve_slope_path
from path_loader import get_deflation_curve_slope_id

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

# helper to read and format saved slope info by ID
def get_saved_slope_text(deflation_curve_slope_id):
	try:
		target = str(deflation_curve_slope_id)
	except Exception:
		target = f"{deflation_curve_slope_id}"
	deflation_curve_slope_path = pl.deflation_curve_slope_path
	if not os.path.exists(deflation_curve_slope_path):
		return "No saved slope"

	matches = []
	try:
		with open(deflation_curve_slope_path, 'r', newline='') as fh:
			reader = csv.reader(fh)
			header = next(reader, None)
			col_map = {}
			if header:
				col_map = {name: idx for idx, name in enumerate(header)}
			slope_idx = col_map.get('slope_nm_per_min', 1)
			r2_idx = col_map.get('r_squared', 2)
			intercept_idx = col_map.get('y_intercept_nm', 4)
			timestamp_idx = col_map.get('timestamp', 3)
			for row in reader:
				if not row or len(row) < 4:
					continue
				if str(row[0]) == target:
					slope_val = row[slope_idx] if slope_idx < len(row) else ''
					r2_val = row[r2_idx] if r2_idx < len(row) else ''
					intercept_val = row[intercept_idx] if intercept_idx < len(row) else ''
					saved_val = row[timestamp_idx] if timestamp_idx < len(row) else ''
					# try numeric formatting
					try:
						slope_val = f"{float(slope_val):.6g}"
					except Exception:
						pass
					try:
						r2_val = f"{float(r2_val):.6g}"
					except Exception:
						pass
					try:
						intercept_val = f"{float(intercept_val):.6g}"
					except Exception:
						pass
					matches.append((slope_val, r2_val, intercept_val, saved_val))
	except Exception:
		return "No saved slope"

	if not matches:
		return "No saved slope"

	# If multiple matches, show them separated by a blank line
	blocks = []
	for slope_val, r2_val, intercept_val, saved_val in matches:
		intercept_line = f"\nintercept (nm)={intercept_val}" if intercept_val not in (None, '') else ''
		blocks.append(f"slope (nm/min)={slope_val}\nR^2={r2_val}{intercept_line}\nsaved={saved_val}")
	return "\n\n".join(blocks)

# plotting function with interactive selection and slope saving
def plot_deflection_curve(curve_path, deflation_curve_slope_id):

	# compute cumulative linear fit slopes and R^2
	def cumulative_linear_fit(x, y):
		"""Compute cumulative slopes and R^2 for linear fits using points up to each index.

		Returns arrays slope_up_to_i, r2_up_to_i, intercepts_up_to_i (same length as x)
		"""
		n = len(x)
		slopes = np.full(n, np.nan)
		r2s = np.full(n, np.nan)
		intercepts = np.full(n, np.nan)

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
				intercepts[i] = np.nan
				continue
			slopes[i] = m
			intercepts[i] = b
			# compute r^2
			y_pred = m * xi + b
			ss_res = np.sum((yi - y_pred) ** 2)
			ss_tot = np.sum((yi - np.mean(yi)) ** 2)
			r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0
			r2s[i] = r2

		# For the first point (index 0) slope is undefined; keep nan or set 0
		return slopes, r2s, intercepts

	# compute slope to previous point and slope from local 3-point regression
	def adjacent_slopes(x, y):
		n = len(x)
		slope_prev = np.full(n, np.nan)
		slope_local3 = np.full(n, np.nan)

		# slope between current point and previous point
		for i in range(1, n):
			dx = x[i] - x[i - 1]
			if dx != 0:
				slope_prev[i] = (y[i] - y[i - 1]) / dx
			else:
				slope_prev[i] = np.nan

		# slope from linear regression over previous, current, next (3-point window)
		for i in range(1, n - 1):
			xi = x[i - 1: i + 2]
			yi = y[i - 1: i + 2]
			A = np.vstack([xi, np.ones_like(xi)]).T
			try:
				m, b = np.linalg.lstsq(A, yi, rcond=None)[0]
				slope_local3[i] = m
			except Exception:
				slope_local3[i] = np.nan

		return slope_prev, slope_local3

	times, defs = load_csv(curve_path)

	slopes, r2s, intercepts = cumulative_linear_fit(times, defs)
	slopes_prev, slopes_local3 = adjacent_slopes(times, defs)

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

	# Slope plot: now show three series
	axes[2].scatter(times, slopes, s=10, color='tab:green', label='cumulative fit')
	axes[2].scatter(times, slopes_prev, s=20, color='tab:purple', marker='x', label='pairwise prev')
	axes[2].scatter(times, slopes_local3, s=12, color='tab:cyan', marker='s', label='local 3-pt fit')
	axes[2].set_ylabel('Slope (nm/min)')
	axes[2].set_xlabel('Time (minutes)')
	# place legend inside the bottom-left corner of the bottom subplot
	axes[2].legend(loc='lower left', fontsize=9)

	try:
		x_left = -0.1
		# Apply left limit to the top axes (sharex=True will propagate)
		axes[0].set_xlim(left=x_left)
	except Exception:
		# if times invalid, skip
		pass

	# Interactive click handling: highlight selected index across all subplots and show annotation.
	prev_artists = []   # previously drawn markers/annotations to remove
	selected_idx = [None]  # Track currently selected index (use list to allow modification in nested function)
	
	def clear_prev():
		nonlocal prev_artists
		for art in prev_artists:
			try:
				art.remove()
			except Exception:
				pass
		prev_artists = []

	def on_click(event):
		# Only respond to clicks inside axes and with valid xdata/ydata
		if event.inaxes is None or event.xdata is None or event.ydata is None:
			return

		# Right-click (button==3): set the lower y-limit of the clicked axes to the y-value clicked
		if getattr(event, "button", None) == 3:
			ax = event.inaxes
			try:
				cur_ymin, cur_ymax = ax.get_ylim()
				new_ymin = float(event.ydata)

				# Determine which axis was clicked to decide if we auto-scale upper limit
				# axes[1] is the R² plot - keep its upper limit at 1
				if ax == axes[1]:
					# R² plot: keep upper limit fixed at current ymax (typically 1.05)
					if new_ymin >= cur_ymax:
						pad = max(0.02 * max(abs(new_ymin), 1.0), 1e-6)
						new_ymax = new_ymin + pad
					else:
						new_ymax = cur_ymax
				else:
					# For deflection (axes[0]) and slope (axes[2]) plots: auto-scale upper limit
					# Find the maximum y-value in the data that is >= new_ymin
					if ax == axes[0]:
						# deflection plot
						data_y = defs
					elif ax == axes[2]:
						# slope plot: consider all three slope series
						data_y = np.concatenate([slopes[~np.isnan(slopes)], 
												 slopes_prev[~np.isnan(slopes_prev)], 
												 slopes_local3[~np.isnan(slopes_local3)]])
					else:
						# fallback
						data_y = np.array([])
					
					# Filter data >= new_ymin
					data_above = data_y[data_y >= new_ymin]
					
					if len(data_above) > 0:
						data_max = float(np.max(data_above))
						# Add a small padding (5% of the range or a minimum)
						data_range = data_max - new_ymin
						pad = max(0.05 * data_range, 0.02 * max(abs(data_max), 1.0), 1e-6)
						new_ymax = data_max + pad
					else:
						# No data above new_ymin, use a small default range
						pad = max(0.02 * max(abs(new_ymin), 1.0), 1e-6)
						new_ymax = new_ymin + pad

				ax.set_ylim(new_ymin, new_ymax)
				try:
					fig.canvas.draw_idle()
				except Exception:
					pass
			except Exception:
				# ignore any issues setting limits
				pass
			return

		# Left-click and other buttons: selection/highlight behavior, find nearest index by time
		try:
			idx = int(np.argmin(np.abs(times - event.xdata)))
		except Exception:
			return
		# ensure idx in range
		if idx < 0 or idx >= len(times):
			return

		# Store the selected index
		selected_idx[0] = idx

		t = float(times[idx])
		d = float(defs[idx])
		s = float(slopes[idx]) if not np.isnan(slopes[idx]) else float('nan')
		r2 = float(r2s[idx]) if not np.isnan(r2s[idx]) else float('nan')
		s_prev = float(slopes_prev[idx]) if not np.isnan(slopes_prev[idx]) else float('nan')
		s_local3 = float(slopes_local3[idx]) if not np.isnan(slopes_local3[idx]) else float('nan')
		b = float(intercepts[idx]) if not np.isnan(intercepts[idx]) else float('nan')

		# remove previous highlights
		clear_prev()

		# highlight on deflection plot
		m0, = axes[0].plot(t, d, marker='o', color='red', markersize=9, markeredgecolor='white', zorder=11)
		prev_artists.append(m0)

		# Plot the line of best fit on the deflection plot if we have valid slope and intercept
		if not np.isnan(s) and not np.isnan(b):
			# Get x-axis limits to extend the line across the visible range
			x_min, x_max = axes[0].get_xlim()
			line_x = np.array([x_min, x_max])
			line_y = s * line_x + b

			# Freeze and restore y-limits so plotting doesn't change them
			ymin, ymax = axes[0].get_ylim()
			line, = axes[0].plot(line_x, line_y, '--', color='red', linewidth=1.5, alpha=0.7, zorder=10, label='Best fit line')
			axes[0].set_ylim(ymin, ymax)

			prev_artists.append(line)

		# highlight on R^2 plot (may be nan)
		if not np.isnan(r2):
			m1, = axes[1].plot(t, r2, marker='o', color='red', markersize=7, markeredgecolor='white', zorder=11)
			prev_artists.append(m1)

		# highlight on slope plot (may be nan)
		if not np.isnan(s):
			m2, = axes[2].plot(t, s, marker='o', color='tab:green', markersize=9, markeredgecolor='white', zorder=11)
			prev_artists.append(m2)
		if not np.isnan(s_prev):
			m3, = axes[2].plot(t, s_prev, marker='x', color='tab:purple', markersize=10, markeredgecolor='white', zorder=12)
			prev_artists.append(m3)
		if not np.isnan(s_local3):
			m4, = axes[2].plot(t, s_local3, marker='s', color='tab:cyan', markersize=8, markeredgecolor='white', zorder=12)
			prev_artists.append(m4)

		# Create annotation text and attach to deflection axes
		txt = (
			"Selected Point\n=========\n"
			f"t={t:.6g} min\n"
			f"def={d:.6g} nm\n"
			f"cumulative_slope={s:.6g} nm/min\n"
			f"intercept={b:.6g} nm\n"
			f"R²={r2:.6g}\n"
			f"pairwise_prev_slope={s_prev:.6g} nm/min\n"
			f"local3_slope={s_local3:.6g} nm/min"
		)
		# Append "Last Saved" section
		saved_txt = get_saved_slope_text(deflation_curve_slope_id)
		txt += "\n\nLast Saved\n=======\n" + saved_txt

		ann = axes[0].annotate(
			txt,
			xy=(0.02, 0.02),                # position in axes fraction (bottom-left with small margin)
			xycoords='axes fraction',
			xytext=None,
			textcoords='axes fraction',
			horizontalalignment='left',
			verticalalignment='bottom',
			bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
			fontsize=9,
			zorder=12
		)
		prev_artists.append(ann)

		# Print to console
		print(f"Selected index {idx}: time={t:.6g}, deflection={d:.6g}, cumulative_slope={s:.6g}, intercept={b:.6g}, pairwise_prev={s_prev:.6g}, local3={s_local3:.6g}, R^2={r2:.6g}")

		# redraw
		try:
			fig.canvas.draw_idle()
		except Exception:
			pass

	def on_key(event):
		"""Handle key press events. Space bar saves the currently selected slope."""
		if event.key == ' ' and selected_idx[0] is not None:
			idx = selected_idx[0]
			if idx < 0 or idx >= len(times):
				return

			s = float(slopes[idx]) if not np.isnan(slopes[idx]) else float('nan')
			r2 = float(r2s[idx]) if not np.isnan(r2s[idx]) else float('nan')
			b = float(intercepts[idx]) if not np.isnan(intercepts[idx]) else float('nan')
			
			# Get current timestamp
			timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			
			# Create CSV file with headers if it doesn't exist
			file_exists = os.path.exists(deflation_curve_slope_path)
			if not file_exists:
				try:
					os.makedirs(os.path.dirname(deflation_curve_slope_path), exist_ok=True)
				except Exception:
					pass
			
			# Append the new row
			try:
				with open(deflation_curve_slope_path, 'a', newline='') as fh:
					writer = csv.writer(fh)
					if not file_exists:
						writer.writerow(['id', 'slope_nm_per_min', 'r_squared', 'timestamp', 'y_intercept_nm'])
					writer.writerow([deflation_curve_slope_id, s, r2, timestamp, b])
				print(f"Saved: id={deflation_curve_slope_id}, slope={s:.6g}, intercept={b:.6g}, R²={r2:.6g}, time={timestamp}")
			
				# Update the annotation to show the newly saved value
				if prev_artists:
					# Remove old annotation and recreate with updated "Last Saved" info
					clear_prev()
					t = float(times[idx])
					d = float(defs[idx])
					s_prev = float(slopes_prev[idx]) if not np.isnan(slopes_prev[idx]) else float('nan')
					s_local3 = float(slopes_local3[idx]) if not np.isnan(slopes_local3[idx]) else float('nan')
					
					# Get intercept for this index (needed to draw best-fit line)
					b = float(intercepts[idx]) if not np.isnan(intercepts[idx]) else float('nan')
					
					# Re-highlight all markers
					m0, = axes[0].plot(t, d, marker='o', color='red', markersize=9, markeredgecolor='white', zorder=11)
					prev_artists.append(m0)
					if not np.isnan(r2):
						m1, = axes[1].plot(t, r2, marker='o', color='red', markersize=7, markeredgecolor='white', zorder=11)
						prev_artists.append(m1)
					if not np.isnan(s):
						m2, = axes[2].plot(t, s, marker='o', color='tab:green', markersize=9, markeredgecolor='white', zorder=11)
						prev_artists.append(m2)
					if not np.isnan(s_prev):
						m3, = axes[2].plot(t, s_prev, marker='x', color='tab:purple', markersize=10, markeredgecolor='white', zorder=12)
						prev_artists.append(m3)
					if not np.isnan(s_local3):
						m4, = axes[2].plot(t, s_local3, marker='s', color='tab:cyan', markersize=8, markeredgecolor='white', zorder=12)
						prev_artists.append(m4)
					
					# Plot the line of best fit
					if not np.isnan(s) and not np.isnan(b):
						x_min, x_max = axes[0].get_xlim()
						line_x = np.array([x_min, x_max])
						line_y = s * line_x + b

						# Freeze and restore y-limits so plotting doesn't change them
						ymin, ymax = axes[0].get_ylim()
						line, = axes[0].plot(line_x, line_y, '--', color='red', linewidth=1.5, alpha=0.7, zorder=10, label='Best fit line')
						axes[0].set_ylim(ymin, ymax)

						prev_artists.append(line)
					
					# Recreate annotation with updated saved info
					txt = (
						"Selected Point\n=========\n"
						f"t={t:.6g} min\n"
						f"def={d:.6g} nm\n"
						f"cumulative_slope={s:.6g} nm/min\n"
						f"intercept={b:.6g} nm\n"
						f"R²={r2:.6g}\n"
						f"pairwise_prev_slope={s_prev:.6g} nm/min\n"
						f"local3_slope={s_local3:.6g} nm/min"
					)
					saved_txt = get_saved_slope_text(deflation_curve_slope_id)
					txt += "\n\nLast Saved\n=======\n" + saved_txt
					
					ann = axes[0].annotate(
						txt,
						xy=(0.02, 0.02),
						xycoords='axes fraction',
						xytext=None,
						textcoords='axes fraction',
						horizontalalignment='left',
						verticalalignment='bottom',
						bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
						fontsize=9,
						zorder=12
					)
					prev_artists.append(ann)
					
					try:
						fig.canvas.draw_idle()
					except Exception:
						pass
						
			except Exception as e:
				print(f"Error saving slope: {e}")

	# connect the click event
	fig.canvas.mpl_connect('button_press_event', on_click)
	# connect the key press event
	fig.canvas.mpl_connect('key_press_event', on_key)

	plt.tight_layout()
	
	# Maximize the window
	try:
		manager = plt.get_current_fig_manager()
		manager.window.state('zoomed')  # For TkAgg backend
	except Exception:
		try:
			manager = plt.get_current_fig_manager()
			manager.window.showMaximized()  # For Qt backends
		except Exception:
			pass  # If maximizing fails, just show normally
	
	plt.show()



csv_paths = pl.get_all_deflation_curve_paths()

filter_by_sample = '37' # set to None to disable filtering
filter_by_transfer_location = '$(6,3)' # set to None to disable filtering
filter_by_cavity_position = 'blue' # set to None to disable filtering
filter_by_depressurized_date = None # set to None to disable filtering, or 'YYYYMMDD' string
filter_by_depressurized_time = None # set to None to disable filtering, or 'HHMMSS' string
filter_at_least_two_points = True  # if True, only show CSVs with at least 2 data points
filter_at_least_two_positive_points = True  # if True, only show CSVs with at least 2 positive deflection points

if filter_by_sample is not None:
	csv_paths = [p for p in csv_paths if f"_sample{filter_by_sample}_" in os.path.basename(p)]
if filter_by_transfer_location is not None:
	csv_paths = [p for p in csv_paths if f"_loc{filter_by_transfer_location}_" in os.path.basename(p)]
if filter_by_cavity_position is not None:
	csv_paths = [p for p in csv_paths if f"_cav{filter_by_cavity_position}" in os.path.basename(p)]
if filter_by_depressurized_date is not None:
	csv_paths = [p for p in csv_paths if f"_depressurized{filter_by_depressurized_date}_" in os.path.basename(p)]
if filter_by_depressurized_time is not None:
	csv_paths = [p for p in csv_paths if f"_{filter_by_depressurized_time}_" in os.path.basename(p)]
if filter_at_least_two_points:
	filtered_paths = []
	for p in csv_paths:
		times, defs = load_csv(p)
		if len(times) >= 2:
			filtered_paths.append(p)
	csv_paths = filtered_paths

if filter_at_least_two_positive_points:
	filtered_paths = []
	for p in csv_paths:
		times, defs = load_csv(p)
		if sum(d > 0 for d in defs) >= 2:
			filtered_paths.append(p)
	csv_paths = filtered_paths

def csv_path_to_slope_id(path):
	# Expect filenames like:
	# deflation_curve_sample{sample_number}_depressurized{depressurized_date}_{depressurized_time}_loc{transfer_location}_cav{cavity_position}.csv
	base = os.path.basename(path)

	# Use regex to robustly capture each component, allowing non-underscore characters for fields.
	pat = re.compile(
		r"^deflation_curve_sample(?P<sample>[^_]+)"
		r"_depressurized(?P<date>[^_]+)_(?P<time>[^_]+)"
		r"_loc(?P<loc>[^_]+)_cav(?P<cav>[^.]+)\.csv$"
	)
	m = pat.match(base)
	if not m:
		# helpful debug message
		raise ValueError(f"Filename does not match expected pattern: {base}")

	sample_number = m.group('sample')
	depressurized_date = m.group('date')
	depressurized_time = m.group('time')
	transfer_location = m.group('loc')
	cavity_position = m.group('cav')

	# call existing helper to get slope id
	return get_deflation_curve_slope_id(sample_number, depressurized_date, depressurized_time, transfer_location, cavity_position)

def load_saved_slopes():
	"""Load all saved slopes into a dictionary keyed by slope_id."""
	slopes_dict = {}
	if not os.path.exists(deflation_curve_slope_path):
		return slopes_dict
	
	try:
		with open(deflation_curve_slope_path, 'r', newline='') as fh:
			reader = csv.reader(fh)
			header = next(reader, None)
			for row in reader:
				if not row or len(row) < 4:
					continue
				slope_id = str(row[0])
				slope_val = row[1]
				r2_val = row[2]
				# Keep the most recent entry for each id
				slopes_dict[slope_id] = (slope_val, r2_val)
	except Exception:
		pass
	
	return slopes_dict

def parse_csv_metadata(path):
	"""Extract metadata from CSV filename."""
	base = os.path.basename(path)
	pat = re.compile(
		r"^deflation_curve_sample(?P<sample>[^_]+)"
		r"_depressurized(?P<date>[^_]+)_(?P<time>[^_]+)"
		r"_loc(?P<loc>[^_]+)_cav(?P<cav>[^.]+)\.csv$"
	)
	m = pat.match(base)
	if not m:
		return None
	
	return {
		'sample': m.group('sample'),
		'date': m.group('date'),
		'time': m.group('time'),
		'loc': m.group('loc'),
		'cav': m.group('cav')
	}

def show_csv_table(csv_paths, filter_by_sample, filter_by_transfer_location, 
				   filter_by_cavity_position, filter_by_depressurized_date, 
				   filter_by_depressurized_time):
	"""Display a tkinter table with CSV files and allow loading."""
	
	# Load saved slopes
	saved_slopes = load_saved_slopes()
	
	# Parse metadata for all paths
	table_data = []
	for idx, path in enumerate(csv_paths):
		metadata = parse_csv_metadata(path)
		if metadata is None:
			continue
		
		slope_id = csv_path_to_slope_id(path)
		slope_val, r2_val = '', ''
		if slope_id in saved_slopes:
			slope_val, r2_val = saved_slopes[slope_id]
			try:
				slope_val = f"{float(slope_val):.6g}"
				r2_val = f"{float(r2_val):.6g}"
			except Exception:
				pass
		
		table_data.append({
			'index': idx,
			'path': path,
			'sample': metadata['sample'],
			'loc': metadata['loc'],
			'cav': metadata['cav'],
			'date': metadata['date'],
			'time': metadata['time'],
			'slope': slope_val,
			'r2': r2_val,
			'slope_id': slope_id
		})
	
	# Determine which columns to show
	columns = ['index']
	if filter_by_sample is None:
		columns.append('sample')
	if filter_by_transfer_location is None:
		columns.append('loc')
	if filter_by_cavity_position is None:
		columns.append('cav')
	if filter_by_depressurized_date is None:
		columns.append('date')
	if filter_by_depressurized_time is None:
		columns.append('time')
	columns.extend(['slope', 'r2'])
	
	# Create tkinter window
	root = tk.Tk()
	root.title("Deflation Curve Selector")
	root.geometry("900x600")
	
	# Create frame for table
	frame = ttk.Frame(root)
	frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
	
	# Create treeview with scrollbars
	tree_scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL)
	tree_scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
	
	tree = ttk.Treeview(frame, columns=columns, show='headings',
						yscrollcommand=tree_scroll_y.set,
						xscrollcommand=tree_scroll_x.set)
	
	tree_scroll_y.config(command=tree.yview)
	tree_scroll_x.config(command=tree.xview)
	tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
	tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
	tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	
	# Configure column headings and widths
	column_headings = {
		'index': 'Index',
		'sample': 'Sample',
		'loc': 'Location',
		'cav': 'Cavity',
		'date': 'Date',
		'time': 'Time',
		'slope': 'Saved Slope (nm/min)',
		'r2': 'Saved R²'
	}
	
	column_widths = {
		'index': 60,
		'sample': 70,
		'loc': 100,
		'cav': 70,
		'date': 90,
		'time': 80,
		'slope': 150,
		'r2': 100
	}
	
	for col in columns:
		tree.heading(col, text=column_headings[col])
		tree.column(col, width=column_widths[col], anchor=tk.CENTER)
	
	# Insert data
	for item in table_data:
		values = [item[col] for col in columns]
		# use the index as the item's iid so we can update values later
		tree.insert('', tk.END, iid=str(item['index']), values=values, tags=(item['path'], item['slope_id']))
	
	# Create button frame
	button_frame = ttk.Frame(root)
	button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
	
	def on_load():
		selection = tree.selection()
		if not selection:
			return
		
		item = tree.item(selection[0])
		tags = item['tags']
		if not tags:
			return
		
		csv_path = tags[0]
		slope_id = tags[1]
		
		# Hide the tkinter window, plot, then re-show and refresh saved values
		try:
			root.withdraw()
			# ensure the window updates before blocking on matplotlib
			root.update_idletasks()
		except Exception:
			pass

		# show the plot (blocking until the matplotlib window is closed)
		plot_deflection_curve(csv_path, slope_id)

		# After the plot window is closed, re-show the tkinter window
		try:
			# refresh saved slopes and update table values
			new_saved = load_saved_slopes()
			for td in table_data:
				iid = str(td['index'])
				# recompute displayed slope/r2 from saved file
				new_slope, new_r2 = '', ''
				if td['slope_id'] in new_saved:
					new_slope, new_r2 = new_saved[td['slope_id']]
					try:
						new_slope = f"{float(new_slope):.6g}"
						new_r2 = f"{float(new_r2):.6g}"
					except Exception:
						pass
				# build new values list in same column order
				new_vals = [td[col] for col in columns]
				# find positions of slope/r2 columns and replace if present
				if 'slope' in columns:
					new_vals[columns.index('slope')] = new_slope
				if 'r2' in columns:
					new_vals[columns.index('r2')] = new_r2
				# update tree row
				try:
					tree.item(iid, values=new_vals)
				except Exception:
					pass

			root.deiconify()
			root.update_idletasks()
		except Exception:
			# if anything fails, at least try to restore the window
			try:
				root.deiconify()
			except Exception:
				pass
	
	load_button = ttk.Button(button_frame, text="Load", command=on_load)
	load_button.pack(side=tk.LEFT, padx=5)
	
	# Allow double-click to load
	tree.bind('<Double-Button-1>', lambda e: on_load())

	# Allow pressing Enter to load selected row (both Return and keypad Enter)
	root.bind('<Return>', lambda e: on_load())
	root.bind('<KP_Enter>', lambda e: on_load())
	
	root.mainloop()

# Show the table instead of directly plotting
show_csv_table(csv_paths, filter_by_sample, filter_by_transfer_location,
			   filter_by_cavity_position, filter_by_depressurized_date,
			   filter_by_depressurized_time)

