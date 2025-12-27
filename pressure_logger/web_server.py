from flask import Flask, render_template, jsonify, send_file, Response
import csv
import os
import glob
from datetime import datetime, timedelta
import time
import threading
import json

app = Flask(__name__)

# Shared data updated by the logger
current_data = {}
data_lock = threading.Lock()

def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def get_latest_readings():
    """Get the latest pressure readings from in_progress files."""
    script_dir = get_script_dir()
    in_progress_dir = os.path.join(script_dir, "in_progress")
    
    if not os.path.exists(in_progress_dir):
        return {}
    
    readings = {}
    csv_files = glob.glob(os.path.join(in_progress_dir, "*.csv"))
    
    for filepath in csv_files:
        try:
            # Extract sample name from filename
            filename = os.path.basename(filepath)
            # Format: sample{name}_cell{cell}_pressure_log_start{date}.csv
            if filename.startswith("sample"):
                parts = filename.split("_")
                sample_name = parts[0].replace("sample", "")
                
                # Read last line of CSV
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Skip header
                        last_line = lines[-1].strip()
                        if last_line:
                            timestamp, pressure = last_line.split(',')
                            readings[sample_name] = {
                                'timestamp': int(timestamp),
                                'pressure': float(pressure),
                                'datetime': datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
                            }
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    return readings

def get_historical_data(sample_name, hours):
    """Get historical data for a sample over a time period."""
    script_dir = get_script_dir()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cutoff_timestamp = int(cutoff_time.timestamp())
    
    data_points = []
    
    # Check both in_progress and finished directories
    for subdir in ["in_progress", "finished"]:
        dir_path = os.path.join(script_dir, subdir)
        if not os.path.exists(dir_path):
            continue
            
        pattern = os.path.join(dir_path, f"sample{sample_name}_*.csv")
        csv_files = glob.glob(pattern)
        
        for filepath in csv_files:
            try:
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        timestamp = int(row['timestamp_unix'])
                        if timestamp >= cutoff_timestamp:
                            data_points.append({
                                'timestamp': timestamp,
                                'pressure': float(row['avg_pressure_kPa_gage'])
                            })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    # Sort by timestamp
    data_points.sort(key=lambda x: x['timestamp'])
    return data_points

def downsample_data(data_points, max_points=500):
    """Downsample data to reduce number of points for plotting."""
    if len(data_points) <= max_points:
        return data_points
    
    # Take every nth point
    step = len(data_points) // max_points
    return data_points[::step]

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('dashboard.html')

@app.route('/api/current')
def api_current():
    """Get current readings for all samples."""
    readings = get_latest_readings()
    return jsonify(readings)

@app.route('/api/history/<sample_name>/<int:hours>')
def api_history(sample_name, hours):
    """Get historical data for a sample."""
    data = get_historical_data(sample_name, hours)
    downsampled = downsample_data(data)
    return jsonify(downsampled)

@app.route('/api/files')
def api_files():
    """List all finished CSV files."""
    script_dir = get_script_dir()
    finished_dir = os.path.join(script_dir, "finished")
    
    if not os.path.exists(finished_dir):
        return jsonify([])
    
    files = []
    csv_files = glob.glob(os.path.join(finished_dir, "*.csv"))
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        mod_time = os.path.getmtime(filepath)
        
        files.append({
            'filename': filename,
            'size_kb': round(file_size / 1024, 1),
            'modified': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify(files)

@app.route('/download/<filename>')
def download_file(filename):
    """Download a CSV file."""
    script_dir = get_script_dir()
    finished_dir = os.path.join(script_dir, "finished")
    filepath = os.path.join(finished_dir, filename)
    
    if os.path.exists(filepath) and filename.endswith('.csv'):
        return send_file(filepath, as_attachment=True)
    else:
        return "File not found", 404

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates."""
    def event_stream():
        while True:
            readings = get_latest_readings()
            yield f"data: {json.dumps(readings)}\n\n"
            time.sleep(5)  # Update every 5 seconds
    
    return Response(event_stream(), mimetype='text/event-stream')

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    run_server(debug=True)
