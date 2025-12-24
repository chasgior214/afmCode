import serial
import csv
from datetime import datetime

PORT = "/dev/ttyACM0"
BAUD = 115200
FLUSH_INTERVAL_SEC = 30

def make_csv_writer(sample_name, start_dt, pressure_cell):
    filename = f"sample{sample_name}_cell{pressure_cell}_pressure_log_start{start_dt}.csv"
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["timestamp_unix", "avg_pressure_kPa_gage"])
    f.flush()  # header hits disk immediately
    return f, writer

def prompt_with_default(prompt, default=None):
    if default is None:
        resp = input(f"{prompt}: ").strip()
        return resp
    resp = input(f"{prompt} [default: {default}]: ").strip()
    return resp if resp else default

def prompt_yn(prompt, default=False):
    d = "y" if default else "n"
    resp = input(f"{prompt} (y/n) [default: {d}]: ").strip().lower()
    if not resp:
        return default
    return resp == "y"

def default_pressure_cell(sample):
    if sample == "37":
        return "A"
    if sample == "53":
        return "B"
    return None

def open_csvs(active_samples, pressure_cells, start_dt):
    csv_files = {}
    csv_writers = {}
    for sample in active_samples:
        f, w = make_csv_writer(sample, start_dt, pressure_cells[sample])
        csv_files[sample] = f
        csv_writers[sample] = w
    return csv_files, csv_writers

def close_csvs(csv_files):
    for f in csv_files.values():
        try:
            f.flush()
        finally:
            f.close()

def main():
    sample0 = prompt_with_default("Enter sample on A0", "37")
    sample5 = prompt_with_default("Enter sample on A5", "53")

    active_samples = tuple(s for s in (sample0, sample5) if s != "0")

    save_csv = prompt_yn("Save data to CSV?", default=True)

    pressure_cells = {}
    if save_csv:
        for sample in active_samples:
            default = default_pressure_cell(sample)
            pressure_cells[sample] = prompt_with_default(
                f"Enter pressure cell for sample {sample}",
                default
            )

    csv_files = {}
    csv_writers = {}

    # Track file "segment" start time and day for daily rotation
    current_day = datetime.now().date()
    segment_start_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if save_csv:
        csv_files, csv_writers = open_csvs(active_samples, pressure_cells, segment_start_dt)

    last_flush = datetime.now()

    with serial.Serial(PORT, BAUD, timeout=2) as ser:
        ser.reset_input_buffer()
        try:
            while True:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue

                parts = raw.split(",")
                if len(parts) != 4:
                    continue

                try:
                    v0, p0, v5, p5 = map(float, parts)
                except ValueError:
                    continue

                now = datetime.now()

                # Daily rotation: if date changed, close current files and open new ones
                if save_csv and now.date() != current_day:
                    close_csvs(csv_files)
                    current_day = now.date()
                    segment_start_dt = now.strftime("%Y-%m-%d_%H-%M-%S")
                    csv_files, csv_writers = open_csvs(active_samples, pressure_cells, segment_start_dt)
                    last_flush = now  # reset flush timer for the new files

                ts_human = now.strftime("%Y-%m-%d %H:%M:%S")
                ts_epoch = int(now.timestamp())

                readings = []
                if sample0 != "0":
                    readings.append((sample0, v0, p0))
                if sample5 != "0":
                    readings.append((sample5, v5, p5))

                if readings:
                    print(ts_human)
                    for sample, v, p in readings:
                        print(f"\t{sample}: Avg Voltage: {v:.3f}, Avg Pressure: {p:.2f} kPa")

                if save_csv:
                    for sample, _, p in readings:
                        if sample in csv_writers:
                            csv_writers[sample].writerow([ts_epoch, f"{p:.2f}"])

                    # Timed flush (every 30 seconds)
                    if (now - last_flush).total_seconds() >= FLUSH_INTERVAL_SEC:
                        for f in csv_files.values():
                            f.flush()
                        last_flush = now

        finally:
            if save_csv:
                close_csvs(csv_files)

if __name__ == "__main__":
    main()
