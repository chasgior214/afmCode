from datetime import datetime

def convert_unix_to_readable(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')

time = 1768675890
print(convert_unix_to_readable(time))