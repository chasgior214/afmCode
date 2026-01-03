import os
import webbrowser

try:
    IP_address = os.environ["UbuntuLaptopTailscaleIP"]
    print(f"Opening dashboard at http://{IP_address}:5000")
    webbrowser.open(f"http://{IP_address}:5000")
except:
    print("UbuntuLaptopTailscaleIP environment variable not found")
    print("Opening dashboard at http://localhost:5000")
    webbrowser.open("http://localhost:5000")