import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import platform
import subprocess

import numpy as np

import AFMImage
import visualizations as vis


def _json_default(obj):
    """Convert numpy/scalar types to JSON-serializable objects."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _copy_to_clipboard(text: str):
    """
    Copy the provided text to the system clipboard.

    Returns (success: bool, error_message: str | None).
    """
    system = platform.system()
    if system == "Windows":
        cmd = ["clip"]
    elif system == "Darwin":
        cmd = ["pbcopy"]
    else:
        # Best-effort for Linux; xclip/xsel may not be present.
        cmd = ["xclip", "-selection", "clipboard"]

    try:
        subprocess.run(cmd, input=text, text=True, check=True)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{cmd[0]} failed: {exc}"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: launch_select_heights.py <path-to-file.ibw>")

    ibw_path = Path(sys.argv[1]).expanduser()
    if not ibw_path.exists():
        sys.exit(f"File not found: {ibw_path}")

    image = AFMImage.AFMImage(str(ibw_path))
    result = vis.select_heights(image)

    if not result:
        print("No selection returned; clipboard unchanged.")
        return

    serialized = json.dumps(result, indent=2, default=_json_default)
    success, error = _copy_to_clipboard(serialized)
    if success:
        print("Selection copied to clipboard.")
    else:
        print(f"Failed to copy selection to clipboard: {error}")


if __name__ == "__main__":
    main()