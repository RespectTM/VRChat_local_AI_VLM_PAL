"""
Windows Graphics Capture — pure ctypes COM/WinRT interop.

Captures a window's own rendered pixels from the GPU compositor even when
the window is completely hidden behind other windows.

Public API:
    capture_hwnd(hwnd, out_path, timeout=10.0)
    capture_title(title, out_path, timeout=10.0)
"""

import ctypes
import ctypes.wintypes
import os
from PIL import Image
from capture_wgc import capture_window_wgc


# --- Window title helpers ----------------------------------------------------

def _title_from_hwnd(hwnd: int) -> str:
    """Return the window title for a given HWND."""
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


# --- Public API --------------------------------------------------------------

def capture_hwnd(hwnd: int, out_path: str, timeout: float = 10.0) -> None:
    """
    Capture the window identified by hwnd into out_path (PNG).
    Uses Windows Graphics Capture -- works even when fully occluded.
    """
    title = _title_from_hwnd(hwnd)
    if not title:
        raise RuntimeError(f"Could not get window title for hwnd={hwnd}")
    capture_window_wgc(title, out_path, timeout=timeout)


def capture_title(title: str, out_path: str, timeout: float = 10.0) -> None:
    """
    Capture the window with the given title into out_path (PNG).
    Uses Windows Graphics Capture -- works even when fully occluded.
    """
    capture_window_wgc(title, out_path, timeout=timeout)


# --- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime
    from capture_vrchat import find_vrchat_window

    hwnd, title = find_vrchat_window()
    if not hwnd:
        print("No VRChat window found.")
    else:
        out_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(out_dir, f"vrchat_wgc_native_{ts}.png")
        capture_hwnd(hwnd, out)
        print(f"Saved WGC capture to {out}")
        img = Image.open(out)
        print(f"Dimensions: {img.width}x{img.height}")
