"""
Best-effort VRChat capture with three tiers:
  1. Windows Graphics Capture (WGC) — captures the game's own pixels even
     when the window is completely occluded by other windows. Requires Win10+.
  2. dxcam (DXGI Desktop Duplication) — fast GPU path; only works when the
     window is visible on screen.
  3. PrintWindow (GDI fallback) — always available but may return black for
     full-screen DirectX titles.
"""

import os
from datetime import datetime

from capture_vrchat import find_vrchat_window, get_window_rect


def _try_wgc(out_path):
    from capture_wgc import capture_window_wgc
    # WGC matches by window title; VRChat's title is literally "VRChat"
    capture_window_wgc('VRChat', out_path, timeout=10.0)


def _try_dxcam(rect, out_path):
    import dxcam
    from PIL import Image

    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    cam = dxcam.create()
    try:
        frame = cam.grab(region=(left, top, width, height))
    except Exception as ex:
        raise RuntimeError('dxcam grab failed: ' + str(ex))

    if frame is None:
        raise RuntimeError('dxcam returned empty frame — window may be occluded')

    arr = frame if frame.dtype.name == 'uint8' else frame.astype('uint8')
    # dxcam returns BGR; convert to RGB for PIL
    img = Image.fromarray(arr[:, :, ::-1])
    img.save(out_path)


def _try_printwindow(hwnd, out_path):
    import capture_vrchat as cv
    cv.capture_window_hwnd(hwnd, out_path)
    print('Saved screenshot via PrintWindow to', out_path)


def main():
    hwnd, title = find_vrchat_window()
    if not hwnd:
        print('No VRChat window found — make sure VRChat is running.')
        return

    out_dir = os.path.join(os.getcwd(), 'captures')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Tier 1: Windows Graphics Capture (works even when occluded) ---
    out_path = os.path.join(out_dir, f'vrchat_wgc_{timestamp}.png')
    try:
        _try_wgc(out_path)
        print('Saved WGC capture (occluded-window-safe) to', out_path)
        return
    except Exception as e:
        print(f'WGC failed: {e}')

    # --- Tier 2: dxcam (visible region, GPU-accelerated) ---
    out_path = os.path.join(out_dir, f'vrchat_dx_{timestamp}.png')
    try:
        rect = get_window_rect(hwnd)
        _try_dxcam(rect, out_path)
        print('Saved dxcam capture to', out_path)
        return
    except Exception as e:
        print(f'dxcam failed: {e}')

    # --- Tier 3: PrintWindow (GDI fallback) ---
    out_path = os.path.join(out_dir, f'vrchat_pw_{timestamp}.png')
    try:
        _try_printwindow(hwnd, out_path)
    except Exception as e:
        print(f'PrintWindow also failed: {e}')


if __name__ == '__main__':
    main()
