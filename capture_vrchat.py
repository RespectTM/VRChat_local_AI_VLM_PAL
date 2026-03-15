import ctypes
from ctypes import wintypes
from PIL import ImageGrab
import os
from datetime import datetime

USER32 = ctypes.WinDLL('user32', use_last_error=True)

EnumWindows = USER32.EnumWindows
EnumWindows.restype = wintypes.BOOL

GetWindowText = USER32.GetWindowTextW
GetWindowText.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GetWindowText.restype = ctypes.c_int

IsWindowVisible = USER32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

GetWindowRect = USER32.GetWindowRect
GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
GetWindowRect.restype = wintypes.BOOL


def list_windows():
    windows = []

    # Define callback type for EnumWindows
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _enum(hwnd, lParam):
        if not IsWindowVisible(hwnd):
            return True
        length = 512
        buf = ctypes.create_unicode_buffer(length)
        GetWindowText(hwnd, buf, length)
        title = buf.value
        if title:
            windows.append((hwnd, title))
        return True

    EnumWindows(WNDENUMPROC(_enum), 0)
    return windows


def find_vrchat_window():
    for hwnd, title in list_windows():
        if 'vrchat' in title.lower():
            return hwnd, title
    return None, None


def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ok = GetWindowRect(hwnd, ctypes.byref(rect))
    if not ok:
        raise OSError('GetWindowRect failed')
    return rect.left, rect.top, rect.right, rect.bottom


def capture_window(bbox, out_path):
    img = ImageGrab.grab(bbox=bbox)
    img.save(out_path)


def main():
    hwnd, title = find_vrchat_window()
    if not hwnd:
        print('No VRChat window found. Make sure VRChat is running and visible.')
        print('Top windows:')
        for h, t in list_windows()[:30]:
            print(f'{h}: {t}')
        return

    print(f'Found VRChat window: {title} (hwnd={hwnd})')
    left, top, right, bottom = get_window_rect(hwnd)
    print(f'Window rect: {left},{top},{right},{bottom}')

    out_dir = os.path.join(os.getcwd(), 'captures')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'vrchat_{timestamp}.png')

    capture_window((left, top, right, bottom), out_path)
    print('Saved screenshot to', out_path)

    try:
        os.startfile(out_path)
        print('Opened screenshot with default image viewer.')
    except Exception:
        print('Could not open image automatically. Please open:', out_path)


if __name__ == '__main__':
    main()
