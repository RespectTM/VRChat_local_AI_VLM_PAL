import ctypes
from ctypes import wintypes
from PIL import ImageGrab
import os
from datetime import datetime
import ctypes.util
import sys

try:
    import psutil
except Exception:
    psutil = None

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
    # Prefer matching by process executable name (vrchat.exe).
    # This avoids matching other windows that include the word "VRChat" in their title (e.g. editor tabs).
    for hwnd, title in list_windows():
        try:
            # Get PID for the window
            pid = wintypes.DWORD()
            USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pid_val = pid.value
        except Exception:
            pid_val = None

        if pid_val and psutil:
            try:
                p = psutil.Process(pid_val)
                pname = p.name().lower()
                if 'vrchat' in pname:  # usually 'vrchat.exe'
                    return hwnd, title
            except Exception:
                pass

        # Fallback: strict title matching to avoid accidental matches
        # Only accept titles that are exactly 'vrchat' (case-insensitive) or start with 'vrchat -' or end with ' - vrchat'.
        t = title.lower().strip()
        if t == 'vrchat' or t.startswith('vrchat -') or t.endswith(' - vrchat'):
            return hwnd, title
    return None, None


def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ok = GetWindowRect(hwnd, ctypes.byref(rect))
    if not ok:
        raise OSError('GetWindowRect failed')
    return rect.left, rect.top, rect.right, rect.bottom


def capture_window(bbox, out_path):
    # Deprecated: screen grab may include overlapping windows.
    # Keep this as a fallback, but prefer a PrintWindow-based capture below.
    img = None
    try:
        hwnd = None
        # If bbox was provided as a tuple of ints, we can't map it back to hwnd reliably here.
        # The caller should use the hwnd-based capture (see capture_window_hwnd).
        img = ImageGrab.grab(bbox=bbox)
        img.save(out_path)
        return
    except Exception:
        pass


def capture_window_hwnd(hwnd, out_path):
    # Capture the window content using PrintWindow into a bitmap and save via PIL.
    gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)

    # Required functions
    PrintWindow = USER32.PrintWindow
    PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    PrintWindow.restype = wintypes.BOOL

    GetWindowDC = USER32.GetWindowDC
    GetWindowDC.argtypes = [wintypes.HWND]
    GetWindowDC.restype = wintypes.HDC

    ReleaseDC = USER32.ReleaseDC
    ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    ReleaseDC.restype = ctypes.c_int

    GetClientRect = USER32.GetClientRect
    GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    GetClientRect.restype = wintypes.BOOL

    CreateCompatibleDC = gdi32.CreateCompatibleDC
    CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    CreateCompatibleDC.restype = ctypes.c_void_p

    CreateCompatibleBitmap = gdi32.CreateCompatibleBitmap
    CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    CreateCompatibleBitmap.restype = ctypes.c_void_p

    SelectObject = gdi32.SelectObject
    SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    SelectObject.restype = ctypes.c_void_p

    GetDIBits = gdi32.GetDIBits
    GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    GetDIBits.restype = ctypes.c_int

    BitBlt = gdi32.BitBlt
    BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    BitBlt.restype = ctypes.c_bool

    DeleteObject = gdi32.DeleteObject
    DeleteObject.argtypes = [ctypes.c_void_p]
    DeleteObject.restype = ctypes.c_bool

    DeleteDC = gdi32.DeleteDC
    DeleteDC.argtypes = [ctypes.c_void_p]
    DeleteDC.restype = ctypes.c_bool

    SRCCOPY = 0x00CC0020

    # Get client size
    rect = wintypes.RECT()
    if not GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError('GetClientRect failed')
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width == 0 or height == 0:
        raise ValueError('Window has zero size')

    hwindc = GetWindowDC(hwnd)
    memdc = CreateCompatibleDC(hwindc)
    hbmp = CreateCompatibleBitmap(hwindc, width, height)
    old = SelectObject(memdc, hbmp)

    # Try to render full window content even if occluded (Windows 8+)
    PW_RENDERFULLCONTENT = 0x00000002
    ok = PrintWindow(hwnd, memdc, PW_RENDERFULLCONTENT)
    if not ok:
        # Fallback: try without flags
        ok = PrintWindow(hwnd, memdc, 0)
    if not ok:
        # Final fallback: copy from window DC (may include overlapped areas if window is occluded)
        BitBlt(memdc, 0, 0, width, height, hwindc, rect.left, rect.top, SRCCOPY)

    # Prepare BITMAPINFOHEADER
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', wintypes.DWORD),
            ('biWidth', wintypes.LONG),
            ('biHeight', wintypes.LONG),
            ('biPlanes', wintypes.WORD),
            ('biBitCount', wintypes.WORD),
            ('biCompression', wintypes.DWORD),
            ('biSizeImage', wintypes.DWORD),
            ('biXPelsPerMeter', wintypes.LONG),
            ('biYPelsPerMeter', wintypes.LONG),
            ('biClrUsed', wintypes.DWORD),
            ('biClrImportant', wintypes.DWORD),
        ]

    class RGBQUAD(ctypes.Structure):
        _fields_ = [('rgbBlue', ctypes.c_ubyte), ('rgbGreen', ctypes.c_ubyte), ('rgbRed', ctypes.c_ubyte), ('rgbReserved', ctypes.c_ubyte)]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', RGBQUAD * 1)]

    bmi = BITMAPINFO()
    ctypes.memset(ctypes.byref(bmi), 0, ctypes.sizeof(bmi))
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 24
    bmi.bmiHeader.biCompression = 0  # BI_RGB

    # Buffer size: 3 bytes per pixel, padded to 4-byte rows
    row_bytes = ((width * 3 + 3) // 4) * 4
    buf_size = row_bytes * height
    buf = ctypes.create_string_buffer(buf_size)

    res = GetDIBits(memdc, hbmp, 0, height, buf, ctypes.byref(bmi), 0)
    if res == 0:
        # Cleanup
        SelectObject(memdc, old)
        DeleteObject(hbmp)
        DeleteDC(memdc)
        ReleaseDC(hwnd, hwindc)
        raise OSError('GetDIBits failed')

    # Build PIL Image from raw BGR data (row_bytes includes GDI 4-byte row padding)
    from PIL import Image
    img = Image.frombuffer('RGB', (width, height), buf, 'raw', 'BGR', row_bytes, 1)
    img.save(out_path)

    # Cleanup
    SelectObject(memdc, old)
    DeleteObject(hbmp)
    DeleteDC(memdc)
    ReleaseDC(hwnd, hwindc)


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

    print('Please open:', out_path)


if __name__ == '__main__':
    main()
