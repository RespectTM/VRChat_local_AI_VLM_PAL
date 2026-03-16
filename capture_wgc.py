"""
Windows Graphics Capture (WGC) screen capture.

Captures a named window's own rendered pixels via the Windows 10+ WGC API,
which works even when the window is completely occluded or behind other windows.
"""

import threading
from windows_capture import WindowsCapture, Frame, InternalCaptureControl


def capture_window_wgc(window_name: str, out_path: str, timeout: float = 10.0):
    """
    Capture the first frame of `window_name` and save it to `out_path`.
    Uses Windows Graphics Capture so the window can be behind other windows.
    Raises RuntimeError on failure.
    """
    done = threading.Event()
    errors = []

    capture = WindowsCapture(
        cursor_capture=False,
        draw_border=False,
        window_name=window_name,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        try:
            frame.save_as_image(out_path)
        except Exception as e:
            errors.append(e)
        finally:
            capture_control.stop()
            done.set()

    @capture.event
    def on_closed():
        done.set()

    capture.start()

    if not done.wait(timeout=timeout):
        raise RuntimeError(
            f'WGC capture timed out after {timeout}s — is "{window_name}" running?'
        )

    if errors:
        raise errors[0]
