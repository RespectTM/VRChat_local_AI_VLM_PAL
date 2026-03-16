"""
Non-blocking TTS using pyttsx3 (Windows SAPI voices).
Speech runs in a dedicated thread so the main loop is never stalled.
"""
import queue
import threading
from typing import Optional

_q: queue.Queue = queue.Queue()
_thread: Optional[threading.Thread] = None
_rate: int = 175
_volume: float = 1.0


def _worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        while True:
            item = _q.get()
            if item is None:          # poison pill — shut down
                break
            text, rate, vol = item
            engine.setProperty('rate', rate)
            engine.setProperty('volume', vol)
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print(f'[TTS] worker error: {e}')


def init(rate: int = 175, volume: float = 1.0) -> bool:
    """
    Start the TTS worker thread. Call once at startup.
    Returns True if pyttsx3 is available, False otherwise.
    """
    global _thread, _rate, _volume
    _rate, _volume = rate, volume
    try:
        import pyttsx3  # noqa: F401 — presence check only
        _thread = threading.Thread(target=_worker, daemon=True)
        _thread.start()
        return True
    except ImportError:
        print('[TTS] pyttsx3 not installed — TTS disabled.')
        return False


def speak(text: str) -> None:
    """Queue text for speech. Returns immediately (non-blocking)."""
    if _thread and _thread.is_alive():
        _q.put((text.strip(), _rate, _volume))
