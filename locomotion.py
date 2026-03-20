"""VRChat OSC locomotion — sends movement input pulses to VRChat.

VRChat OSC movement addresses (float 1.0 = full input, 0.0 = release):
  /input/MoveForward    — walk forward
  /input/MoveBackward   — walk backward
  /input/MoveLeft       — strafe left
  /input/MoveRight      — strafe right
  /input/LookLeft       — turn/yaw left
  /input/LookRight      — turn/yaw right
  /input/Jump           — jump (hold for height)
  /input/Run            — sprint modifier

Usage::
    locomotion.move(osc, 'forward', 2.0)            # blocking: moves 2 s then stops
    locomotion.move_async(osc, 'left', 1.0)         # non-blocking thread
    locomotion.extract_move_tags('[MOVE: forward 2]') # -> ([('forward', 2.0)], '')
    locomotion.parse_move_hint('walk forward for 3s') # -> ('forward', 3.0)
"""

import re
import threading
import time

# ---------------------------------------------------------------------------
# OSC address map
# ---------------------------------------------------------------------------

_ADDR: dict[str, str] = {
    'forward':    '/input/MoveForward',
    'backward':   '/input/MoveBackward',
    'back':       '/input/MoveBackward',
    'left':       '/input/MoveLeft',
    'right':      '/input/MoveRight',
    'turn_left':  '/input/LookLeft',
    'turn_right': '/input/LookRight',
    'jump':       '/input/Jump',
    'run':        '/input/Run',
}

_ALIASES: dict[str, str] = {
    'w':           'forward',
    'a':           'left',
    's':           'back',
    'd':           'right',
    'strafe_left': 'left',
    'strafe_right': 'right',
    'backwards':   'backward',
    'turn_left':   'turn_left',
    'turn_right':  'turn_right',
}


def _canonical(direction: str) -> str:
    """Normalise a direction string to a key in _ADDR."""
    d = direction.lower().strip().replace(' ', '_')
    return _ALIASES.get(d, d)


# ---------------------------------------------------------------------------
# Core movement
# ---------------------------------------------------------------------------

def move(osc, direction: str, duration: float = 1.0) -> None:
    """Send movement input for *duration* seconds then stop.  Blocks the caller."""
    if not osc:
        return
    key  = _canonical(direction)
    addr = _ADDR.get(key)
    if not addr:
        return
    duration = max(0.1, min(float(duration), 10.0))
    try:
        osc.send_message(addr, 1)   # VRChat expects int 1 (button press)
        time.sleep(duration)
        osc.send_message(addr, 0)   # int 0 (button release)
    except Exception as e:
        print(f'  [locomotion] OSC error {addr}: {e}', flush=True)


def move_async(osc, direction: str, duration: float = 1.0) -> threading.Thread:
    """Non-blocking movement — returns the daemon thread executing the move."""
    t = threading.Thread(target=move, args=(osc, direction, duration), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# [MOVE: direction duration] tag parser — embedded in think() output
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(
    r'\[MOVE:\s*([a-z_]+(?:\s+[a-z_]+)?)\s+(\d+(?:\.\d+)?)\s*\]',
    re.IGNORECASE,
)


def extract_move_tags(text: str) -> tuple[list[tuple[str, float]], str]:
    """Remove all [MOVE: direction duration] tags from *text*.

    Returns ``(moves_list, clean_text)`` where *moves_list* is a list of
    ``(direction, seconds)`` tuples in the order they appeared.

    AI-issued move caps (enforced here, regardless of what the model outputs):
      - turns (turn_left / turn_right): max 1.0 s
      - walks (forward / backward / left / right): max 1.0 s
    """
    moves: list[tuple[str, float]] = []
    _TURN_DIRS = {'turn_left', 'turn_right'}

    def _replace(m: re.Match) -> str:
        direction = _canonical(m.group(1).strip())
        raw_dur   = float(m.group(2))
        max_dur   = 1.0  # both turns and walks capped at 1 s for AI decisions
        duration  = max(0.1, min(raw_dur, max_dur))
        moves.append((direction, duration))
        return ''

    clean = _TAG_RE.sub(_replace, text).strip()
    return moves, clean


# ---------------------------------------------------------------------------
# Natural-language hint parser
# ---------------------------------------------------------------------------

_HINT_RE = re.compile(
    r'(?:move|walk|go|step|strafe|turn|rotate|spin)?\s*'
    r'(forward|backward|back(?:wards?)?|left|right|turn\s+left|turn\s+right)'
    r'(?:\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:onds?)?)?)?)?',
    re.IGNORECASE,
)


def parse_move_hint(hint: str) -> tuple[str, float] | None:
    """Parse a natural-language movement instruction.

    Examples::

        'move forward 2 seconds'  → ('forward', 2.0)
        'walk left'               → ('left', 1.0)
        'go back for 3s'          → ('backward', 3.0)

    Returns ``(direction, seconds)`` or ``None`` if no movement found.
    """
    m = _HINT_RE.search(hint.lower().strip())
    if not m:
        return None
    direction = _canonical(m.group(1).strip())
    duration  = float(m.group(2)) if m.group(2) else 1.0
    return direction, max(0.2, min(duration, 10.0))


def execute_hint(osc, hint: str) -> bool:
    """Parse *hint* as a movement command and execute it asynchronously.

    Returns ``True`` if *hint* was a movement command (and was executed).
    """
    parsed = parse_move_hint(hint)
    if not parsed:
        return False
    direction, duration = parsed
    move_async(osc, direction, duration)
    return True
