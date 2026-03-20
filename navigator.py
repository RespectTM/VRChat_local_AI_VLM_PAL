"""Autonomous navigation behaviour for PAL.

Three behaviours, in priority order:
  1. APPROACH  — a player/avatar is visible in the scene but no nametag is readable yet
                 → walk toward them to close the distance and reveal their name.
  2. INTEREST  — an interesting object/sign/structure is visible, no players around
                 → walk toward it to read it or see it better.
  3. WANDER    — fully idle (no players, nothing interesting) for a while
                 → take small random exploratory steps.

The navigator does NOT fire when:
  • Active chatbox messages or nametags are present (conversation in progress).
  • The model itself just issued a [MOVE: ...] tag (it handled it).
  • We have reached the max consecutive approach attempts (target may be unreachable).
"""

import re
import random
import time
from typing import Optional

import explorer as exp

# ---------------------------------------------------------------------------
# Scene analysis patterns
# ---------------------------------------------------------------------------

# Signals that a humanoid avatar is visible (but may not be named yet)
_AVATAR_RE = re.compile(
    r'\b(avatar|person|someone|player|character|figure|user|'
    r'they|she|he|her|him|costume|outfit|dressed as|wearing)\b',
    re.IGNORECASE,
)

# Signals something spatially interesting to walk towards
_INTEREST_RE = re.compile(
    r'\b(sign|notice|noticeboard|board|poster|banner|text|writing|message|'
    r'building|structure|arch|gate|door|portal|stairs|stage|platform|'
    r'exhibit|display|screen|monitor|crystal|tree|flower|fountain|statue)\b',
    re.IGNORECASE,
)

# Modifiers that suggest the subject is far away or barely visible
_FAR_RE = re.compile(
    r'\b(far|distant|background|across|beyond|ahead|'
    r'in the distance|further|barely visible|silhouette)\b',
    re.IGNORECASE,
)

# Modifiers that suggest the subject is already close / readable
_CLOSE_RE = re.compile(
    r'\b(close|nearby|next to|beside|in front|standing near|right in front|'
    r'right next|approaching|face to face)\b',
    re.IGNORECASE,
)


def _lateral_hint(scene: str) -> str:
    """Return 'left', 'right', or 'forward' based on scene position clues."""
    s = scene.lower()
    # Count explicit positional mentions
    l = s.count(' left') + s.count('to the left') + s.count('on the left')
    r = s.count(' right') + s.count('to the right') + s.count('on the right')
    if l > r + 1:
        return 'left'
    if r > l + 1:
        return 'right'
    return 'forward'


# ---------------------------------------------------------------------------
# Navigator state machine
# ---------------------------------------------------------------------------

class Navigator:
    """Decides autonomous movement every think cycle."""

    # Seconds before idle exploration kicks in (no players, nothing new)
    WANDER_IDLE = 5.0
    # Minimum gap between approach-generated moves (seconds)
    APPROACH_COOLDOWN  = 3.0
    WANDER_COOLDOWN    = 4.0
    # After this many unanswered approach steps, give up (player probably unreachable)
    MAX_APPROACH_STEPS = 12
    # Idle time with no players before explorer kicks in; long enough for PAL to fully
    # engage with a person before wandering off
    WANDER_IDLE        = 20.0

    def __init__(self):
        self._last_move_t: float   = 0.0
        self._last_wander_t: float = 0.0
        self._last_active_t: float = time.time()  # last time a player was present
        self._approach_steps: int  = 0            # consecutive approach steps taken
        self._state: str           = 'idle'       # 'idle' | 'approaching' | 'wandering'
        self._route_queue: list    = []            # pre-planned cruise moves
        self.explorer: exp.Explorer = exp.Explorer()  # occupancy grid + frontier nav

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_player_present(self) -> None:
        """Call whenever a nametag or chatbox message from another player is seen."""
        self._last_active_t  = time.time()
        self._approach_steps = 0
        self._state          = 'idle'
        self._route_queue.clear()
        # Pause frontier exploration while talking (explorer cooldown resets itself)

    def reset_route(self) -> None:
        """Discard any pre-planned cruise route. Call when the think model issues a move."""
        self._route_queue.clear()

    def between_scenes_move(
        self,
        scene: str,
        texts: list,
        osc_enabled: bool,
    ) -> Optional[tuple[str, float]]:
        """Return the next pre-planned cruise move, or None.

        Called once per scene observation during scene accumulation. Enables
        continuous walking between think cycles when the map is ready and no
        players are present. The caller must invoke
        explorer.apply_cruise_move(direction, duration) after executing the move.
        """
        if not osc_enabled:
            return None
        has_tags    = any(s == 'nametag'  for s, _ in texts)
        has_chatbox = any(s == 'chatbox'  for s, _ in texts)
        if has_tags or has_chatbox:
            self._route_queue.clear()
            self.notify_player_present()
            return None
        # Also stop cruise when an avatar or person is visible — let the think cycle
        # handle the approach so PAL walks toward them and reads their nametag properly
        if bool(_AVATAR_RE.search(scene)):
            self._route_queue.clear()
            return None
        now = time.time()
        if now - self._last_move_t < self.APPROACH_COOLDOWN:
            return None
        if now - self._last_active_t < self.WANDER_IDLE:
            return None
        if not self.explorer.map_ready:
            return None
        if not self._route_queue:
            route = self.explorer.plan_route()
            if not route:
                return None
            self._route_queue.extend(route)
        direction, duration = self._route_queue.pop(0)
        self._state       = 'wandering'
        self._last_move_t = now
        return (direction, duration)

    def decide(
        self,
        scene: str,
        texts: list,
        osc_enabled: bool,
        model_moved: bool = False,
    ) -> Optional[tuple[str, float]]:
        """Return ``(direction, seconds)`` or ``None``.

        Args:
            scene:        Combined scene description from vision model.
            texts:        ``[(source, text), ...]`` extracted from scene.
            osc_enabled:  False when running ``--no-osc``.
            model_moved:  True when the think model already emitted a [MOVE:] this cycle.
        """
        if not osc_enabled or model_moved:
            return None

        now         = time.time()
        has_tags    = any(s == 'nametag' for s, _ in texts)
        has_chatbox = any(s == 'chatbox'  for s, _ in texts)

        # Active conversation → stay put, reset approach counter
        if has_tags or has_chatbox:
            self.notify_player_present()
            return None

        # Enforce per-move cooldown
        if now - self._last_move_t < self.APPROACH_COOLDOWN:
            return None

        # ------------------------------------------------------------------
        # Priority 1: approach a visible but unnamed avatar
        # ------------------------------------------------------------------
        avatar_visible  = bool(_AVATAR_RE.search(scene))
        already_close   = bool(_CLOSE_RE.search(scene))

        if avatar_visible and not already_close and self._approach_steps < self.MAX_APPROACH_STEPS:
            self._route_queue.clear()   # abort any cruise — person takes priority
            direction = _lateral_hint(scene)
            duration  = 1.0
            self._state          = 'approaching'
            self._approach_steps += 1
            self._last_move_t    = now
            self._last_active_t  = now  # treat avatar sighting as "player present" for idle timer
            return (direction, duration)

        # Already close or gave up → reset approach counter but keep active timer warm
        if avatar_visible or has_tags:
            self._approach_steps = 0
            self._last_active_t  = now
            self._state = 'idle'
        elif self._approach_steps >= self.MAX_APPROACH_STEPS:
            self._approach_steps = 0
            self._state = 'idle'

        # ------------------------------------------------------------------
        # Priority 2: approach an interesting object
        # ------------------------------------------------------------------
        has_interest = bool(_INTEREST_RE.search(scene))
        if has_interest and not already_close:
            direction = _lateral_hint(scene)
            self._state       = 'approaching'
            self._last_move_t = now
            return (direction, 1.0)

        # ------------------------------------------------------------------
        # Priority 3: frontier exploration when idle
        # ------------------------------------------------------------------
        idle_secs = now - self._last_active_t
        if idle_secs > self.WANDER_IDLE:
            frontier_move = self.explorer.next_move()
            if frontier_move:
                self._last_wander_t = now
                self._last_move_t   = now
                self._state = 'wandering'
                return frontier_move

        return None

    # ------------------------------------------------------------------
    # Readable state for dashboard / logging
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        now  = time.time()
        idle = int(now - self._last_active_t)
        if self._state == 'approaching':
            return (f'🎯 Approaching ({self._approach_steps}/{self.MAX_APPROACH_STEPS} steps) | '
                    f'{self.explorer.status}')
        if self._state == 'wandering':
            return f'🔍 Exploring (idle {idle}s) | {self.explorer.status}'
        return f'💤 Idle ({idle}s) | {self.explorer.status}'
