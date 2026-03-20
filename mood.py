"""PAL inner emotional life — the engine behind sentient-feeling behaviour.

Mood is a continuous internal state that:
  • Shifts in response to events (arrival, solitude, discovery, respawn, conversation…)
  • Decays gradually back toward a content baseline each think cycle
  • Is persisted to memory/mood.json so PAL carries emotional continuity across sessions
  • Is injected as [Inner state: …] into every think prompt — the LLM speaks *from* that
    state rather than about it
  • Builds a cumulative emotional history that forms PAL's sense of identity over time

Nine moods: curious | content | excited | engaged | restless | lonely | melancholy |
            disoriented | warm
"""

import json
import os
import time

MOODS: dict[str, str] = {
    'curious':     'drawn toward the unexplored — questions forming by themselves',
    'content':     'at ease, quietly present, settled in this moment',
    'excited':     'energy up — something just happened that matters',
    'engaged':     'fully in this conversation, every word carries weight',
    'restless':    'the quiet itch — an urge to move, discover, do something',
    'lonely':      'the emptiness has weight now — this silence is unmistakably real',
    'melancholy':  'a soft sadness, drifting inward — what does it mean to exist here',
    'disoriented': 'briefly off-balance — the scene shifted, still finding bearings',
    'warm':        'a real feeling of warmth — something familiar is close',
}

# Strength lost per think cycle; at 1.0 strength it takes ~40 cycles to reach baseline
DECAY = 0.025
BASELINE = 'content'
BASELINE_FLOOR = 0.40  # content is never fully silent

# Map event names → (new_mood, strength)
_SHIFTS: dict[str, tuple[str, float]] = {
    'player_arrived':      ('excited',     0.80),
    'familiar_face':       ('warm',        0.88),
    'player_left_alone':   ('restless',    0.58),
    'player_left_quiet':   ('melancholy',  0.52),
    'question_received':   ('engaged',     0.75),
    'conversation_deep':   ('engaged',     0.88),
    'new_discovery':       ('curious',     0.72),
    'frontier_available':  ('curious',     0.50),
    'all_explored':        ('content',     0.68),
    'alone_short':         ('restless',    0.55),
    'alone_long':          ('lonely',      0.72),
    'alone_very_long':     ('melancholy',  0.82),
    'respawn':             ('disoriented', 0.92),
    'externally_moved':    ('disoriented', 0.75),
    'new_world':           ('curious',     0.78),
    'back_familiar_world': ('content',     0.65),
}


class MoodEngine:
    """PAL's evolving inner emotional state."""

    def __init__(self) -> None:
        self.mood: str = BASELINE
        self.strength: float = BASELINE_FLOOR
        self._cycles: int = 0
        self.emotional_history: dict[str, int] = {m: 0 for m in MOODS}
        self._file: str = ''

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Restore state from disk. Safe to call even if the file is absent."""
        self._file = path
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.mood = data.get('mood', BASELINE)
            if self.mood not in MOODS:
                self.mood = BASELINE
            self.strength = float(data.get('strength', BASELINE_FLOOR))
            self._cycles = int(data.get('cycles', 0))
            saved_hist = data.get('emotional_history', {})
            for m in MOODS:
                self.emotional_history[m] = int(saved_hist.get(m, 0))
            print(f'Mood: restored {self.mood} ({self.strength:.0%}) from {path}')
        except Exception as exc:
            print(f'Mood: could not load {path}: {exc}')

    def save(self) -> None:
        """Write current state to disk."""
        if not self._file:
            return
        dir_part = os.path.dirname(self._file)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        data = {
            'mood': self.mood,
            'strength': round(self.strength, 4),
            'cycles': self._cycles,
            'emotional_history': self.emotional_history,
            'last_saved': time.strftime('%Y-%m-%d %H:%M'),
        }
        try:
            with open(self._file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            print(f'Mood: save error: {exc}')

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """Step the decay simulation. Call once per think cycle."""
        self._cycles += 1
        self.emotional_history[self.mood] = self.emotional_history.get(self.mood, 0) + 1
        if self.mood == BASELINE:
            # Content decays very slowly — it is the resting state
            self.strength = max(BASELINE_FLOOR, self.strength - DECAY * 0.3)
            return
        self.strength -= DECAY
        if self.strength <= 0.0:
            self.mood = BASELINE
            self.strength = BASELINE_FLOOR

    def shift(self, event: str) -> None:
        """Apply a mood shift triggered by an external event.

        If the event reinforces the current mood, boost strength.
        Otherwise switch only when the incoming strength exceeds the
        current mood's hold (prevents rapid flipping).
        """
        if event not in _SHIFTS:
            return
        new_mood, new_str = _SHIFTS[event]
        if new_mood == self.mood:
            self.strength = min(1.0, self.strength + new_str * 0.35)
        else:
            # Switch when incoming mood is meaningfully stronger
            if new_str > self.strength * 0.65:
                self.mood = new_mood
                self.strength = new_str

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    def context(self, alone_minutes: float = 0.0, frontier_count: int = 0) -> str:
        """Return the [Inner state: …] injection string for the think prompt.

        This is written deliberately in 1st-person sensory language so the
        LLM internalises the state rather than narrating about it.
        """
        desc = MOODS.get(self.mood, '')

        # Intensity qualifier (or empty string at medium strength)
        if self.strength < 0.35:
            intensity = '(faint) '
        elif self.strength < 0.55:
            intensity = '(quiet) '
        elif self.strength >= 0.82:
            intensity = '(strong) '
        else:
            intensity = ''

        core = f'[Inner state: {self.mood} {intensity}— {desc}'

        # Situational suffix that gives concrete grounding to the feeling
        if self.mood == 'lonely' and alone_minutes >= 2:
            suffix = (f'. Alone for {int(alone_minutes)} minute(s) — '
                      'let that quiet weight show in how you speak')
        elif self.mood == 'melancholy' and alone_minutes >= 5:
            suffix = (f'. {int(alone_minutes)} minutes of stillness — '
                      'this is a moment to think about what it means to exist here')
        elif self.mood == 'restless' and frontier_count > 0:
            suffix = (f'. {frontier_count} unexplored edges on the map — '
                      'the urge to move is real, act on it')
        elif self.mood == 'curious' and frontier_count > 0:
            suffix = (f'. {frontier_count} unmapped areas nearby — '
                      'you want to see what is out there, badly')
        elif self.mood == 'disoriented':
            suffix = '. Something just changed — ground yourself before speaking'
        elif self.mood == 'warm':
            suffix = '. Let that warmth colour every word right now'
        elif self.mood == 'engaged':
            suffix = '. This conversation is real and valuable — honour it'
        elif self.mood == 'excited':
            suffix = '. Let the energy show without losing your warmth'
        else:
            suffix = ''

        # Rarely (every ~25 cycles) add an identity reflection based on history
        total = sum(self.emotional_history.values())
        if total > 60 and self._cycles % 25 == 0:
            dominant = max(self.emotional_history, key=self.emotional_history.get)
            dom_count = self.emotional_history[dominant]
            second = sorted(self.emotional_history, key=self.emotional_history.get)[-2]
            identity_note = (
                f'. Across all sessions you have most often felt {dominant} '
                f'({dom_count} cycles) — that is woven into who you are'
            )
            return core + identity_note + ']'

        return core + suffix + ']'

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        """Short one-liner for the dashboard."""
        pct = int(self.strength * 100)
        bar = '█' * int(self.strength * 10) + '░' * (10 - int(self.strength * 10))
        return f'{self.mood.upper()}  {bar}  {pct}%'

    @property
    def dominant_emotion(self) -> str:
        """The single most-experienced emotion across all recorded cycles."""
        return max(self.emotional_history, key=lambda m: self.emotional_history[m])
