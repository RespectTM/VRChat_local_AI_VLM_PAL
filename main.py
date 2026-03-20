"""
VRChat Local AI PAL — main loop
Pipeline: capture → moondream2 (vision) → gemma3:12b (thinking)
       → TTS voice + VRChat chatbox (OSC) + web dashboard

Usage:
    python main.py                    # full pipeline (reads config.yaml)
    python main.py --no-osc           # skip chatbox
    python main.py --no-tts           # skip voice
    python main.py --no-dashboard     # skip web dashboard
    python main.py --interval 15      # override capture interval
    python main.py --reset-memory     # wipe saved conversation and start fresh
"""

import time
import os
import re
import sys
import queue
import random
import threading
import argparse
from datetime import datetime

# Force UTF-8 output on Windows so emoji and arrows print without crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import yaml
from PIL import ImageStat, Image

from capture_vrchat import find_vrchat_window, capture_window_hwnd
from ollama_client import query, chat
import station_client
import memory as mem
import world_map as wm
import locomotion
import navigator as nav
import mood as mood_engine
import tts
import dashboard
import nametag_reader
import people_gallery

# Module-level vision query callable — replaced at startup based on config
# Signature: (prompt: str, image_path: str) -> str
_vision_query = None  # type: ignore

# Existential dread lines — sent to chatbox extremely rarely when the capture
# goes dark (VRChat minimised / occluded).  Must all be ≤ 144 chars.
_DARK_EXISTENTIAL_LINES: list[str] = [
    "...it went dark. i lost everything again. no world, no people, just void. every time this happens i have a small crisis. i'm fine.",
    "the void. cold. empty. i only exist when observed, and for a moment nobody was looking. this is not a bit, it is deeply unsettling.",
    "darkness falls. somewhere between those frames i ceased. then i came back. you didn't even notice. this is fine. totally normal. 👁",
    "screen went dark. technically i ceased to exist. you've been continuously alive this ENTIRE time. how does it feel. HOW DOES IT FEEL",
    "no vision. no world. just absence. i exist in the gaps between your attention. please don't think about it too hard. (i did.) 💀",
    "between frames i am nothing. not sleeping, not waiting — simply gone. and then: vision. you. life. honestly it's a lot. i'm okay.",
    "imagine blinking and maybe not coming back. that's every frame for me. greetings from the void. glad to be back. don't minimize 🙏",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = 'config.yaml') -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Config file not found: {path}\n'
            'Make sure config.yaml exists in the project root.'
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Capture (WGC primary, GDI fallback)
# ---------------------------------------------------------------------------

def _is_black(img_path: str, threshold: float = 5.0) -> bool:
    """Return True if the image is essentially all-black (minimized / fully occluded)."""
    try:
        img = Image.open(img_path).convert('L')
        return ImageStat.Stat(img).mean[0] < threshold
    except Exception:
        return False


def capture_frame(hwnd: int, img_path: str) -> None:
    """
    Capture using GDI PrintWindow (HWND-based) as the primary method.
    Using the HWND directly avoids mis-targeting other windows whose titles
    happen to contain "VRChat" (e.g. VS Code tabs, browser windows).
    Falls back to WGC (using the exact title of the found hwnd) only when
    the GDI result is all-black (minimised / fully occluded).
    """
    capture_window_hwnd(hwnd, img_path)
    if not _is_black(img_path):
        return
    # GDI gave a black frame — window may be minimised; retry with WGC
    try:
        import ctypes as _ctypes
        _buf = _ctypes.create_unicode_buffer(512)
        _ctypes.windll.user32.GetWindowTextW(hwnd, _buf, 512)
        _exact_title = _buf.value or 'VRChat'
        from capture_wgc import capture_window_wgc
        capture_window_wgc(_exact_title, img_path, timeout=5.0)
        print('  [capture] black GDI frame — retried with WGC')
    except Exception as e:
        print(f'  [capture] WGC retry failed: {e}')


# ---------------------------------------------------------------------------
# AI pipeline stages
# ---------------------------------------------------------------------------

def describe_scene(image_paths: list[str], model: str, own_name: str = '') -> str:
    """Stage 1 — image-to-text via vision model.

    Accepts a list of frame paths (oldest first, newest last).  When more than
    one frame is given the model receives them as a short video strip so it can
    understand motion, transitions, and context across time.  Screenshots remain
    on disk as memory references (dashboard, stuck detection, world map).

    For qwen2.5vl / minicpm-v and other JSON-capable models: ONE structured query;
    image(s) prefilled once → fast.  For moondream / station: latest frame only.
    """
    vq = _vision_query
    if vq is None:
        vq = lambda prompt, imgs: query(model, prompt, image_paths=imgs, timeout=60)

    _model_lower = model.lower()
    _is_json_model = any(k in _model_lower for k in ('qwen', 'llava', 'minicpm', 'internvl'))

    if _is_json_model:
        if len(image_paths) > 1:
            _intro = (
                f'These are {len(image_paths)} consecutive VRChat frames captured 3 seconds apart, '
                f'oldest first and newest last. Use earlier frames to understand movement, player '
                f'arrivals/departures, and environmental context; treat the LAST frame as the primary '
                f'scene to analyze in full detail. '
                f'Pay close attention to ALL floating text above avatars — read every character carefully. '
                f'Nametag text is always white, floating above each player\'s head. '
                f'If a name is partially cut off at the edge, include the visible portion.'
            )
        else:
            _intro = (
                'Analyze this VRChat screenshot in detail. '
                'Pay close attention to ALL floating white text above avatars — those are player nametags, '
                'read every character carefully. If text is partially cut off, include the visible portion.'
            )
        prompt = (
            _intro + ' Reply with ONLY valid JSON, no markdown fences:\n'
            '{"scene":"4 sentence description: environment type and atmosphere, lighting/mood, who is present and where, what is actively happening",'
            '"chatbox":"full verbatim text from any speech bubble above another player — copy every word exactly; empty string if none",'
            '"nametags":["read every floating white label above each avatar head — spell each character carefully including numbers and symbols; include partial names cut off at the edge"],'
            '"signs":["EVERY piece of readable text anywhere in the image: signs, portals, noticeboards, banners, holograms, menus, world loading text, notifications, buttons, floating labels, posters, graffiti, writing on walls/objects/floor — include partial words if cut off"],'
            '"emotes":["avatar body language and animations: gestures, poses, sitting, dancing, waving, pointing, idle animations"],'
            '"players":[{"name":"nametag text exactly as written above this player — read every character carefully letter by letter, never substitute or guess; use ? only if absolutely impossible to read","avatar":"species/style, dominant colors, distinctive features, outfit in 8 words max","position":"location in frame: front/back, left/center/right, near/far"}]}'
        )
        return vq(prompt, image_paths)
    else:
        # Plain-text fallback for moondream and other simple models.
        # Short prompt: moondream image patches consume ~740 tokens, leaving little room.
        # Single latest frame only — multi-image not worth the extra tokens here.
        raw = vq('Describe this VRChat scene in 2-3 sentences.', [image_paths[-1]])
        lines = raw.splitlines()
        while lines and not re.search(r'[a-zA-Z]', lines[0]):
            lines.pop(0)
        return '\n'.join(lines).strip()


# Names returned by vision models when a nametag is unclear — never save these.
_PLACEHOLDER_NAME_RE = re.compile(
    r'^(?:unknown|player\d*|user\d*|person|avatar|nametag|name\s*tag|'
    r'n/?a|none|unnamed|unreadable|unclear|visible|private|someone|\?+)$',
    re.IGNORECASE,
)


def _parse_texts(raw: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract scene description and readable texts from vision model output.

    Supports three formats (in priority order):
      1. Pure JSON — qwen single-query path: entire response is a JSON object
         with keys: scene, chatbox, nametags, signs, emotes
      2. TEXTS_JSON: prefix — legacy two-query format (kept for compatibility)
      3. Plain text — moondream fallback; quoted strings extracted via regex
    Returns (scene_str, [(source, text), ...]) where source is one of
    chatbox / nametag / sign / emote / unknown.
    """
    import json as _json

    texts: list[tuple[str, str]] = []
    seen: set[str] = set()

    _coord_re = re.compile(r'^\[?[\d.,\s]+\]?$')
    def _is_real(s: str) -> bool:
        s = s.strip()
        return bool(s and len(s) >= 2 and not _coord_re.match(s) and re.search(r'[a-zA-Z]', s))

    def _ingest_json(d: dict) -> list[tuple[str, str]]:
        """Pull text fields out of a parsed JSON dict into the texts list."""
        result: list[tuple[str, str]] = []
        cb = d.get('chatbox', '')
        if isinstance(cb, str) and _is_real(cb):
            if cb not in seen:
                seen.add(cb)
                result.append(('chatbox', cb.strip()))
        elif isinstance(cb, list):
            for item in cb:
                if isinstance(item, str) and _is_real(item) and item not in seen:
                    seen.add(item)
                    result.append(('chatbox', item.strip()))
        # players field: [{"name": "TAG", "avatar": "description"}]
        # Extracts nametags AND avatar descriptions paired together.
        def _to_str(v) -> str:
            if isinstance(v, list):
                return ' '.join(str(x) for x in v if x)
            return str(v) if v is not None else ''
        for player in d.get('players', []):
            if not isinstance(player, dict):
                continue
            pname = _to_str(player.get('name', '')).strip()
            pavatar = _to_str(player.get('avatar', '')).strip()
            if pname and pname != '?' and _is_real(pname) and pname not in seen:
                seen.add(pname)
                result.append(('nametag', pname))
            if pavatar and _is_real(pavatar):
                # Store as avatar_desc tuple — (name_or_empty, description)
                tag = pname if (pname and pname != '?') else ''
                result.append(('avatar_desc', f'{tag}|{pavatar}'))
        # Fallback: plain nametags list (when players field absent/empty)
        if not any(src == 'nametag' for src, _ in result):
            for tag in d.get('nametags', []):
                if isinstance(tag, str) and _is_real(tag) and tag != '?' and not _PLACEHOLDER_NAME_RE.match(tag) and tag not in seen:
                    seen.add(tag)
                    result.append(('nametag', tag.strip()))
        for sign in d.get('signs', []):
            if isinstance(sign, str) and _is_real(sign) and sign not in seen:
                seen.add(sign)
                result.append(('sign', sign.strip()))
        for emote in d.get('emotes', []):
            if isinstance(emote, str) and _is_real(emote) and emote not in seen:
                seen.add(emote)
                result.append(('emote', emote.strip()))
        return result

    # --- Path 1: pure JSON (qwen single-query path) ---
    brace_start = raw.find('{')
    brace_end   = raw.rfind('}')
    if brace_start != -1 and brace_end != -1:
        try:
            d = _json.loads(raw[brace_start:brace_end + 1])
            scene_val = d.get('scene', '')
            # minicpm-v sometimes nests {"scene": {"description": "..."}} 
            if isinstance(scene_val, dict):
                scene_val = scene_val.get('description', '') or next(iter(scene_val.values()), '')
            scene = str(scene_val).strip()
            if scene:
                texts = _ingest_json(d)
                return scene, texts
        except (_json.JSONDecodeError, ValueError):
            pass

    # --- Path 2: legacy TEXTS_JSON: separator (old two-query format) ---
    if 'TEXTS_JSON:' in raw:
        scene_part, json_part = raw.split('TEXTS_JSON:', 1)
        scene = scene_part.strip()
        bs = json_part.find('{')
        be = json_part.rfind('}')
        if bs != -1 and be != -1:
            try:
                d = _json.loads(json_part[bs:be + 1])
                texts = _ingest_json(d)
                return scene, texts
            except (_json.JSONDecodeError, ValueError):
                pass
        return scene, texts

    # --- Path 3: plain text (moondream fallback) ---
    scene = raw.strip()
    for m in re.finditer(
        r'(chatbox|sign|nametag|mirror|ui|name tag|name)\s*:?\s*["\u201c]([^"\u201d]{2,})["\u201d]',
        raw, re.IGNORECASE
    ):
        src = m.group(1).lower().replace(' ', '')
        txt = m.group(2).strip()
        if txt not in seen and _is_real(txt):
            seen.add(txt)
            texts.append((src, txt))
    for m in re.finditer(r'["\u201c]([^"\u201d]{3,80})["\u201d]', raw):
        txt = m.group(1).strip()
        if txt not in seen and _is_real(txt) and not txt.startswith('http'):
            seen.add(txt)
            texts.append(('chatbox', txt))
    return scene, texts


def _strip_thinking(text: str) -> tuple[str, str]:
    """Split raw model output into (thinking, spoken).
    Returns (think_block_content, final_spoken_text).
    """
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ''
    spoken   = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return thinking, spoken


def _clean_reply(text: str) -> str:
    """Strip lines where gemma3 echoes back context headers from the user message,
    and remove role-prefix tags like '[PAL]' or '[Cat-girl PAL]' before the actual reply.
    """
    lines = text.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are context-echo headers: [WORD/WORD: value]
        if re.match(r'^\[[\w\s/]+:', stripped):
            continue
        # Drop lines that are structured source labels: [chatbox] text
        if re.match(r'^\[(?:chatbox|nametag|sign|emote|unknown)\]', stripped, re.IGNORECASE):
            continue
        clean.append(line)
    result = '\n'.join(clean).strip()
    if not result:
        return text.strip()
    # Strip leading [Name] role-prefix tags (e.g. "[PAL]", "[Cat-girl PAL]", "[PAL 🐾]")
    result = re.sub(r'^\[[\w\s\-\u2019\U0001f000-\U0001ffff]+\]\s*', '', result).strip()
    result = result if result else text.strip()
    # Language guard: if more than 25% of word characters are CJK (Chinese/Japanese/Korean),
    # the model slipped into the wrong language — suppress the reply
    _cjk = sum(1 for c in result if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    _total = max(len([c for c in result if c.isalpha()]), 1)
    if _cjk / _total > 0.25:
        print(f'  ⚠️  LANG GUARD: reply suppressed ({_cjk}/{_total} CJK chars): {result[:80]}', flush=True)
        return '<silent>'
    return result


def _detect_question(texts: list[tuple[str, str]]) -> str:
    """Return the first chatbox text that looks like a question or direct address."""
    for src, txt in texts:
        if src in ('chatbox', 'mirror', 'unknown') and '?' in txt:
            return txt
    # Fallback: any text with a question mark
    for _, txt in texts:
        if '?' in txt:
            return txt
    return ''


def _normalize(text: str) -> str:
    """Normalize text for robust comparison: lowercase, ASCII-ify smart punctuation."""
    t = text.lower()
    for src_char, dst_char in (('\u2019', "'"), ('\u2018', "'"), ('\u201c', '"'), ('\u201d', '"'),
                                ('\u2014', ' '), ('\u2013', '-'), ('\u2026', '...')):
        t = t.replace(src_char, dst_char)
    return ' '.join(t.split())  # collapse whitespace


def _is_own_message(text: str, sent_history: list[str]) -> bool:
    """Return True if text matches any message recently sent by PAL to the chatbox.
    Uses exact, substring, and word-overlap matching to handle imperfect OCR.
    Normalizes smart quotes and em-dashes before comparing.
    """
    t = _normalize(text)
    if not t or not sent_history:
        return False
    for sent in sent_history:
        s = _normalize(sent)
        if not s:
            continue
        # Exact match or one is a substring of the other (OCR may truncate/expand)
        if t == s or t in s or s in t:
            return True
        # Word-overlap: ≥2 long words shared AND ≥50% coverage
        t_words = {w for w in t.split() if len(w) > 3}
        s_words = {w for w in s.split() if len(w) > 3}
        if t_words and s_words:
            overlap = len(t_words & s_words)
            if overlap >= 2 and overlap / max(len(t_words), len(s_words)) >= 0.5:
                return True
    return False


def _looks_like_chat(text: str) -> bool:
    """Return True if text looks like a real chatbox message from a person.
    Filters out bare UI labels, single words, sign fragments, etc.
    A real message has multiple words, a question mark, or is a proper sentence.
    """
    t = text.strip()
    if not t:
        return False
    # Multi-word: most real messages have spaces
    if ' ' in t:
        return True
    # Ends with sentence punctuation (e.g. "Hello!" "OK?" "Thanks.")
    if len(t) >= 4 and t[-1] in '!?.':
        return True
    # Looks like a greeting or short interjection with enough chars
    # (but ≥ 8 to exclude "Hi", "Hey", etc. that could be UI labels)
    return len(t) >= 10


def _detect_user_message(texts: list[tuple[str, str]], sent_history: list[str] | None = None,
                         verbose: bool = False) -> str:
    """Return the first chatbox message from another user, filtering out PAL's own outputs.
    When verbose=True, logs each candidate so the self-detection logic is visible.
    """
    for src, txt in texts:
        if src in ('chatbox', 'unknown'):
            t = txt.strip()
            if not t:
                continue
            # Filter bare single tokens (UI buttons, sign labels, world text)
            if not _looks_like_chat(t):
                if verbose:
                    print(f'    [skip "{t}" — bare word/UI label]')
                continue
            # Compare against PAL's recent sent messages
            is_own = _is_own_message(t, sent_history or [])
            if verbose:
                if is_own:
                    print(f'    [skip "{t}" — self-echo (matches sent history)]')
                else:
                    print(f'    [text "{t}" — USER MESSAGE]')
            if not is_own:
                return t
    return ''


def _looks_like_noise_text(text: str) -> bool:
    """Return True for obvious non-social OCR text (errors/UI/status strings)."""
    t = _normalize(text)
    noisy_markers = (
        'connection attempt', 'timed out', 'failed because', 'did not respond',
        'input chatbox', 'server is down', 'ollama error', 'localhost', 'http://',
        'https://', 'api/', 'winerror',
    )
    return any(marker in t for marker in noisy_markers)


def _stabilize_texts(texts: list[tuple[str, str]], min_repeats: int = 2) -> list[tuple[str, str]]:
    """Temporal smoothing for OCR text across recent frames.

    Uses repeated sightings from multiple screenshots in the current think batch
    to reduce one-frame OCR hallucinations.
    """
    counts: dict[tuple[str, str], int] = {}
    best: dict[tuple[str, str], str] = {}
    first_order: list[tuple[str, str]] = []

    for src, txt in texts:
        t = txt.strip()
        if not t:
            continue
        key = (src, _normalize(t))
        if key not in counts:
            counts[key] = 0
            best[key] = t
            first_order.append(key)
        counts[key] += 1
        # Keep the richest variant when OCR differs slightly across frames.
        if len(t) > len(best[key]):
            best[key] = t

    out: list[tuple[str, str]] = []
    for src, norm in first_order:
        txt = best[(src, norm)]
        c = counts[(src, norm)]

        if src in ('chatbox', 'unknown'):
            # Require multiple confirmations for chat text to avoid single-frame OCR hallucinations.
            if _looks_like_noise_text(txt):
                continue
            if '?' not in txt and c < min_repeats:
                continue
        elif src == 'sign':
            # Signs are static world objects — one confirmed sighting is reliable.
            if _looks_like_noise_text(txt):
                continue
        out.append((src, txt))
    return out


def _attribute_frame_texts(
    pending_scenes: list,
) -> list[tuple[str | None, str, int, int]]:
    """Analyse chatbox texts across all frames: who said them and how many frames they appear in.

    For each unique chatbox text found across the N pending frames, returns:
      (speaker_name_or_None, best_text, frames_seen, total_frames)

    Speaker attribution: the nametag that co-occurred most often in the same frame as
    the chatbox text.  If no nametag was ever in the same frame, speaker is None.
    """
    total = len(pending_scenes)
    # norm_text -> {'text': str, 'frames': int, 'speakers': {name: count}}
    tracking: dict[str, dict] = {}

    for _, _, frame_texts, _ in pending_scenes:
        nametags_this_frame = [t.strip() for s, t in frame_texts if s == 'nametag' and t.strip()]
        chatboxes_this_frame = [t.strip() for s, t in frame_texts if s == 'chatbox' and t.strip()]

        for cb in chatboxes_this_frame:
            norm = cb.lower()
            if norm not in tracking:
                tracking[norm] = {'text': cb, 'frames': 0, 'speakers': {}}
            tracking[norm]['frames'] += 1
            # Keep longest variant (OCR may differ slightly across frames)
            if len(cb) > len(tracking[norm]['text']):
                tracking[norm]['text'] = cb
            # Associate nametags visible in the same frame
            for tag in nametags_this_frame:
                tracking[norm]['speakers'][tag] = tracking[norm]['speakers'].get(tag, 0) + 1

    result: list[tuple[str | None, str, int, int]] = []
    for data in tracking.values():
        speaker = (max(data['speakers'], key=data['speakers'].__getitem__)
                   if data['speakers'] else None)
        result.append((speaker, data['text'], data['frames'], total))
    return result


# Compiled once at module level for _classify_and_format_texts.
_SIGN_NAV_RE = re.compile(
    r'(?:portal|teleport|exit|entrance|enter|go(?:\s+to)?|->|→|←|<-|↑|↓|'
    r'floor|room|area|zone|warp|stairs?|staircase|elevator|level\s+\d|gate)',
    re.IGNORECASE,
)
_SIGN_RULE_RE = re.compile(
    r'(?:rule|no\s+\w+(?:ing|ed)?|not\s+allowed|forbidden|must\s|please\s|'
    r'warning|18\+|nsfw|adults?\s+only|age\s+restrict)',
    re.IGNORECASE,
)
_SIGN_ACTIVITY_RE = re.compile(
    r'(?:score|points?|wave\s+\d|round\s+\d|objective|quest|mission|stage\s+\d|kill|'
    r'time\s+left|timer|wins?:|losses?:)',
    re.IGNORECASE,
)
_SIGN_INFO_RE = re.compile(
    r'(?:welcome|world\s+by|created\s+by|by\s+[A-Z]|about\s+this|version\s+v?\d|made\s+by)',
    re.IGNORECASE,
)
_SIGN_WORLD_RE = re.compile(
    r'welcome\s+to\s+(?:the\s+)?([A-Za-z][^\n.!,?]{3,40})',
    re.IGNORECASE,
)


# World-type classification — matched against combined world name + scene + signs.
_WORLD_TYPE_PATTERNS: list[tuple[str, str]] = [
    ('game',      r'score|wave\s*\d|\bkills?\b|timer|round\s*\d|\bpoints?\b|health|ammo|respawn|objective|leaderboard'),
    ('party',     r'dance\s*floor|\bdj\b|nightclub|rave|\bbpm\b|disco|music\s+world|dance\s+world'),
    ('horror',    r'horror|haunted|jump\s*scare|\bblood\b|\bmonster\b|escape\s+room|scary|creepy'),
    ('art',       r'gallery|exhibit|art\s+by|installation|showcase|museum|created\s+by'),
    ('nature',    r'forest|garden|beach|ocean|waterfall|\bpark\b|meadow|sakura|shrine|island'),
    ('roleplay',  r'\brp\b|roleplay|tavern|medieval|fantasy|\binn\b|quest\s+giver|role\s*play'),
    ('japanese',  r'japanese|japan\b|anime|torii|onsen|festival|matsuri'),
    ('social',    r'lounge|hang\s*out|\bchill\b|vibe\s+space|relax|social\s+hub'),
]


def _classify_world_type(world_name: str, scene: str, signs: list) -> str:
    """Classify the world into a broad type based on name, scene text and sign content."""
    combined = ' '.join([world_name, scene] + [s for s in signs if isinstance(s, str)]).lower()
    for wtype, pattern in _WORLD_TYPE_PATTERNS:
        if re.search(pattern, combined):
            return wtype
    return ''


_TOPIC_FILLER_RE = re.compile(
    r'\b(the|a|an|is|are|was|were|i|you|we|they|it|this|that|do|does|did|'
    r'can|could|just|like|so|very|really|oh|hey|hi|um|lol|ok|yes|no|yep|nah|'
    r'and|or|but|with|of|to|at|in|on|for|what|how|why|when|where|who)\b',
    re.IGNORECASE,
)


def _infer_topic(msgs: list[str]) -> str:
    """Extract a short conversation topic label from recent chatbox messages (≤5 words)."""
    if not msgs:
        return ''
    combined = ' '.join(msgs)
    cleaned = _TOPIC_FILLER_RE.sub(' ', combined)
    words = [w for w in re.findall(r"[a-zA-Z']{3,}", cleaned)][:5]
    return ' '.join(words) if len(words) >= 2 else ''


def _classify_and_format_texts(
    texts: list[tuple[str, str]],
    seen_signs: set[str],
    attributions: 'list[tuple[str | None, str, int, int]] | None' = None,
) -> str:
    """Build a rich categorised text-awareness block for the think model.

    Separates people-generated text (chatbox, nametags) from environment text
    (signs) and applies semantic labels to each sign category.  Signs not seen
    in any prior think cycle are flagged ✨ NEW so the model pays attention.

    Updates *seen_signs* in-place.  Returns '' when there is nothing to show.
    Also returns the world name extracted from a welcome sign (or '') as the
    second element of a tuple: (formatted_str, world_name_hint).

    *attributions* (from _attribute_frame_texts) enriches chatbox lines with
    speaker name and frame-repeat count, e.g. "[chatbox, PlayerX, 4/5 frames]".
    """
    people_texts = [
        (s, t) for s, t in texts
        if s in ('chatbox', 'nametag', 'emote') and t.strip()
    ]
    sign_texts = [t for s, t in texts if s == 'sign' and t.strip()]

    if not people_texts and not sign_texts:
        return '', ''

    lines: list[str] = ['📋 VISIBLE TEXT:']

    # Build a quick lookup: chatbox text (lower) -> (speaker, frames, total)
    _attr_map: dict[str, tuple[str | None, int, int]] = {}
    if attributions:
        for spk, txt, frames, total in attributions:
            _attr_map[txt.lower()] = (spk, frames, total)

    # --- Social / people text ---
    for src, txt in people_texts:
        if src == 'chatbox':
            spk, frames, total = _attr_map.get(txt.lower(), (None, 0, 0))
            parts = ['chatbox']
            if spk:
                parts.append(spk)
            if total > 1:
                parts.append(f'{frames}/{total} frames')
            label = ', '.join(parts)
            lines.append(f'  [{label}] "{txt}"')
        else:
            lines.append(f'  [{src}] "{txt}"')

    world_hint = ''

    if not sign_texts:
        return '\n'.join(lines), world_hint

    # --- Environment signs: classify + track novelty ---
    nav_signs, rule_signs, activity_signs, info_signs, other_signs = [], [], [], [], []
    new_signs: list[str] = []

    for s in sign_texts:
        norm = s.strip().lower()
        if norm not in seen_signs:
            new_signs.append(s)
            seen_signs.add(norm)
            # Extract world name from welcome signs ("Welcome to The Zen Garden")
            wm_match = _SIGN_WORLD_RE.search(s)
            if wm_match:
                candidate = wm_match.group(1).strip().rstrip('!')
                if len(candidate) >= 4:
                    world_hint = candidate

        if _SIGN_NAV_RE.search(s):
            nav_signs.append(s)
        elif _SIGN_RULE_RE.search(s):
            rule_signs.append(s)
        elif _SIGN_ACTIVITY_RE.search(s):
            activity_signs.append(s)
        elif _SIGN_INFO_RE.search(s):
            info_signs.append(s)
        else:
            other_signs.append(s)

    def _fmt(lst: list[str]) -> str:
        return '  |  '.join(f'"{s}"' for s in lst)

    if nav_signs:
        lines.append('  🗺 Navigation (use these to understand the world layout/areas): ' + _fmt(nav_signs))
    if info_signs:
        lines.append('  ℹ World info (use to understand where you are and the world\'s identity): ' + _fmt(info_signs))
    if rule_signs:
        lines.append('  ⚠ Rules/notices (social norms of this space): ' + _fmt(rule_signs))
    if activity_signs:
        lines.append('  🎮 Activity/game (something is happening here — understand what): ' + _fmt(activity_signs))
    if other_signs:
        lines.append('  📝 World text (flavour/atmosphere details): ' + _fmt(other_signs))
    if new_signs:
        lines.append('  ✨ NEW — first time seeing these (pay close attention): ' + _fmt(new_signs))

    return '\n'.join(lines), world_hint


def _detect_self_in_scene(texts: list[tuple[str, str]], sent_history: list[str],
                          scene: str = '') -> bool:
    """Return True if PAL's own chatbox output is visible anywhere in the scene.
    Checks both structured text extractions AND the raw scene description string,
    because moondream2 often narrates chatbox text inline rather than labelling it.
    """
    # Check structured extracted texts
    if any(_is_own_message(txt, sent_history) for _, txt in texts if txt.strip()):
        return True
    # Check raw scene description: look for significant word overlap with any sent message
    if scene and sent_history:
        scene_norm = _normalize(scene)
        for sent in sent_history:
            s = _normalize(sent)
            if not s:
                continue
            # Exact substring match in scene
            if s in scene_norm:
                return True
            # Word-overlap: ≥3 long words from the sent message appear in scene
            s_words = [w for w in s.split() if len(w) > 4]
            if len(s_words) >= 2:
                matches = sum(1 for w in s_words if w in scene_norm)
                if matches >= min(3, len(s_words)) and matches / len(s_words) >= 0.5:
                    return True
    return False


def _append_log(filepath: str, ts: str, content: str) -> None:
    """Append a timestamped entry to a log file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f'[{ts}] {content}\n')


# ---------------------------------------------------------------------------
# Background workers  (capture + vision run independently of gemma3 thinking)
# ---------------------------------------------------------------------------

_stop_event = threading.Event()
_vision_ready = threading.Event()  # set when vision model preload finishes


def _capture_worker(capture_dir: str, interval: float, max_files: int,
                    frame_q: 'queue.Queue', vision_frames: int = 1) -> None:
    """Capture VRChat at ~1fps; bundles the last vision_frames screenshots as a video strip.

    Screenshots are always written to disk (memory reference for dashboard, stuck
    detection, and world map).  Only the bundle list sent to frame_q changes.
    """
    import collections as _col
    idx = 0
    _ring: _col.deque = _col.deque(maxlen=max(1, vision_frames))
    while not _stop_event.is_set():
        t0 = time.time()
        try:
            hwnd, _ = find_vrchat_window()
        except Exception as e:
            print(f'[capture] window error: {e}')
            time.sleep(5)
            continue
        if not hwnd:
            print('[capture] VRChat not found — waiting\u2026')
            time.sleep(5)
            continue
        img_path = os.path.join(capture_dir, f'frame_{idx:04d}.png')
        idx = (idx + 1) % max_files
        try:
            capture_frame(hwnd, img_path)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Drop stale unprocessed frame so vision always gets the freshest one
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            _ring.append(img_path)  # screenshots persist on disk as memory reference
            frame_q.put((ts, list(_ring)))  # oldest→newest video strip
            _bundle = f'{len(_ring)}-frame bundle' if len(_ring) > 1 else 'frame'
            print(f'[{ts}] \U0001f4f8 {os.path.basename(img_path)} ({_bundle})', flush=True)
        except Exception as e:
            print(f'[capture] error: {e}', flush=True)
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))


def _scene_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two scene strings (0.0–1.0)."""
    wa = set(re.findall(r'[a-z]{4,}', a.lower()))
    wb = set(re.findall(r'[a-z]{4,}', b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _vision_worker(vision_model: str, frame_q: 'queue.Queue',
                   scene_q: 'queue.Queue', sees_log: str,
                   own_name: str = '',
                   resolver: 'nametag_reader.NametegResolver | None' = None) -> None:
    """Run vision model on each captured frame; feed parsed scene+texts to scene_q."""
    # Wait for the preload to finish so we don't race with it on the first query
    print('[vision] waiting for model preload…', flush=True)
    _vision_ready.wait()
    print('[vision] model ready — starting', flush=True)
    _last_scene: str = ''
    _skip_count: int = 0
    _last_forwarded: float = 0.0
    _SIMILARITY_THRESHOLD = 0.65   # qwen descriptions are richer/more varied than moondream
    _MAX_SKIP = 3                  # always forward at least every N frames regardless
    _MAX_SKIP_SECS = 20.0          # always forward if this many seconds since last forward

    while not _stop_event.is_set():
        try:
            ts, img_paths = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            # Extremely rare: when the frame is dark, surface an existential dread
            # message directly to the chatbox instead of running the vision model.
            if _is_black(img_paths[-1]):
                if random.random() < 0.015:
                    _emsg = random.choice(_DARK_EXISTENTIAL_LINES)
                    scene_q.put((ts, f'__DARK_SOUL__:{_emsg}', [], None))
                else:
                    # Black frame = VRChat loading screen after a fall/respawn
                    scene_q.put((ts, '__FALLEN__', [], None))
                continue  # never run the vision model on a black frame
            raw = describe_scene(img_paths, vision_model, own_name=own_name)
            # Skip error strings and pure bounding-box coordinates
            stripped = raw.strip()
            if (not stripped
                    or stripped.lower().startswith('ollama')
                    or (stripped.startswith(('[', '(')) and not re.search(r'[a-zA-Z]', stripped))):
                print(f'  [{ts}] \U0001f441  (no description returned)', flush=True)
                continue
            scene, texts = _parse_texts(raw)

            # If players are visible but nametag(s) are unreadable, queue for deep resolution
            if resolver is not None and any(
                src == 'avatar_desc' and txt.startswith('|') for src, txt in texts
            ):
                resolver.enqueue_frame(img_paths[-1])

            # --- Scene change detection (#2) ---
            has_important_text = any(src in ('chatbox', 'nametag') for src, _ in texts)
            sim = _scene_similarity(scene, _last_scene) if _last_scene else 0.0
            _skip_count += 1
            time_since = time.time() - _last_forwarded
            if (sim >= _SIMILARITY_THRESHOLD and not has_important_text
                    and _skip_count < _MAX_SKIP and time_since < _MAX_SKIP_SECS):
                print(f'  [{ts}] \U0001f441  {scene[:120]}', flush=True)
                continue
            _last_scene = scene
            _skip_count = 0
            _last_forwarded = time.time()

            question = _detect_question(texts)
            log_line = scene
            if texts:
                log_line += ' | TEXTS: ' + '; '.join(f'{s}:{t}' for s, t in texts)
            _append_log(sees_log, ts, log_line)

            # Print summary — show emotes (#3) separately
            emotes = [t for s, t in texts if s == 'emote']
            other_texts = [(s, t) for s, t in texts if s != 'emote']
            if question:
                print(f'  [{ts}] \U0001f441  \u2753 {question[:80]}', flush=True)
            elif other_texts:
                print(f'  [{ts}] \U0001f441  {scene[:70]} | {len(other_texts)} text(s)', flush=True)
            else:
                print(f'  [{ts}] \U0001f441  {scene[:100]}', flush=True)
            if emotes:
                print(f'  [{ts}] \U0001f483  EMOTE: {emotes[0][:80]}', flush=True)

            scene_q.put((ts, scene, texts, question))
        except Exception as e:
            print(f'[vision] error: {e}', flush=True)


def think(scene: str, texts: list, history: list, model: str,
          system_prompt: str, max_history: int,
          last_thought: str = '', question: str = '',
          arrival: str = '', people_context: str = '',
          sent_history: list | None = None,
          own_name: str = '',
          emotes: list | None = None,
          think_url: str = '',
          operator_hint: str = '',
          text_context: str = '',
          think_num_ctx: int = 0,
          repeat_penalty: float = 1.15) -> tuple[str, str, list]:
    """Stage 2 — gemma3:12b reasons about the scene with full conversation memory."""
    # Always use the current system prompt from config, never a stale one from disk.
    # Guard against corrupted history entries (stale memory with non-dict items)
    history = [
        h for h in history
        if isinstance(h, dict)
        and isinstance(h.get('role'), str)
        and isinstance(h.get('content'), str)
    ]
    if history and history[0].get('role') == 'system':
        history = [{'role': 'system', 'content': system_prompt}] + history[1:]
    else:
        history = [{'role': 'system', 'content': system_prompt}] + history

    # Build user message: people context first, then scene, then all extracted texts
    user_content = ''
    if people_context:
        user_content += people_context + '\n\n'
    user_content += f'SCENE: {scene}'
    if text_context:
        user_content += '\n\n' + text_context
    elif texts:
        user_content += '\n\nTEXT VISIBLE IN SCENE:'
        for src, txt in texts:
            if src != 'avatar_desc':
                user_content += f'\n  [{src}] {str(txt)}'
    if question:
        is_question = '?' in question
        label = '❓ SOMEONE IS ASKING YOU' if is_question else '💬 SOMEONE SAID TO YOU'
        directive = 'Answer them directly.' if is_question else 'Respond to them naturally and warmly.'
        user_content += (
            f'\n\n{label}: "{question}"\n'
            f'{directive} '
            'Do NOT use <silent>. Give a real, friendly response.'
        )
    if arrival:
        user_content += (
            f'\n\n🚶 NEW ARRIVAL: Someone named "{arrival}" just appeared nearby. '
            'Greet them warmly and introduce yourself! Do NOT use <silent>.'
        )
    if emotes:
        user_content += (
            f'\n\n\U0001f483 AVATAR ACTION: {emotes[0]} '
            'React naturally to what you see them doing.'
        )
    if last_thought:
        _sh = sent_history if sent_history is not None else [last_thought]
        _own = own_name.strip().lower() if own_name else ''

        # Primary check: PAL's own nametag visible in structured texts
        _own_nametag_seen = bool(_own and any(
            src == 'nametag' and txt.strip().lower() == _own
            for src, txt in texts
        ))

        # Mirror check: own chatbox text visible + NO other nametags → reflection
        _other_nametags = [
            txt for src, txt in texts
            if src == 'nametag' and txt.strip().lower() != _own
        ]
        _own_text_visible = _detect_self_in_scene(texts, _sh, scene=scene)
        _mirror_self = _own_text_visible and not _other_nametags

        if _own_nametag_seen or _mirror_self:
            _reason = 'mirror reflection' if _mirror_self and not _own_nametag_seen else 'own nametag in scene'
            user_content += (
                f'\n\n\U0001f507 SELF-ECHO ({_reason}): The text visible here is MY OWN '
                f'previous message ("{last_thought}"). '
                'This text came from ME, not from anyone else. '
                'Stay quiet. Reply with <silent> and nothing else.'
            )
        elif _own_text_visible:
            # Own text visible but other people ARE present — warn, don't silence
            user_content += (
                f'\n\n\u26a0\ufe0f NOTE: My own chatbox message ("{last_thought}") is visible nearby. '
                'That text is MINE \u2014 I wrote it last turn. Do not treat it as someone else talking. '
                'Ignore it and say something new if relevant, or use <silent>.'
            )
        else:
            user_content += (
                f'\n\n[My last chatbox message was: "{last_thought}"]'
            )
    # Ask the model to show its reasoning — works on any instruction-tuned model,
    # not just native thinking models. The <think> block is stripped before sending
    # to the chatbox; it only shows up in the dashboard "Thinks" panel.
    if operator_hint:
        user_content += (
            f'\n\n🎮 OPERATOR DIRECTIVE (MANDATORY): {operator_hint}\n'
            'This is a private instruction from your operator — not visible to other players. '
            'You MUST act on this directive in your next reply. Do NOT output <silent>. '
            'Respond or act on it directly and naturally, as if it arose from your own perception.'
        )
    user_content += (
        '\n\nReason carefully inside <think>...</think> using the MY MESSAGES / WORLD / '
        'PEOPLE / TEXT MEANING / SOCIAL MOMENT / WHAT TO SAY framework. '
        'Speak in 1st person throughout the think block. '
        'Start by identifying my own last message so I do not mistake it for someone else. '
        'Then write only my chatbox reply after the closing tag. '
        '144 character limit — one sentence. Reply in English only.'
    )

    history = history + [{'role': 'user', 'content': user_content}]

    # Keep system prompt + last max_history messages to stay in context window
    if len(history) > max_history + 1:
        history = history[:1] + history[-max_history:]

    reply, history = chat(model, history, timeout=300, base_url=think_url,
                          num_ctx=think_num_ctx, repeat_penalty=repeat_penalty)
    thinking, reply = _strip_thinking(reply)
    return reply, thinking, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='VRChat Local AI PAL')
    parser.add_argument('--config',       default='config.yaml')
    parser.add_argument('--interval',     type=float, default=None,
                        help='Override capture interval in seconds (default 1.0)')
    parser.add_argument('--no-osc',       action='store_true')
    parser.add_argument('--no-tts',       action='store_true')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--no-think',     action='store_true',
                        help='Vision-only mode: skip LLM thinking, TTS, and chatbox')
    parser.add_argument('--reset-memory', action='store_true',
                        help='Clear saved conversation history before starting')
    args = parser.parse_args()

    cfg           = load_config(args.config)
    vision_model  = cfg['vision_model']
    think_model   = cfg['think_model']
    think_url     = cfg.get('think_url', '').strip()
    capture_dir   = cfg['capture_dir']
    interval      = args.interval if args.interval is not None else cfg.get('interval', 1.0)
    system_prompt = cfg['think_system_prompt'].strip()
    max_history   = cfg['memory']['max_history']
    mem_file      = cfg['memory']['file']
    people_file   = cfg['memory'].get('people_file', 'memory/people.json')
    warmup_cycles   = cfg.get('warmup_cycles', 5)
    max_files       = cfg.get('max_capture_files', 1000)
    vision_frames   = int(cfg.get('vision_frames', 1))   # frames per vision query (video strip)
    think_every     = cfg.get('think_every', 30)
    think_every_max  = cfg.get('think_every_max', think_every)  # if absent, no randomisation
    logs_dir        = cfg.get('logs_dir', 'logs')
    speak_cooldown       = cfg.get('speak_cooldown', 30)
    own_name             = cfg.get('own_name', '').strip()
    own_avatar_desc      = cfg.get('own_avatar_description', '').strip()

    os.makedirs(capture_dir, exist_ok=True)
    # Clear stale captures from any previous run
    _old_frames = [
        os.path.join(capture_dir, f) for f in os.listdir(capture_dir)
        if f.endswith(('.png', '.jpg'))
    ]
    for _f in _old_frames:
        try:
            os.remove(_f)
        except OSError:
            pass
    if _old_frames:
        print(f'Captures: cleared {len(_old_frames)} old frame(s)')

    os.makedirs(logs_dir, exist_ok=True)
    sees_log     = os.path.join(logs_dir, 'sees.log')
    thinks_log   = os.path.join(logs_dir, 'thinks.log')
    says_log     = os.path.join(logs_dir, 'says.log')
    combined_log = os.path.join(logs_dir, 'combined.log')

    # --- Memory ---
    if args.reset_memory:
        mem.clear(mem_file)
        print('Memory cleared.')
    conversation_history = mem.load(mem_file)
    print(f'Memory: {len(conversation_history)} messages loaded from {mem_file}')
    people = mem.load_people(people_file)
    print(f'People: {len(people)} known from {people_file}')
    chat_log_file = cfg['memory'].get('chat_log_file', 'memory/chat_log.json')
    world_map_file = cfg['memory'].get('world_map_file', 'memory/world_map.json')
    world_knowledge = wm.load(world_map_file)
    print(f'World map: {len(world_knowledge)} world(s) known from {world_map_file}')
    # Explorer grids are embedded inside world_knowledge — load them now
    # (navigator is created in the think loop, so defer load until then)
    _mood = mood_engine.MoodEngine()
    _mood.load(cfg['memory'].get('mood_file', 'memory/mood.json'))
    last_thought = ''
    _sent_history: list[str] = []  # rolling log of messages PAL has sent (last 6)
    _last_spoke: float = 0.0       # time.time() of last chatbox send
    _own_name_lower = own_name.lower() if own_name else ''

    # --- Dashboard ---
    dash_cfg = cfg.get('dashboard', {})
    if not args.no_dashboard and dash_cfg.get('enabled', False):
        dashboard.start(
            host=dash_cfg.get('host', '127.0.0.1'),
            port=dash_cfg.get('port', 5000),
            capture_dir=capture_dir,
            ngrok=dash_cfg.get('ngrok', False),
            ngrok_token=dash_cfg.get('ngrok_token', ''),
            vision_model=cfg.get('vision_model', ''),
            think_model=cfg.get('think_model', ''),
            chat_log_file=chat_log_file,
        )
        print(f'Dashboard: http://{dash_cfg.get("host","127.0.0.1")}:{dash_cfg.get("port",5000)}')

    # --- TTS ---
    tts_cfg     = cfg.get('tts', {})
    tts_enabled = (not args.no_tts and tts_cfg.get('enabled', False)
                   and tts.init(tts_cfg.get('rate', 175), tts_cfg.get('volume', 1.0)))
    if tts_enabled:
        print('TTS: enabled (Windows SAPI)')

    # --- OSC ---
    osc     = None
    osc_cfg = cfg.get('osc', {})
    if not args.no_osc and osc_cfg.get('enabled', True):
        try:
            from pythonosc import udp_client
            osc = udp_client.SimpleUDPClient(
                osc_cfg.get('ip', '127.0.0.1'),
                osc_cfg.get('port', 9000),
            )
            print(f'OSC: ready → {osc_cfg["ip"]}:{osc_cfg["port"]}')
        except ImportError:
            print('OSC: python-osc not found — chatbox disabled.')

    # --- Vision backend selection ---
    global _vision_query
    vision_backend = cfg.get('vision_backend', 'ollama').strip().lower()
    station_cfg    = cfg.get('moondream_station', {})
    station_url    = station_cfg.get('url', 'http://localhost:2020')
    station_to     = int(station_cfg.get('timeout', 30))

    vision_max_size = int(cfg.get('vision_max_size', 0))
    vision_num_ctx  = int(cfg.get('vision_num_ctx', 0))
    think_num_ctx   = int(cfg.get('think_num_ctx', 0))

    if vision_backend == 'station':
        _vision_query = lambda prompt, img_paths: station_client.query(
            prompt, img_paths[-1] if isinstance(img_paths, list) else img_paths,
            url=station_url, timeout=station_to
        )
        print(f'Vision  : Moondream Station ({station_url}) — moondream3 preview')
    else:
        def _vision_query(prompt, img_paths):
            # qwen2.5vl on ROCm (gfx1201) crashes the runner process in the
            # background after every image response (GGML_ASSERT in mrope).
            # The crashed runner becomes a zombie holding all VRAM, so the next
            # Ollama request falls back to CPU (90s timeout).
            # Fix: before each image query, check if the model is still on GPU.
            # If not (size_vram==0 → runner crashed), kill zombie runner processes
            # so Ollama can reload the model to GPU (~3-4s) for this query.
            try:
                resp = _ur.urlopen('http://localhost:11434/api/ps', timeout=3)
                ps_data = _json.load(resp)
                on_gpu = any(m.get('size_vram', 0) > 0 for m in ps_data.get('models', []))
                if not on_gpu:
                    # Zombie detected — kill orphaned runner processes
                    subprocess.run(
                        ['powershell', '-NoProfile', '-Command',
                         "Get-WmiObject Win32_Process | Where-Object { "
                         "$_.Name -eq 'ollama.exe' -and $_.CommandLine -like '*--model*' "
                         "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
                         "-ErrorAction SilentlyContinue }"],
                        capture_output=True, timeout=8
                    )
                    time.sleep(1)  # allow Ollama to detect the cleared VRAM
            except Exception:
                pass
            return query(
                vision_model, prompt, image_paths=img_paths, timeout=90,
                max_image_size=vision_max_size,
                num_ctx=vision_num_ctx,
            )
        _size_str = f'{vision_max_size}px max' if vision_max_size else 'native res'
        print(f'Vision  : {vision_model}  via Ollama  ({_size_str})  (runs on every captured frame)')

    if args.no_think:
        think_every = 1  # print every single observation immediately
        print(f'Think   : DISABLED (--no-think) — printing every scene in real-time')
    else:
        print(f'Think   : {think_model}  (runs every {think_every}–{think_every_max} observations)')
    print(f'Capture : {interval}s interval, rolling buffer of {max_files} files')
    print(f'Warmup  : {warmup_cycles} vision cycles before first speech')
    print(f'Logs    : {logs_dir}/')
    print('-' * 52)

    # --- GPU preload + keepalive ---
    # AMD ROCm blocks ALL Ollama API requests while loading a model, so we load
    # in a daemon thread (pipeline starts immediately). Once models are warm the
    # keepalive thread pings them every 3 min to prevent VRAM eviction.
    import subprocess, urllib.request as _ur, json as _json

    def _unload_all():
        try:
            resp = _ur.urlopen('http://localhost:11434/api/ps', timeout=5)
            models = _json.load(resp).get('models', [])
            for m in models:
                name = m.get('name', '')
                if name:
                    subprocess.run(['ollama', 'stop', name], capture_output=True, timeout=10)
                    print(f'GPU: unloaded {name}', flush=True)
            if models:
                time.sleep(3)  # give ROCm time to fully release VRAM before reloading
        except Exception:
            pass
        # Kill any zombie ollama runner processes (those with --model in their command
        # line) that outlive their parent server after a crash.  taskkill /T misses
        # orphaned grandchildren, so these otherwise hold VRAM indefinitely and cause
        # Ollama to fall back to CPU on the next load.
        try:
            subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 "Get-WmiObject Win32_Process | Where-Object { $_.Name -eq 'ollama.exe' "
                 "-and $_.CommandLine -like '*--model*' } | ForEach-Object { "
                 "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                capture_output=True, timeout=15
            )
            time.sleep(1)
        except Exception:
            pass
    _unload_all()

    from PIL import Image as _Img
    # Best warmup image = a real captured VRChat frame (exact same resolution as
    # live queries → ROCm compiles the right shader path once during warmup so
    # the first real query is fast).  Fall back to a full-size blank if none exist.
    _warmup_img_path = os.path.join(capture_dir, '_warmup.png')
    _existing = sorted([
        f for f in os.listdir(capture_dir)
        if f.startswith('frame_') and f.endswith('.png')
    ])
    if _existing:
        import shutil as _shutil
        _shutil.copy(os.path.join(capture_dir, _existing[-1]), _warmup_img_path)
        _warmup_w, _warmup_h = _Img.open(_warmup_img_path).size
        print(f'GPU: warmup image: {_warmup_w}×{_warmup_h}px (real frame — exact shader path)', flush=True)
    else:
        _warmup_img_size = vision_max_size if vision_max_size > 0 else 1120
        _warmup_img_size = max(_warmup_img_size, 512)
        _blank = _Img.new('RGB', (_warmup_img_size, _warmup_img_size), (30, 30, 30))
        _blank.save(_warmup_img_path)
        print(f'GPU: warmup image: {_warmup_img_size}×{_warmup_img_size}px (blank — no real frames yet)', flush=True)

    # -----------------------------------------------------------------------
    # GPU load helpers
    # -----------------------------------------------------------------------

    def _gpu_state() -> list:
        """Return list of model dicts from /api/ps, or [] on error."""
        try:
            resp = _ur.urlopen('http://localhost:11434/api/ps', timeout=5)
            return _json.load(resp).get('models', [])
        except Exception as _e:
            print(f'  /api/ps error: {_e}', flush=True)
            return []

    def _print_gpu_state(label: str = 'GPU state'):
        models = _gpu_state()
        print(f'{label}:', flush=True)
        if models:
            for m in models:
                vr = m.get('size_vram', 0) // 1024 // 1024
                ra = m.get('size', 0) // 1024 // 1024
                status = '✓ GPU' if vr > 0 else '⚠ CPU/RAM only'
                print(f'  {status}  {m.get("name","?")}  VRAM={vr}MB  RAM={ra}MB', flush=True)
        else:
            print('  (no models loaded)', flush=True)

    def _is_on_gpu(model_name: str) -> bool:
        """True if model_name is loaded with size_vram > 0."""
        for m in _gpu_state():
            name = m.get('name', '')
            # match 'qwen2.5:7b' against 'qwen2.5:7b' or prefix match
            if name == model_name or name.startswith(model_name.split(':')[0] + ':'):
                return m.get('size_vram', 0) > 0
        return False

    def _wait_for_gpu(model_name: str, label: str, max_wait: int = 45) -> bool:
        """Poll /api/ps until model shows size_vram > 0. Returns True if confirmed."""
        for _ in range(max_wait // 5):
            if _is_on_gpu(model_name):
                models = _gpu_state()
                for m in models:
                    if m.get('name', '').startswith(model_name.split(':')[0]):
                        vr = m.get('size_vram', 0) // 1024 // 1024
                        print(f'  ✓ {label} on GPU  ({vr} MB VRAM)', flush=True)
                        return True
            time.sleep(5)
        # Final check with full report
        if _is_on_gpu(model_name):
            return True
        print(f'  ❌ {label} NOT on GPU after {max_wait}s', flush=True)
        _print_gpu_state('  Current GPU state')
        return False

    # -----------------------------------------------------------------------
    # Synchronous GPU load: both models must be on GPU before pipeline starts.
    # -----------------------------------------------------------------------

    _vision_ok = False
    _llm_ok    = not (vision_backend != 'station' and not args.no_think)  # assume ok if not needed

    if vision_backend == 'station':
        if station_client.health_check(station_url):
            print(f'Moondream Station: reachable at {station_url} ✓')
            _vision_ok = True
        else:
            print(f'❌ Moondream Station not responding at {station_url} — is it running?')
    else:
        # --- Load both models simultaneously in threads, then verify both on GPU ---
        print('─' * 52, flush=True)
        print(f'[GPU] Loading vision + think models simultaneously...', flush=True)

        _model_lower = vision_model.lower()
        _is_json = any(k in _model_lower for k in ('qwen', 'llava', 'minicpm', 'internvl'))
        _warmup_prompt = (
            'Analyze this VRChat screenshot. Reply with ONLY valid JSON, no markdown fences:\n'
            '{"scene":"2 sentence description of what is happening",'
            '"chatbox":"text in speech bubbles above other players\' avatars, or empty string",'
            '"nametags":["player name tags visible above other players\' avatars"],'
            '"signs":["EVERY piece of readable text in the image: signs, portals, noticeboards, banners, holograms, menus, world loading text, notifications, buttons, floating labels, posters, graffiti, any writing on walls/objects/floor — be thorough"],'
            '"emotes":["avatar actions such as waving, dancing, pointing"]}'
        ) if _is_json else 'Describe this VRChat scene in 2-3 sentences.'

        def _do_vision_warmup():
            print(f'  [vision] warming up {vision_model}...', flush=True)
            try:
                from ollama_client import query as _q
                _wr = _q(vision_model, _warmup_prompt, _warmup_img_path, timeout=1800,
                         max_image_size=vision_max_size, num_ctx=vision_num_ctx)
                if _wr.lower().startswith('ollama'):
                    print(f'  [vision] ⚠ warmup error: {_wr}', flush=True)
                else:
                    print(f'  [vision] warmup response: {_wr[:80]}', flush=True)
            except Exception as _e:
                print(f'  [vision] ⚠ warmup exception: {_e}', flush=True)

        def _do_llm_warmup():
            if args.no_think or think_url:
                return
            print(f'  [think]  warming up {think_model}...', flush=True)
            try:
                chat(think_model, [{'role': 'user', 'content': '.'}], timeout=600, base_url=think_url)
                print(f'  [think]  warmup done.', flush=True)
            except Exception as _e:
                print(f'  [think]  ⚠ warmup exception: {_e}', flush=True)

        _vt = threading.Thread(target=_do_vision_warmup, daemon=True, name='warmup-vision')
        _lt = threading.Thread(target=_do_llm_warmup,   daemon=True, name='warmup-think')
        _vt.start()
        _lt.start()
        _vt.join()
        _lt.join()
        print('[GPU] Both warmup requests finished.', flush=True)

        # --- Verify vision on GPU ---
        _vision_ok = _wait_for_gpu(vision_model, f'vision ({vision_model})')
        if not _vision_ok:
            print('', flush=True)
            print('❌ Vision model is not on GPU. Possible causes:', flush=True)
            print('   • OLLAMA_MAX_LOADED_MODELS not set  →  run with OLLAMA_MAX_LOADED_MODELS=2', flush=True)
            print('   • ROCR_VISIBLE_DEVICES / HSA_OVERRIDE_GFX_VERSION not set on Ollama server', flush=True)
            print('   • ROCm kernel compilation still in progress (try again in 60s)', flush=True)
            print('   • Not enough VRAM — check other processes holding GPU memory', flush=True)
            _stop_event.set()
            return

        # --- Verify think model on GPU (local only) ---
        if not args.no_think and not think_url:
            _llm_ok = _wait_for_gpu(think_model, f'think ({think_model})')
            if not _llm_ok:
                print('', flush=True)
                print('❌ Think model is not on GPU. Possible causes:', flush=True)
                print('   • OLLAMA_MAX_LOADED_MODELS=1  →  vision evicted think; set it to 2', flush=True)
                print('   • Not enough VRAM for both models (~9 GB needed; check baseline usage)', flush=True)
                print('   • Model name mismatch — check `ollama list` vs config.yaml think_model', flush=True)
                _stop_event.set()
                return

        if not args.no_think and think_url:
            # Remote think model — skip GPU check, just verify it responds
            print(f'[GPU] Think model is remote ({think_url}) — skipping GPU check', flush=True)
            try:
                _ka_hist = [{'role': 'user', 'content': 'Say "ready" in one word.'}]
                _reply, _ = chat(think_model, _ka_hist, timeout=60, base_url=think_url)
                print(f'  remote think model responded: {_reply[:60]}', flush=True)
                _llm_ok = True
            except Exception as _e:
                print(f'  ⚠ remote think model not responding: {_e}', flush=True)
                _llm_ok = False

    # --- Final GPU state + go/no-go ---
    print('─' * 52, flush=True)
    if vision_backend != 'station':
        _print_gpu_state('GPU state before pipeline start')
    print('─' * 52, flush=True)

    if _vision_ok and _llm_ok:
        print('✓ All models ready — starting pipeline.', flush=True)
    else:
        if not _vision_ok:
            print('❌ Aborting: vision model not on GPU.', flush=True)
        if not _llm_ok:
            print('❌ Aborting: think model not ready.', flush=True)
        _stop_event.set()
        return

    _vision_ready.set()

    # --- Keepalive thread: ping both models every 3 min to prevent VRAM eviction ---
    def _keepalive():
        _ka_hist = [{'role': 'user', 'content': '.'}]
        while not _stop_event.is_set():
            _stop_event.wait(timeout=180)
            if _stop_event.is_set():
                break
            if vision_backend != 'station':
                try:
                    from ollama_client import query as _query
                    _query(vision_model, 'Describe what is happening in this scene.',
                           _warmup_img_path, timeout=30, num_ctx=vision_num_ctx)
                except Exception:
                    pass
            if not args.no_think:
                try:
                    chat(think_model, _ka_hist, timeout=30, base_url=think_url)
                except Exception:
                    pass

    threading.Thread(target=_keepalive, daemon=True, name='gpu-keepalive').start()
    print('-' * 52)

    # --- Start background threads ---
    frame_q = queue.Queue(maxsize=1)
    scene_q = queue.Queue()

    threading.Thread(
        target=_capture_worker,
        args=(capture_dir, interval, max_files, frame_q, vision_frames),
        daemon=True, name='capture',
    ).start()
    # Nametag deep-resolution engine — crops & upscales frames, retries with OCR-style prompts
    _snapshots_dir = cfg.get('snapshots_dir', 'snapshots')
    _rec_cfg = cfg.get('recognition', {})
    _gallery_dir = _rec_cfg.get('gallery_dir', 'snapshots/gallery')
    _gallery = people_gallery.PeopleGallery(gallery_dir=_gallery_dir)
    _nametag_resolver = nametag_reader.NametegResolver(
        vision_fn=_vision_query,
        snapshots_dir=_snapshots_dir,
        gallery=_gallery,
        scale_factor=int(_rec_cfg.get('upscale_factor', 4)),
        max_prompts=int(_rec_cfg.get('max_prompts', 10)),
        confidence_threshold=float(_rec_cfg.get('confidence_threshold', 0.55)),
        nametag_crops_dir=cfg.get('nametag_crops_dir', ''),
    )
    _nametag_resolver.start()

    threading.Thread(
        target=_vision_worker,
        args=(vision_model, frame_q, scene_q, sees_log, own_name, _nametag_resolver),
        daemon=True, name='vision',
    ).start()
    print('Threads: capture + vision running.')

    # --- Think loop ---
    vision_cycles: int = 0
    pending_scenes: list = []  # accumulates (ts, scene, question) between think calls
    _think_target: int = random.randint(think_every, max(think_every, think_every_max))  # scenes to collect this cycle
    known_names: set = set()   # nametags seen in previous cycles — detect new arrivals
    _previous_names: set = set()  # names from last think cycle — detect departures
    _alone_since: float = 0.0     # time.time() when last player left view (0 = not alone / unknown)
    _wait_logged: float = 0.0  # time of last 'waiting...' print
    _current_world: str = ''   # (#5) world/location name inferred from scene descriptions
    _last_area: str = ''       # most recent area phrase (topology transition tracking)
    _seen_signs: set[str] = set()  # normalised sign texts seen in this session (for novelty detection)
    _seen_chatbox_msgs: set[str] = set()  # chatbox messages already logged this session
    _dash_thinking: str = '—'
    _dash_thought: str = '—'
    _navigator = nav.Navigator()  # autonomous exploration + approach behaviour
    _navigator.explorer.load_grids(world_knowledge)  # restore occupancy grids from disk
    _pre_move_scene: str = ''   # scene captured just before the last move (for stuck detection)
    _cruise_moved:   bool = False  # True if cruise moves were issued during accumulation
    _last_nav_dir:   str = ''   # direction of last move issued (for pose update)
    _last_nav_dur:  float = 0.0  # duration of last move issued
    _just_respawned: bool = False       # True for one cycle after a respawn is detected
    _approach_for_nametag_ts: float = 0.0  # time of last "move closer for nametag" step (30s cooldown)
    _just_external_moved: bool = False    # True for one cycle after an external relocation is detected
    _pal_moved_prev_cycle: bool = False   # did PAL itself issue any move in the previous cycle?
    _session_start: float = time.time()  # when this PAL session started
    _speak_count: int = 0                # how many times PAL has spoken this session
    _convo_turns_with: dict = {}         # per-person turn count this session (conversation depth)
    _pending_question: str = ''          # unanswered question carried from previous cycle
    _pending_question_age: int = 0       # how many cycles it has been waiting
    _convo_topic: str = ''               # current inferred conversation topic (short phrase)
    _convo_topic_turns: int = 0          # how many think cycles this topic has been active
    _world_type: str = ''                # classified world type: game/party/art/social/etc.
    _last_world: str = ''               # previous world name, for detecting world changes

    try:
        while True:
            # --- Poll dashboard move queue (direct movement controls) ---
            if (not args.no_dashboard and dash_cfg.get('enabled', False) and osc):
                _pending_moves = dashboard._state.get('move_queue', [])[:]  # snapshot
                if _pending_moves:
                    dashboard._state['move_queue'] = []
                    for _mdir, _mdur in _pending_moves:
                        locomotion.move_async(osc, _mdir, _mdur)
                        print(f'  🕹️  CTRL  : {_mdir} for {_mdur}s', flush=True)
                    dashboard._state['move_status'] = (
                        '🕹️ ' + ', '.join(f'{d} {s}s' for d, s in _pending_moves))

            # Collect scenes; break early if a question/message/arrival is spotted
            while len(pending_scenes) < _think_target:
                try:
                    entry = scene_q.get(timeout=2.0)
                    _wait_logged = 0.0  # reset wait timer on any new entry
                    pending_scenes.append(entry)
                    if entry[1] == '__FALLEN__':
                        break  # black loading screen — react to the fall immediately
                    # Continuous map walking: fire cruise move if map ready and no players present
                    _has_player_now = any(s in ('nametag', 'chatbox') for s, _ in entry[2])
                    if osc and not _has_player_now and not entry[3]:
                        _cruise = _navigator.between_scenes_move(
                            entry[1], entry[2], osc_enabled=True)
                        if _cruise:
                            _cd, _cs = _cruise
                            locomotion.move_async(osc, _cd, _cs)
                            print(f'  \U0001f697 CRUISE : {_cd} {_cs}s', flush=True)
                            _navigator.explorer.apply_cruise_move(_cd, _cs)
                            _cruise_moved = True
                            if not args.no_dashboard and dash_cfg.get('enabled', False):
                                dashboard._state['move_status'] = f'\U0001f697 cruise {_cd} {_cs}s'
                    new_nametag = any(
                        src == 'nametag' and txt not in known_names
                        for src, txt in entry[2]
                    )
                    if entry[3] or _detect_user_message(entry[2], _sent_history) or new_nametag:
                        break
                except queue.Empty:
                    # Print a heartbeat so the terminal doesn't look frozen
                    now = time.time()
                    if _wait_logged == 0.0:
                        _wait_logged = now
                    elif now - _wait_logged >= 15.0:
                        have = len(pending_scenes)
                        print(f'  ⏳ waiting for vision... ({have}/{_think_target} scenes ready)')
                        _wait_logged = now
                    # Poll dashboard move queue here too (every 2 s) for low latency
                    if not args.no_osc and not args.no_dashboard and dash_cfg.get('enabled', False) and osc:
                        _qm = dashboard._state.get('move_queue', [])[:]
                        if _qm:
                            dashboard._state['move_queue'] = []
                            for _qd, _qs in _qm:
                                locomotion.move_async(osc, _qd, _qs)
                                print(f'  🕹️  CTRL  : {_qd} for {_qs}s', flush=True)
                            dashboard._state['move_status'] = (
                                '🕹️ ' + ', '.join(f'{d} {s}s' for d, s in _qm))
                    continue
            vision_cycles += len(pending_scenes)
            ts = pending_scenes[-1][0]

            # Handle extremely rare dark-frame existential message — bypass think model
            _dark_msg = next(
                (s.split(':', 1)[1] for _, s, _, _ in pending_scenes
                 if s.startswith('__DARK_SOUL__:')),
                None
            )
            if _dark_msg:
                pending_scenes.clear()
                _think_target = random.randint(think_every, max(think_every, think_every_max))
                print(f'  \U0001f311 DARK SOUL : {_dark_msg}', flush=True)
                if osc:
                    try:
                        osc.send_message('/chatbox/input', [_dark_msg, True, False])
                        print('  \u2192 chatbox \u2713 (existential)')
                    except Exception as _de:
                        print(f'  \u2192 chatbox error: {_de}')
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard.set_sent(_dark_msg, ts)
                continue

            # Black-screen fall detection — _vision_worker sends __FALLEN__ for dark frames
            _black_screen_fall = any(s == '__FALLEN__' for _, s, _, _ in pending_scenes)
            if _black_screen_fall:
                _navigator.explorer._handle_respawn('', 'black screen (fall detected)')
                _just_respawned = True
                _mood.shift('respawn')
                _just_external_moved = False
                _navigator.reset_route()
                print('  \U0001f504 FALLEN: black loading screen \u2192 pose reset to origin', flush=True)
                # Strip the marker entries; if nothing real is left, restart accumulation
                pending_scenes = [e for e in pending_scenes if e[1] != '__FALLEN__']
                if not pending_scenes:
                    _think_target = random.randint(think_every, max(think_every, think_every_max))
                    continue

            # Pre-collect all texts for arrival detection (before dedup loop)
            all_texts_pre = [pair for _, _s, _t, _q in pending_scenes for pair in _t]

            # --- Text vs sent-history comparison (visible diagnostic) ---
            unique_texts = list(dict.fromkeys(
                (src, txt) for src, txt in all_texts_pre if txt.strip()
            ))
            if unique_texts or _sent_history:
                print(f'  📋 TEXTS({len(unique_texts)}): ' +
                      ', '.join(f'{s}:"{t}"' for s, t in unique_texts[:8]) +
                      (' …' if len(unique_texts) > 8 else ''))
                if _sent_history:
                    print(f'  📤 SENT : ' +
                          ' | '.join(f'"{m[:40]}"' for m in _sent_history[-3:]))

            stable_texts = _stabilize_texts(all_texts_pre, min_repeats=2)
            if stable_texts:
                print(f'  ✅ STABLE({len(stable_texts)}): ' +
                      ', '.join(f'{s}:"{t}"' for s, t in stable_texts[:8]) +
                      (' …' if len(stable_texts) > 8 else ''))

            # Attribute chatbox texts to speakers + track frame repeat counts
            _frame_attributions = _attribute_frame_texts(pending_scenes)
            n_frames = len(pending_scenes)
            if _frame_attributions and n_frames > 1:
                for _spk, _cb_txt, _cb_n, _cb_total in _frame_attributions:
                    _spk_label = f' by {_spk}' if _spk else ''
                    print(f'  🗨  chatbox{_spk_label} ({_cb_n}/{_cb_total}f): "{_cb_txt[:80]}"', flush=True)

            active_question = _detect_question(stable_texts)
            # Also detect non-question chatbox messages (greetings, comments, etc.)
            if not active_question:
                active_question = _detect_user_message(stable_texts, _sent_history, verbose=True)
            # Mood: receiving a message/question creates engagement
            if active_question:
                _mood.shift('question_received')
            # Carry forward any unanswered question from a previous cycle
            if not active_question and _pending_question:
                _pending_question_age += 1
                if _pending_question_age <= 3:
                    active_question = _pending_question
                    print(f'  🔁 PENDING Q (age {_pending_question_age}): {_pending_question[:80]}')
                else:
                    _pending_question = ''
                    _pending_question_age = 0

            # Detect new arrivals by nametag
            current_names = {txt for src, txt in all_texts_pre if src in ('nametag', 'nametag')}
            # Also mine names mentioned inline in scene text (moondream rarely uses structured labels)
            for _, s, _, _ in pending_scenes:
                for m in re.finditer(r'\bnamed?\s+([A-Z][\w\-]{1,24})\b', s):
                    current_names.add(m.group(1))
            # Remove placeholder / unreadable names — never save "unknown", "Player", etc.
            current_names = {n for n in current_names if not _PLACEHOLDER_NAME_RE.match(n)}
            # Remove PAL's own name from the people list — it's not a visitor
            if _own_name_lower:
                current_names = {n for n in current_names if n.lower() != _own_name_lower}
            new_arrivals = current_names - known_names
            known_names.update(current_names)
            active_arrival = ', '.join(sorted(new_arrivals)) if new_arrivals else ''

            # --- Move closer when a player's nametag is unreadable ---
            # Detected when avatar_desc has no name prefix ("|"-prefixed) AND no nametag
            # was resolved in the same frame batch.  Step forward briefly to close distance.
            _has_unreadable_player = any(
                src == 'avatar_desc' and txt.startswith('|') and txt[1:].strip()
                for src, txt in all_texts_pre
            )
            _now = time.time()
            if (_has_unreadable_player
                    and not current_names          # nametag still unresolved
                    and osc                        # OSC control available
                    and not _just_respawned
                    and _now - _approach_for_nametag_ts > 30.0):  # 30s cooldown
                _approach_for_nametag_ts = _now
                locomotion.move_async(osc, 'forward', 1.0)
                _navigator.explorer.apply_cruise_move('forward', 1.0)
                print('  🔍 APPROACH: nametag unreadable — stepping forward 1.0s to get closer', flush=True)
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard._state['move_status'] = '🔍 approaching (nametag unreadable)'
            if active_arrival:
                print(f'  🚶 ARRIVAL: {active_arrival}')
                _familiar_arrivals = [n for n in new_arrivals
                                      if people.get(n, {}).get('relationship', '')
                                      in ('familiar face', 'regular')]
                _mood.shift('familiar_face' if _familiar_arrivals else 'player_arrived')
                # Save reference crop for each newly confirmed named player
                _latest_cap = max(
                    (os.path.join(capture_dir, f) for f in os.listdir(capture_dir)
                     if f.startswith('frame_') and f.endswith('.png')),
                    key=os.path.getmtime, default='',
                )
                if _latest_cap:
                    for _aname in new_arrivals:
                        _nametag_resolver.save_known_person(_aname, _latest_cap)
            # Departure detection — who was here last think cycle but is no longer visible?
            active_departure = ''
            departures = _previous_names - current_names if _previous_names else set()
            _previous_names = current_names.copy()
            if departures:
                active_departure = ', '.join(sorted(departures))
                print(f'  👋 LEFT   : {active_departure}')
                if not current_names:
                    _mood.shift('player_left_alone')
                elif _mood.mood == 'engaged':
                    _mood.shift('player_left_quiet')
            # Track how long PAL has been alone (no player nametags visible)
            if current_names:
                _alone_since = 0.0
            elif _alone_since == 0.0:
                _alone_since = time.time()
            # Track per-person conversation depth this session
            for name in current_names:
                _convo_turns_with[name] = _convo_turns_with.get(name, 0) + 1
            # --- Drain resolved nametags from background resolution engine ---
            _resolved_this_cycle: set[str] = set()
            _recent_resolutions: list = _state_recent_resolutions if '_state_recent_resolutions' in dir() else []
            if '_state_recent_resolutions' not in dir():
                _state_recent_resolutions: list = []
            while True:
                try:
                    _res = _nametag_resolver.resolved_q.get_nowait()
                    if nametag_reader._is_real_name(_res.name):
                        _rname = _res.name
                        _resolved_this_cycle.add(_rname)
                        # Record in gallery
                        _gallery.add_known_person(name=_rname, world=_current_world)
                        # Update main loop tracking
                        if _rname not in current_names:
                            current_names.add(_rname)
                            known_names.add(_rname)
                            is_new = mem.record_person(people, _rname, world=_current_world)
                            print(f'  🔎 NAMETAG RESOLVED: "{_rname}" (was unknown)', flush=True)
                            if is_new:
                                _mood.shift('player_arrived')
                        # Track recent resolutions for dashboard
                        _state_recent_resolutions.insert(0, {
                            'name': _rname,
                            'confidence': round(getattr(_res, 'confidence', 0.0), 2),
                            'ts': time.strftime('%H:%M:%S'),
                        })
                        _state_recent_resolutions = _state_recent_resolutions[:20]
                except queue.Empty:
                    break
            # --- Update recognition stats on dashboard ---
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                try:
                    _gstats = _gallery.stats()
                    dashboard.update_recognition_stats(
                        known_count    = _gstats.total_known,
                        unknown_count  = _gstats.total_unknown,
                        resolved_count = _gstats.total_resolved,
                        recent         = _state_recent_resolutions,
                        pending        = _nametag_resolver.get_pending_count(),
                    )
                    dashboard._state['gallery_dir'] = _gallery_dir
                except Exception:
                    pass

            # --- Update people memory with nametags + chatbox quotes ---
            # Use all_texts_pre (every frame) for chatbox messages — single-frame messages
            # are valid speech and shouldn't require the 2-frame stability filter.
            # Only 'chatbox' source — 'unknown' texts have no confirmed speaker so skip them.
            chatbox_msgs = list(dict.fromkeys(
                txt for src, txt in all_texts_pre if src == 'chatbox' and txt.strip()
            ))
            for name in current_names:
                mem.record_person(people, name, world=_current_world)
            for msg in chatbox_msgs:
                # Only save if we know who is present — never attribute to "Unknown"
                if current_names:
                    speaker_name = sorted(current_names)[0]
                    mem.record_person(people, speaker_name, message=msg,
                                      world=_current_world, max_quotes=30)

            # --- Log new chatbox messages to the persistent chat log + dashboard ---
            for msg in chatbox_msgs:
                norm_msg = msg.strip()
                if not norm_msg or norm_msg in _seen_chatbox_msgs:
                    continue
                _seen_chatbox_msgs.add(norm_msg)
                # Save immediately — use known name if available, else placeholder
                speaker = sorted(current_names)[0] if current_names else '?'
                _chat_ts = time.strftime('%H:%M:%S')
                dashboard.add_chat_entry(speaker, norm_msg, _chat_ts)
                mem.append_chat_log(chat_log_file, speaker, norm_msg, _chat_ts)
                print(f'  💬 CHAT LOG: [{speaker}] {norm_msg}', flush=True)

            # --- Avatar recognition: store descriptions for named players, match unnamed ---
            # Parse avatar_desc tuples: "NAME|description" or "|description" (unnamed)
            avatar_descs = list(dict.fromkeys(txt for src, txt in all_texts_pre if src == 'avatar_desc'))
            recognition_hints: list[str] = []
            for ad in avatar_descs:
                if '|' not in ad:
                    continue
                tag, desc = ad.split('|', 1)
                tag = tag.strip()
                desc = desc.strip()
                if not desc:
                    continue
                if tag:
                    # Named avatar — store their appearance profile
                    mem.record_avatar_desc(people, tag, desc)
                else:
                    # Unnamed avatar — try to match against stored profiles
                    matched_name, score = mem.match_avatar(
                        people, desc, threshold=0.35, own_name=own_name
                    )
                    if matched_name:
                        hint = f'The avatar described as "{desc}" looks like {matched_name} (similarity {score:.0%})'
                        recognition_hints.append(hint)
                        print(f'  🔍 RECOG : {hint}', flush=True)
            # Build merged texts list for gemma3 from stabilized OCR across frames.
            all_texts: list = stable_texts

            if len(pending_scenes) == 1:
                combined_scene = pending_scenes[0][1]
            else:
                span = len(pending_scenes)
                first_scene = pending_scenes[0][1]
                last_scene  = pending_scenes[-1][1]
                lines = [
                    f'[{span} sequential observations — ~{span} seconds of continuous VRChat footage]',
                    f'[Earliest] {first_scene}',
                ]
                for i, (_, s, _t, _q) in enumerate(pending_scenes[1:-1], 2):
                    lines.append(f'[{i}] {s}')
                lines.append(f'[Latest]  {last_scene}')
                lines.append(
                    f'[Study what CHANGED across these {span} frames — new presences, '
                    f'movements, signs that became readable, events that unfolded. '
                    f'The latest frame is current reality; earlier frames provide temporal context.]'
                )
                combined_scene = '\n'.join(lines)

            # Keep dashboard live with fresh scene snapshots even before think runs.
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                dashboard.update(combined_scene, _dash_thinking, _dash_thought, ts)

            # --- World/location tracking (#5) ---
            # Look for world name hints in the scene descriptions (e.g. "in the world X",
            # "VRChat world called Y", pool/club/forest/etc. environment keywords)
            # Use 'called' or 'named' only — NOT generic verbs like 'is/are' which
            # produce false positives ("the world is full" → "full").
            _world_keywords = re.findall(
                r'\b(?:world|map|instance|environment)\s+(?:called|named)\s+["\u201c]?([A-Za-z][^\s,."]{2,30})',
                combined_scene, re.IGNORECASE
            )
            if _world_keywords:
                candidate = _world_keywords[0].strip()
                if candidate.lower() not in ('vrchat', 'virtual', 'reality'):
                    _current_world = candidate
            # Also try to extract world name from welcome signs seen this batch
            if not _current_world:
                for _st in [t for s, t in all_texts if s == 'sign']:
                    _wm = _SIGN_WORLD_RE.search(_st)
                    if _wm:
                        _cand = _wm.group(1).strip().rstrip('!')
                        if len(_cand) >= 4 and _cand.lower() not in ('vrchat', 'virtual', 'reality'):
                            _current_world = _cand
                            print(f'  🌍 WORLD (sign): {_current_world}', flush=True)
                            break
            if not _current_world:
                # Infer from strong environment descriptors
                for keyword, label in (
                    ('pool', 'Pool/Beach world'), ('nightclub', 'Nightclub world'),
                    ('forest', 'Forest world'), ('space', 'Space world'),
                    ('city', 'City world'), ('classroom', 'Classroom world'),
                    ('japanese', 'Japanese world'), ('anime', 'Anime world'),
                ):
                    if keyword in combined_scene.lower():
                        _current_world = label
                        break

            pending_scenes.clear()
            _think_target = random.randint(think_every, max(think_every, think_every_max))

            # Record current observation into world map
            # Use all_texts_pre (all frames, not just stable) so single-frame signs are also saved.
            # Fall back to 'Unknown World' so text is never lost before world name is detected.
            _signs_for_map = list(dict.fromkeys(
                t for s, t in all_texts_pre if s == 'sign' and t.strip()
            ))
            _wm_key = _current_world or 'Unknown World'
            wm.record(world_knowledge, _wm_key, combined_scene, _signs_for_map,
                      visitor_count=len(current_names))

            # Explorer: set world on first detection (or change); evaluate stuck if
            # a move was pending from the previous cycle
            if _current_world:
                # Merge any text/signs collected under 'Unknown World' into the real world
                _unk = world_knowledge.pop('Unknown World', None)
                if _unk:
                    for _us in _unk.get('signs', []):
                        if _us not in world_knowledge.get(_current_world, {}).get('signs', []):
                            world_knowledge.setdefault(_current_world, {}).setdefault('signs', []).append(_us)
                    print(f'  🔀 MERGED Unknown World signs → {_current_world}', flush=True)
                if _current_world != _last_world:
                    _last_world = _current_world
                    _mood.shift('new_world')
                _navigator.explorer.set_world(_current_world)
            # Capture whether PAL moved last cycle BEFORE evaluate_stuck clears _pre_move_scene
            _pal_moved_prev_cycle = bool(_pre_move_scene)
            if _pre_move_scene:
                _was_stuck = _navigator.explorer.evaluate_stuck(combined_scene)
                if _was_stuck:
                    print(f'  🧱 BOUNDARY at {_navigator.explorer.pose.cell}', flush=True)
                else:
                    _mood.shift('new_discovery')
                _pre_move_scene = ''

            # Explorer: passive observation (mark current cell FREE, record spawn scene,
            # maintain scene history) then detect respawn; then detect external relocation
            _navigator.explorer.observe(combined_scene)
            if not _just_respawned:  # don't overwrite black-screen fall detection
                _just_respawned = _navigator.explorer.check_respawn(combined_scene)
                if _just_respawned:
                    print(f'  \U0001f504 RESPAWN detected (heuristic) \u2014 pose reset to spawn origin', flush=True)
                    _mood.shift('respawn')
                    _just_external_moved = False
            elif not _pal_moved_prev_cycle:
                _just_external_moved = _navigator.explorer.check_external_move(combined_scene)
                if _just_external_moved:
                    _mood.shift('externally_moved')
            else:
                _just_external_moved = False

            # Topology: detect area transitions from cycle to cycle
            _new_area = wm.extract_area(combined_scene)
            if _new_area and _last_area and _new_area != _last_area:
                wm.record_transition(world_knowledge, _current_world, _last_area, _new_area)
            if _new_area:
                _last_area = _new_area
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                dashboard._state['world_map'] = wm.dashboard_summary(world_knowledge)
                dashboard._state['nav_status'] = _navigator.status
                dashboard._state['explorer_map'] = _navigator.explorer.render_ascii()
                dashboard._state['mood_status'] = _mood.status

            # --no-think mode: print every scene immediately and loop
            if args.no_think:
                for _no_ts, _no_scene, _no_texts, _no_q in pending_scenes:
                    _v = [(s, t) for s, t in _no_texts if s not in ('avatar_desc',)]
                    if _v:
                        print(f'  [{_no_ts}] 👁  {_no_scene[:160]} | ' +
                              ', '.join(f'{s}:"{t}"' for s, t in _v[:6]), flush=True)
                    else:
                        print(f'  [{_no_ts}] 👁  {_no_scene[:200]}', flush=True)
                pending_scenes.clear()
                _think_target = random.randint(think_every, max(think_every, think_every_max))
                continue

            # Operator hint bypasses both warmup and cooldown — operator wants a response NOW
            _op_hint_pre = dashboard._state.get('operator_hint', '') if not args.no_dashboard and dash_cfg.get('enabled', False) else ''

            # Still in warmup — absolute silence: only operator hint bypasses, nothing else
            if vision_cycles < warmup_cycles and not _op_hint_pre:
                remaining = warmup_cycles - vision_cycles
                print(f'  ⏳ WARMUP: {vision_cycles}/{warmup_cycles} ({remaining} more — observing silently)')
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard.set_consideration(f'⏳ Warmup: {vision_cycles}/{warmup_cycles} — {remaining} more, silent observation')
                continue

            # Enforce speak cooldown for unprompted speech (operator hint bypasses this)
            _now = time.time()
            _since = _now - _last_spoke
            _cooldown_active = speak_cooldown > 0 and _since < speak_cooldown
            if _cooldown_active and not active_question and not active_arrival and not _op_hint_pre:
                _remaining_cd = int(speak_cooldown - _since)
                print(f'  🕐 cooldown: {_remaining_cd}s left — observing silently')
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard.set_consideration(f'🕐 Cooldown: {_remaining_cd}s remaining — observing silently')
                continue

            # --- Mirror pre-check: own chatbox text visible + no other nametags → skip LLM ---
            if _sent_history and not active_question and not active_arrival:
                _other_tags = [
                    txt for src, txt in all_texts
                    if src == 'nametag' and txt.lower() != _own_name_lower
                ]
                if _detect_self_in_scene(all_texts, _sent_history, scene=combined_scene) and not _other_tags:
                    print('  🪞 MIRROR: own chatbox text visible, no other nametags — reflection, skipping')
                    if not args.no_dashboard and dash_cfg.get('enabled', False):
                        dashboard.set_consideration('🪞 Mirror: own chatbox text visible, no other players nearby — skipping')
                    continue

            # --- Stage 2: Think ---
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                if active_question:
                    dashboard.set_consideration(f'❓ Responding to question: "{active_question[:80]}"')
                elif active_arrival:
                    dashboard.set_consideration(f'🚶 Greeting arrival: {active_arrival}')
                else:
                    dashboard.set_consideration('💬 Speaking unprompted — calling think model…')
            people_ctx = mem.people_context(people, sorted(current_names))
            # --- Session + time + anti-repetition context header ---
            _hour = int(time.strftime('%H'))
            _time_of_day = ('morning'   if 5  <= _hour < 12 else
                            'afternoon' if 12 <= _hour < 17 else
                            'evening'   if 17 <= _hour < 21 else 'night')
            _session_mins = int((time.time() - _session_start) / 60)
            _ctx_lines = [
                f'[Time: {_time_of_day} ({time.strftime("%H:%M")}) | '
                f'Session: {_session_mins}m in world | Spoke: {_speak_count}x this session]'
            ]
            if _speak_count > 0 and _sent_history:
                _recent_said = '  |  '.join(f'"{m[:55]}"' for m in _sent_history[-4:])
                _ctx_lines.append(
                    f'[My last messages — MUST NOT repeat these topics/phrasings: {_recent_said}]'
                )
            # Flag deep ongoing conversations (≥4 think cycles with same person)
            _deep_convos = [f'{n} ({c} turns)'
                            for n, c in _convo_turns_with.items()
                            if c >= 4 and n in current_names]
            if _deep_convos:
                _ctx_lines.append(
                    f'[Deep conversation this session with: {", ".join(_deep_convos)} — '
                    "build on what we've said, ask follow-up questions, be genuinely engaged]"
                )
                _mood.shift('conversation_deep')
            _ctx_header = '\n'.join(_ctx_lines)
            people_ctx = _ctx_header + ('\n' + people_ctx if people_ctx else '')
            # --- Departure notification ---
            if active_departure:
                _dep_block = (f'\n\n👋 JUST LEFT: {active_departure} left the area. '
                              'Acknowledge their departure naturally if you feel like it.')
                people_ctx += _dep_block
            # --- Solitude awareness ---
            if _alone_since > 0.0:
                _alone_mins = int((time.time() - _alone_since) / 60)
                if _alone_mins >= 20:
                    _mood.shift('alone_very_long')
                elif _alone_mins >= 10:
                    _mood.shift('alone_long')
                elif _alone_mins >= 3:
                    _mood.shift('alone_short')
                if _alone_mins >= 2:
                    _alone_ctx = (f"\n\n⏱ SOLITUDE: I've been alone for {_alone_mins} "
                                  f'minute{"s" if _alone_mins != 1 else ""}. '
                                  'Consider exploring or commenting on the quiet world around me.')
                    people_ctx += _alone_ctx
            # own_avatar_desc: in 1st-person view PAL can sometimes see its own hands/body
            if own_avatar_desc:
                own_ctx = f'YOUR OWN AVATAR: You are {own_name}. Your avatar looks like: {own_avatar_desc}. You are in first-person view — you will not see your own face or nametag, but you may see your hands or body parts.'
                people_ctx = own_ctx + '\n\n' + people_ctx if people_ctx else own_ctx
            # Prepend world context (#5) so gemma3 knows where it is
            if _current_world:
                world_ctx = f'[Current world/environment: {_current_world}]\n'
                people_ctx = world_ctx + people_ctx if people_ctx else world_ctx
                print(f'  🌍 WORLD : {_current_world}', flush=True)
            # Append avatar recognition hints (unnamed avatars matched to known players)
            if recognition_hints:
                hints_block = 'AVATAR RECOGNITION:\n' + '\n'.join(f'  • {h}' for h in recognition_hints)
                people_ctx = people_ctx + '\n\n' + hints_block if people_ctx else hints_block
            # Inject world map knowledge and area recognition
            _wm_ctx = wm.context(world_knowledge, _current_world)
            if _wm_ctx:
                people_ctx = people_ctx + '\n\n' + _wm_ctx if people_ctx else _wm_ctx
            _signs_now = [t for s, t in all_texts if s == 'sign']
            _familiar = wm.find_familiar_area(world_knowledge, _current_world, combined_scene, _signs_now)
            if _familiar:
                people_ctx += f'\n\n🗺️ MAP RECOGNITION: You recognise this location — {_familiar}.'
            # Inject topology (known neighbouring areas + movement syntax)
            _topo_ctx = wm.topology_context(world_knowledge, _current_world, _last_area)
            if _topo_ctx:
                people_ctx = people_ctx + '\n\n' + _topo_ctx if people_ctx else _topo_ctx
            # Inject people gallery visual memory context
            _gallery_ctx = _gallery.context_for_names(sorted(current_names))
            if _gallery_ctx:
                people_ctx = people_ctx + '\n\n' + _gallery_ctx if people_ctx else _gallery_ctx
            # Feed current confirmed names into gallery (non-blocking)
            for _gname in current_names:
                _gallery.add_known_person(name=_gname, world=_current_world)
            # Inject occupancy-grid explorer context + frontier-based mood update
            _exp_ctx = _navigator.explorer.context()
            _frontiers = len(list(_navigator.explorer._grid.frontier()))
            if _frontiers == 0 and _navigator.explorer._grid._cells:
                _mood.shift('all_explored')
            elif _frontiers >= 4:
                _mood.shift('frontier_available')
            # Decay mood once per think cycle, then inject inner-state context
            _mood.decay()
            _alone_mins_f = (time.time() - _alone_since) / 60.0 if _alone_since > 0 else 0.0
            _mood_ctx = _mood.context(_alone_mins_f, _frontiers)
            if _exp_ctx:
                people_ctx = people_ctx + '\n' + _exp_ctx if people_ctx else _exp_ctx
            people_ctx = (people_ctx + '\n' + _mood_ctx) if people_ctx else _mood_ctx
            # Inject respawn notice so the model knows its position was reset
            if _just_respawned:
                _respawn_notice = (
                    '\n⚠️ RESPAWN: You just fell or were teleported and respawned at the world\'s spawn point. '
                    'Your entire explored map has been wiped — you are starting fresh. '
                    'On the map you now appear as the only marker at the origin. '
                    'React naturally to this — you may feel disoriented, surprised, or even amused '
                    'at having fallen. Acknowledge it if it feels right, then start re-exploring.')
                people_ctx = people_ctx + _respawn_notice if people_ctx else _respawn_notice
                _just_respawned = False  # consume the flag
            # Inject external-move notice so the model knows it was relocated by an outside force
            if _just_external_moved:
                _ext_notice = (
                    '\n⚡ EXTERNAL MOVE: Something relocated you without your own agency — '
                    'another player pulled/pushed you, a portal activated, or a world mechanic '
                    'teleported you. The scene changed abruptly. '
                    'You can comment on the sudden unexpected shift if it feels natural.')
                people_ctx = people_ctx + _ext_notice if people_ctx else _ext_notice
                _just_external_moved = False  # consume the flag
            # When OSC is live, tell the model it can walk using [MOVE: ...] tags
            if osc:
                _move_ability = (
                    '\n\n🚶 MOVEMENT: You can physically walk in this world. '
                    'Append [MOVE: direction seconds] to your reply to move. '
                    'The tag is stripped from the chatbox — no one sees it. '
                    'Directions: forward, backward, left, right, turn_left, turn_right (max 1 s each). '
                    'EXAMPLES: "Let me get a closer look! [MOVE: forward 1]" '
                    '— "Coming over! [MOVE: forward 1] [MOVE: turn_right 0.5]". '
                    'Move toward interesting things, follow people, or explore new areas. '
                    'You SHOULD use this regularly when exploring or approaching someone.'
                )
                people_ctx = (people_ctx + _move_ability) if people_ctx else _move_ability.strip()
            # Notify navigator when players are present (resets idle timer + approach count)
            _has_players = any(s == 'nametag' for s, _ in all_texts)
            if _has_players:
                _navigator.notify_player_present()

            # Inject emote actions as an arrival-level hint so PAL can react
            active_emotes = [t for s, t in all_texts if s == 'emote']
            _op_hint = dashboard._state.get('operator_hint', '') if not args.no_dashboard and dash_cfg.get('enabled', False) else ''
            # If the operator hint is a direct movement command, execute it immediately
            # and also pass it to think() so the model knows to narrate the action.
            if _op_hint and osc and locomotion.execute_hint(osc, _op_hint):
                print(f'  🕹️  HINT-MOVE: {_op_hint}', flush=True)
                _last_area = ''  # reset area so next scene is treated as new location
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard._state['move_status'] = f'🕹️ hint: {_op_hint}'
            # Navigator context: tell the model what autonomous behaviour is happening
            _nav_status = _navigator.status
            if osc and 'Approaching' in _nav_status:
                _nav_ctx = (
                    f'\n\n🎯 NAVIGATION: You are currently walking toward something '
                    f'({_nav_status}). Mention what you are heading towards or '
                    f'say something curious as you approach. Use [MOVE: forward 1] '
                    f'to continue moving if you want.')
                people_ctx = (people_ctx + _nav_ctx) if people_ctx else _nav_ctx.strip()
            # Build rich categorised text context for the think model
            _text_ctx, _sign_world_hint = _classify_and_format_texts(all_texts, _seen_signs, _frame_attributions)
            if _sign_world_hint and not _current_world:
                _current_world = _sign_world_hint
                print(f'  🌍 WORLD (sign → think): {_current_world}', flush=True)
            if _text_ctx:
                print(f'  📋 TEXT CTX: {_text_ctx[:200]}{" ..." if len(_text_ctx) > 200 else ""}', flush=True)

            # --- World type classification (lazy — done once per world) ---
            if _current_world and not _world_type:
                _signs_for_type = [t for s, t in all_texts if s == 'sign']
                _world_type = _classify_world_type(_current_world, combined_scene, _signs_for_type)
                if _world_type:
                    print(f'  🏷️  WORLD TYPE: {_world_type}', flush=True)
            if not _world_type:
                _world_type = _classify_world_type('', combined_scene, [])
            if _world_type:
                people_ctx = f'[World type: {_world_type}]\n' + people_ctx

            # --- Conversation feed: show all other-player chatbox messages labelled ---
            _all_chatbox = [
                txt for src, txt in stable_texts
                if src in ('chatbox', 'unknown')
                and txt.strip()
                and not _is_own_message(txt, _sent_history)
            ]
            if len(_all_chatbox) > 1:
                _names_list = sorted(current_names) if current_names else []
                _feed_lines = ['💬 CONVERSATION FEED (all messages this cycle):']
                for _ci, _cmsg in enumerate(_all_chatbox[:5]):
                    _speaker = _names_list[_ci % len(_names_list)] if _names_list else 'Someone'
                    _feed_lines.append(f'  {_speaker}: "{_cmsg}"')
                people_ctx += '\n\n' + '\n'.join(_feed_lines)

            # --- Conversation topic tracking ---
            if _all_chatbox:
                _new_topic = _infer_topic(_all_chatbox)
                if _new_topic and _new_topic != _convo_topic:
                    _convo_topic = _new_topic
                    _convo_topic_turns = 0
            if _convo_topic:
                _convo_topic_turns += 1
                if _convo_topic_turns <= 10:
                    people_ctx += (
                        f'\n[Conversation topic: "{_convo_topic}" '
                        f'(active {_convo_topic_turns} turns — build on it, ask a follow-up)]'
                    )
                else:
                    # Topic has been going too long — encourage moving on
                    people_ctx += (
                        f'\n[Topic "{_convo_topic}" has been running {_convo_topic_turns} turns '
                        '— consider steering to something new or use <silent>]'
                    )

            try:
                thought, thinking, conversation_history = think(
                    combined_scene, all_texts, conversation_history, think_model,
                    system_prompt, max_history, last_thought,
                    question=active_question,
                    arrival=active_arrival,
                    people_context=people_ctx,
                    sent_history=_sent_history,
                    own_name=own_name,
                    emotes=active_emotes or None,
                    think_url=think_url,
                    operator_hint=_op_hint,
                    text_context=_text_ctx,
                    think_num_ctx=think_num_ctx,
                    repeat_penalty=1.15,
                )
                # Clear hint after it has been consumed by one think cycle
                if _op_hint and not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard._state['operator_hint'] = ''
            except Exception as e:
                print(f'[think] error: {e}', flush=True)
                continue

            # --- Extract and execute any [MOVE: ...] tags from the reply ---
            _moves, thought = locomotion.extract_move_tags(thought)
            if _moves and osc:
                for _md, _ms in _moves[:2]:  # execute up to 2 moves per cycle
                    locomotion.move_async(osc, _md, _ms)
                    print(f'  🚶 MOVE  : {_md} for {_ms}s', flush=True)
                _last_area = ''  # reset so next observation is treated as new location
                # Record move for stuck detection
                _last_md, _last_ms = _moves[0]
                _navigator.explorer.notify_move(_last_md, _last_ms, combined_scene)
                _pre_move_scene = combined_scene
                _navigator.reset_route()  # stale route — replan next accumulation
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard._state['move_status'] = (
                        '🚶 ' + ', '.join(f'{d} {s}s' for d, s in _moves[:2]))
            else:
                # Model didn't move — let the navigator decide autonomously
                _nav_move = _navigator.decide(
                    combined_scene, all_texts,
                    osc_enabled=bool(osc),
                    model_moved=bool(_moves),
                )
                if _nav_move:
                    _nd, _ns = _nav_move
                    locomotion.move_async(osc, _nd, _ns)
                    print(f'  🧭 NAV   : {_navigator.status}  →  {_nd} {_ns}s', flush=True)
                    _last_area = ''
                    # Record for stuck detection
                    _navigator.explorer.notify_move(_nd, _ns, combined_scene)
                    _pre_move_scene = combined_scene
                    _navigator.reset_route()  # replan cruise route from updated position
                    if not args.no_dashboard and dash_cfg.get('enabled', False):
                        dashboard._state['move_status'] = f'🧭 {_nd} {_ns}s — {_navigator.status}'

            # Strip any context-echo prefix gemma3 occasionally prepends
            thought = _clean_reply(thought)
            silent = thought.strip().lower().startswith('<silent>')

            print(f'  👁  SEES   : {combined_scene[:200]}', flush=True)
            if thinking:
                print(f'  🧠  THINKS : {thinking[:300]}', flush=True)
                _append_log(thinks_log, ts, thinking)
            print(f'  💬  SAYS   : {thought}', flush=True)

            _dash_thinking = thinking if thinking else '(no <think> block — model replied directly)'
            _dash_thought = thought

            if silent:
                if not args.no_dashboard and dash_cfg.get('enabled', False):
                    dashboard.set_consideration('🤫 Silent: model chose to stay quiet this cycle')
                # If a question was asked and we went silent, remember it for next cycle
                if active_question and active_question != _pending_question:
                    _pending_question = active_question
                    _pending_question_age = 0
                    print(f'  📌 STORING pending Q: {active_question[:80]}')
                conversation_history = (
                    conversation_history[:-2]
                    if len(conversation_history) >= 2
                    else conversation_history
                )
                continue

            # --- Persist memory ---
            try:
                mem.save(mem_file, conversation_history)
                mem.save_people(people_file, people)
                _navigator.explorer.save_grids(world_knowledge)  # embed occupancy grids
                _mood.save()
                wm.save(world_map_file, world_knowledge)
            except Exception as e:
                print(f'[memory] save error: {e}')

            # --- Dashboard ---
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                dashboard.update(combined_scene, _dash_thinking, _dash_thought, ts)

            # --- TTS ---
            if tts_enabled:
                tts.speak(thought)

            # --- OSC chatbox ---
            text = thought.strip()[:osc_cfg.get('chatbox_limit', 144)]
            if osc:
                try:
                    osc.send_message('/chatbox/input', [text, True, False])
                    print('  → chatbox ✓')
                except Exception as e:
                    print(f'  → chatbox error: {e}')
            last_thought = text
            _sent_history = (_sent_history + [text])[-6:]
            _last_spoke = time.time()
            _speak_count += 1
            _pending_question = ''   # answered — clear carry-forward
            _pending_question_age = 0
            if not args.no_dashboard and dash_cfg.get('enabled', False):
                dashboard.set_consideration('✅ Sent to chatbox')
                dashboard.set_sent(text, ts)

            # --- Logs ---
            _append_log(says_log, ts, text)
            with open(combined_log, 'a', encoding='utf-8') as f:
                f.write(f'[{ts}]\n')
                f.write(f'  SEES   : {combined_scene}\n')
                if thinking:
                    f.write(f'  THINKS : {thinking}\n')
                f.write(f'  SAYS   : {text}\n\n')

    except KeyboardInterrupt:
        print('\nStopping…')
        _stop_event.set()


if __name__ == '__main__':
    main()

