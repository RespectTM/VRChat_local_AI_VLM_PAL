"""Spatial world memory — builds a persistent map of each VRChat world PAL observes.

Each world entry accumulates:
  • signs/text objects seen (permanent fixtures like noticeboards, banners)
  • distinct area/feature descriptions (what different parts of the world look like)
  • observation count + first/last-seen dates

This is saved to JSON between sessions so PAL remembers worlds it has visited before.
"""
import json
import os
import re
import time
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load(path: str) -> Dict:
    """Load world map from JSON. Returns {} if file missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save(path: str, data: Dict) -> None:
    """Persist world map to JSON."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record(world_map: Dict, world_name: str,
           scene: str, signs: List[str],
           visitor_count: int = 0,
           max_signs: int = 40, max_areas: int = 30) -> None:
    """Add one observation cycle to the world map.

    world_name:    inferred world/environment name — skip if empty.
    scene:         combined scene description string from vision model.
    signs:         list of sign/notice/world-text strings from vision output.
    visitor_count: number of player nametags visible this cycle.
    """
    if not world_name:
        return

    entry = world_map.setdefault(world_name, {
        'first_seen': time.strftime('%Y-%m-%d'),
        'observations': 0,
        'signs': [],
        'areas': [],
    })
    entry['last_seen'] = time.strftime('%Y-%m-%d')
    entry['observations'] = entry.get('observations', 0) + 1

    # Track rolling visitor counts to estimate typical crowd size
    counts = entry.setdefault('visitor_counts', [])
    counts.append(visitor_count)
    if len(counts) > 20:
        counts.pop(0)
    entry['typical_crowd'] = round(sum(counts) / len(counts), 1)

    # Accumulate distinct signs (case-insensitive dedup, hallucination filtered)
    known_lower = {s.lower() for s in entry['signs']}
    for s in signs:
        s = s.strip()
        if len(s) >= 3 and s.lower() not in known_lower and not _is_hallucinated_text(s):
            entry['signs'].append(s)
            known_lower.add(s.lower())
            if len(entry['signs']) > max_signs:
                entry['signs'].pop(0)

    # Extract and store a compact area phrase from the scene
    area = _extract_area(scene)
    if area and area not in entry['areas'] and not _is_hallucinated_text(area):
        entry['areas'].append(area)
        if len(entry['areas']) > max_areas:
            entry['areas'].pop(0)


# Phrases that indicate the vision model returned its own prompt or structured output
# instead of actual world content — these must never be saved as memories.
_HALLUCINATION_PATTERNS = re.compile(
    r'every piece of readable text'
    r'|based on the image'
    r'|following (?:the|your|all) instructions'
    r'|```json'
    r'|{"scene"'
    r'|structured (?:json|description|output|analysis)'
    r'|as per (?:the|your) instructions'
    r'|\binstruction(?:s)?\s+(?:provided|carefully|given)'
    r'|in this (?:screenshot|image|scene),?\s+(?:the|we|i)'
    r'|this is (?:a|an|the) (?:screenshot|image|scene)'
    r'|\[(?:earliest|latest|sequential observations)'
    r'|nametag(?:s)? (?:above|visible|indicating)'
    r'|the player.s previous message'
    r'|do not include the cent(?:re|er) avatar',
    re.IGNORECASE,
)


def _is_hallucinated_text(text: str) -> bool:
    """Return True if *text* looks like a leaked prompt or model meta-output."""
    if len(text) > 200:  # suspiciously long — likely a raw model dump
        return True
    return bool(_HALLUCINATION_PATTERNS.search(text))


def _extract_area(scene: str) -> str:
    """Extract a compact environment phrase from a scene description string."""
    # Use first sentence only
    first = re.split(r'[.!?]', scene.strip())[0].strip()
    # Strip generic openers that add no spatial information
    first = re.sub(
        r'^(?:a|an|the|this is a?n?|this shows?|i (?:can )?see'
        r'|showing?|depicting?|in this (?:screenshot|image|scene)'
        r'|VRChat\s*(?:scene\s*(?:with)?)?|scene\s*(?:with)?'
        r'|avatars?\s+(?:in|at|near)?)\s+',
        '', first, flags=re.IGNORECASE
    ).strip()
    if len(first) > 90:
        first = first[:87] + '…'
    return first.lower() if len(first) >= 8 else ''


def extract_area(scene: str) -> str:
    """Public alias for _extract_area — call from outside this module."""
    return _extract_area(scene)


# ---------------------------------------------------------------------------
# Topology — builds a graph of connected areas as PAL moves around
# ---------------------------------------------------------------------------

def _area_key(area: str) -> str:
    """Normalised dict key for a topology node (lowercase, collapsed whitespace, ≤60 chars)."""
    return re.sub(r'\s+', ' ', area.strip().lower())[:60]


def _area_similarity(a: str, b: str) -> float:
    """Word-overlap (Jaccard) between two area strings. Returns 0.0–1.0."""
    wa = set(re.findall(r'[a-z]{4,}', a.lower()))
    wb = set(re.findall(r'[a-z]{4,}', b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def record_transition(world_map: Dict, world_name: str,
                      from_area: str, to_area: str) -> None:
    """Record that PAL moved from *from_area* to *to_area* inside *world_name*.

    Builds a bidirectional adjacency entry in
    ``world_map[world_name]['topology']``.
    Skip if the two areas are too similar (probably the same location).
    """
    if not world_name or not from_area or not to_area:
        return
    if _area_similarity(from_area, to_area) >= 0.4:
        return  # not a real transition

    entry = world_map.setdefault(world_name, {
        'first_seen': time.strftime('%Y-%m-%d'),
        'observations': 0,
        'signs': [],
        'areas': [],
    })
    topo = entry.setdefault('topology', {})

    fk = _area_key(from_area)
    tk = _area_key(to_area)

    fn = topo.setdefault(fk, {'label': from_area[:60], 'connects_to': []})
    if tk not in fn['connects_to']:
        fn['connects_to'].append(tk)

    tn = topo.setdefault(tk, {'label': to_area[:60], 'connects_to': []})
    if fk not in tn['connects_to']:
        tn['connects_to'].append(fk)


def topology_context(world_map: Dict, world_name: str, current_area: str,
                     max_neighbors: int = 6) -> str:
    """Return a prompt block describing known connected areas from the current location.

    Also tells the model the [MOVE: ...] syntax so it can navigate.
    Returns empty string when there is no topology data yet.
    """
    if not world_name or not current_area:
        return ''
    entry = world_map.get(world_name)
    if not entry:
        return ''
    topo = entry.get('topology', {})
    if not topo:
        return ''

    ck   = _area_key(current_area)
    node = topo.get(ck)

    if not node:
        # Fuzzy match: find the closest known node
        best_k, best_sim = '', 0.0
        cur_words = set(re.findall(r'[a-z]{4,}', current_area.lower()))
        for k in topo:
            kw = set(re.findall(r'[a-z]{4,}', k))
            if not kw:
                continue
            sim = len(cur_words & kw) / len(cur_words | kw)
            if sim > best_sim:
                best_sim, best_k = sim, k
        if best_sim < 0.2 or not best_k:
            return ''
        node = topo[best_k]

    connected = node.get('connects_to', [])
    if not connected:
        return ''

    labels = []
    for k in connected[:max_neighbors]:
        n = topo.get(k)
        labels.append(f'"{n["label"]}"' if n else f'"{k}"')

    return (
        f'\U0001f5fa\ufe0f MOVEMENT TOPOLOGY: From your current location you have previously '
        f'walked to: {", ".join(labels)}. '
        f'You can move using [MOVE: direction seconds] e.g. [MOVE: forward 2]. '
        f'Directions: forward, backward, left, right, turn_left, turn_right. Max 10 s.'
    )


# ---------------------------------------------------------------------------
# Recognition — match current scene against known areas of this world
# ---------------------------------------------------------------------------

def find_familiar_area(world_map: Dict, world_name: str,
                       scene: str, signs: List[str]) -> str:
    """Check whether the current scene matches a previously seen area.

    Returns a short recognition note (e.g. 'sign "VR test" seen here before')
    or '' if nothing familiar is found.
    """
    if not world_name:
        return ''
    entry = world_map.get(world_name)
    if not entry or entry.get('observations', 0) < 3:
        # Not enough data yet to make recognition claims
        return ''

    # Sign match: any currently-visible sign was noted here before
    known_signs = {s.lower() for s in entry.get('signs', [])}
    for s in signs:
        if s.strip().lower() in known_signs:
            return f'sign "{s}" was recorded here in a previous visit'

    # Area word-overlap match: current scene vs stored area phrases
    current_words = set(re.findall(r'[a-z]{4,}', scene.lower()))
    best_score, best_area = 0.0, ''
    for area in entry.get('areas', []):
        area_words = set(re.findall(r'[a-z]{4,}', area))
        if not area_words:
            continue
        overlap = len(current_words & area_words) / len(current_words | area_words)
        if overlap > best_score:
            best_score = overlap
            best_area = area
    if best_score >= 0.30 and best_area:
        return f'area resembling "{best_area}" (visited before)'

    return ''


# ---------------------------------------------------------------------------
# Context for think prompt
# ---------------------------------------------------------------------------

def context(world_map: Dict, world_name: str,
            max_signs: int = 12, max_areas: int = 6) -> str:
    """Build a compact world-knowledge block to inject into the think prompt."""
    if not world_name:
        return ''
    entry = world_map.get(world_name)
    if not entry:
        return ''
    obs = entry.get('observations', 0)
    first = entry.get('first_seen', '?')
    lines = [f'WORLD KNOWLEDGE — {world_name} (observed {obs}x, first visited {first}):']
    signs = entry.get('signs', [])
    if signs:
        sign_list = ', '.join(f'"{s}"' for s in signs[-max_signs:])
        lines.append(f'  Permanent signs/text here: {sign_list}')
    areas = entry.get('areas', [])
    if areas:
        area_list = '; '.join(areas[-max_areas:])
        lines.append(f'  Areas & features seen: {area_list}')
    typical_crowd = entry.get('typical_crowd')
    if typical_crowd is not None:
        if typical_crowd < 1:
            crowd_desc = 'usually empty'
        elif typical_crowd < 2:
            crowd_desc = 'usually 1 person'
        else:
            crowd_desc = f'usually ~{typical_crowd:.0f} people'
        lines.append(f'  Typical crowd: {crowd_desc}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------

def dashboard_summary(world_map: Dict) -> str:
    """Multi-line summary of all mapped worlds for the dashboard."""
    if not world_map:
        return 'No worlds mapped yet.'
    lines = []
    for name, entry in sorted(world_map.items()):
        obs   = entry.get('observations', 0)
        nsign = len(entry.get('signs', []))
        narea = len(entry.get('areas', []))
        ntopo = len(entry.get('topology', {}))
        last  = entry.get('last_seen', '?')
        topo_str = f', {ntopo} topo nodes' if ntopo else ''
        lines.append(f'📍 {name}  —  {obs} scans, {nsign} signs, {narea} areas{topo_str}  (last: {last})')
        signs = entry.get('signs', [])
        if signs:
            lines.append('   Signs: ' + ', '.join(f'"{s}"' for s in signs[-6:]))
        areas = entry.get('areas', [])
        if areas:
            lines.append('   Areas: ' + '; '.join(areas[-3:]))
        topo = entry.get('topology', {})
        if topo:
            # Show a few edges in the dashboard
            edge_lines = []
            for k, nd in list(topo.items())[:4]:
                neighbors = nd.get('connects_to', [])
                if neighbors:
                    nb_labels = ', '.join(
                        f'"{topo[nk]["label"]}"' if nk in topo else f'"{nk}"'
                        for nk in neighbors[:3]
                    )
                    edge_lines.append(f'"{nd["label"]}" → {nb_labels}')
            if edge_lines:
                lines.append('   Paths: ' + ' | '.join(edge_lines))
    return '\n'.join(lines)
