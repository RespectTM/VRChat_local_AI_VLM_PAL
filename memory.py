"""
Persistent conversation memory — saves/loads gemma3:12b chat history as JSON.
"""
import json
import os
import re
import time
from typing import List, Dict


def load(path: str) -> List[Dict]:
    """Load conversation history from JSON. Returns [] if file missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save(path: str, history: List[Dict]) -> None:
    """Persist conversation history to JSON."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear(path: str) -> None:
    """Delete the history file (resets memory)."""
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# People memory — remembers names and what they said across sessions
# ---------------------------------------------------------------------------

def load_people(path: str) -> Dict[str, Dict]:
    """Load the people-book from JSON. Returns {} if missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_people(path: str, people: Dict[str, Dict]) -> None:
    """Persist the people-book to JSON."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(people, f, indent=2, ensure_ascii=False)


def record_person(people: Dict[str, Dict], name: str,
                  message: str = '', world: str = '',
                  max_quotes: int = 10) -> bool:
    """Update the people-book with a name (and optionally a message they said).
    Returns True if the person is new (first ever encounter).
    """
    today = time.strftime('%Y-%m-%d')
    is_new = name not in people
    entry = people.setdefault(name, {
        'first_seen': today,
        'seen_count': 0,
        'session_dates': [],
        'worlds_met': [],
        'quotes': [],
        'relationship_score': 0,
    })
    entry['last_seen'] = today
    entry['seen_count'] = entry.get('seen_count', 0) + 1
    # Track unique session dates (max 30) to give a real sense of history
    session_dates = entry.setdefault('session_dates', [])
    if today not in session_dates:
        session_dates.append(today)
        if len(session_dates) > 30:
            session_dates.pop(0)
    # Track which worlds we've crossed paths in (max 10)
    if world:
        worlds_met = entry.setdefault('worlds_met', [])
        if world not in worlds_met:
            worlds_met.append(world)
            if len(worlds_met) > 10:
                worlds_met.pop(0)
    if message:
        quotes = entry.setdefault('quotes', [])
        if message not in quotes:
            quotes.append(message)
            entry['relationship_score'] = entry.get('relationship_score', 0) + 1
            if len(quotes) > max_quotes:
                quotes.pop(0)
    return is_new


def _relationship_level(entry: Dict) -> str:
    """Return a warmth label based on session history and interaction score."""
    sessions = len(entry.get('session_dates', []))
    score = entry.get('relationship_score', 0)
    if sessions <= 1 and score == 0:
        return 'just met'
    if sessions <= 2 and score <= 3:
        return 'seen before'
    if sessions <= 5 or score <= 8:
        return 'regular'
    return 'familiar face'


def record_avatar_desc(people: Dict[str, Dict], name: str,
                       desc: str, max_descs: int = 5) -> None:
    """Store an avatar visual description for a known named player."""
    if not name or not desc:
        return
    entry = people.get(name)
    if entry is None:
        return
    descs = entry.setdefault('avatar_descs', [])
    if desc not in descs:
        descs.append(desc)
        if len(descs) > max_descs:
            descs.pop(0)


def _desc_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two avatar description strings."""
    wa = set(re.findall(r'[a-z]{3,}', a.lower()))
    wb = set(re.findall(r'[a-z]{3,}', b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def match_avatar(people: Dict[str, Dict], desc: str,
                 threshold: float = 0.35, own_name: str = '') -> tuple:
    """Try to identify an unnamed avatar by comparing its description to stored profiles.
    Returns (name, score) of the best match above threshold, or ('', 0.0) if none.
    """
    if not desc:
        return ('', 0.0)
    own_lower = own_name.strip().lower()
    best_name, best_score = '', 0.0
    for name, entry in people.items():
        if own_lower and name.strip().lower() == own_lower:
            continue
        for stored_desc in entry.get('avatar_descs', []):
            score = _desc_similarity(desc, stored_desc)
            if score > best_score:
                best_score = score
                best_name = name
    if best_score >= threshold:
        return (best_name, best_score)
    return ('', 0.0)


# ---------------------------------------------------------------------------
# Chat log — persists every unique chatbox message heard during sessions
# ---------------------------------------------------------------------------

CHAT_LOG_MAX = 500  # max entries stored in file


def load_chat_log(path: str) -> List[Dict]:
    """Load the chat log from JSON. Returns [] if missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def append_chat_log(path: str, name: str, text: str, timestamp: str) -> None:
    """Prepend a new chat entry to the persistent log (newest-first). Caps at CHAT_LOG_MAX."""
    entries = load_chat_log(path)
    entries.insert(0, {'name': name, 'text': text, 'ts': timestamp})
    if len(entries) > CHAT_LOG_MAX:
        entries = entries[:CHAT_LOG_MAX]
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def people_context(people: Dict[str, Dict], names: List[str]) -> str:
    """Build a compact context string for the given names, to inject into the prompt."""
    if not people or not names:
        return ''
    lines = []
    for name in names:
        entry = people.get(name)
        if not entry:
            continue
        relationship = _relationship_level(entry)
        sessions = len(entry.get('session_dates', []))
        session_str = (f'{sessions} sessions together' if sessions > 1
                       else 'first time meeting')
        parts = [f'{name} ({relationship}, {session_str}, last seen {entry.get("last_seen","?")}']        
        worlds_met = entry.get('worlds_met', [])
        if worlds_met:
            parts.append(f'met in: {", ".join(worlds_met[-3:])}')
        quotes = entry.get('quotes', [])
        if quotes:
            parts.append(f'said: {"; ".join(repr(q) for q in quotes[-3:])}')
        avatar_descs = entry.get('avatar_descs', [])
        if avatar_descs:
            parts.append(f'avatar: {avatar_descs[-1]}')
        lines.append(' — '.join(parts) + ')')
    if not lines:
        return ''
    return 'PEOPLE YOU KNOW IN THIS SCENE:\n' + '\n'.join(f'  • {l}' for l in lines)
