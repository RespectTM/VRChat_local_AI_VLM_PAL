"""
Name Validator — Steps 56–65 of the 100-step recognition plan.
================================================================
Validates, cleans, and deduplicates name strings returned by the vision model.

VRChat username rules (as of 2024):
  - 3–32 characters long
  - Letters, digits, underscores, hyphens, dots (and most unicode)
  - Cannot start or end with a special char (rough heuristic)
  - Must contain at least one letter
  - Must NOT be a generic placeholder or descriptor word

The validator catches:
  - Generic model output like "player", "unknown", "name", "user", "person"
  - Garbage strings that are clearly not names ("...", "???", "N/A")
  - Strings that are too short, too long, or entirely non-alphanumeric
  - Near-duplicate candidates that refer to the same name
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Step 57: Blacklist set — strings the vision model commonly hallucinates
# ---------------------------------------------------------------------------

BLACKLIST: frozenset = frozenset({
    # Described/generic
    'player', 'user', 'name', 'person', 'avatar', 'character',
    'nametag', 'tag', 'label', 'text', 'unknown', 'unnamed',
    # Uncertainty markers
    '?', '??', '???', '????', 'unclear', 'unreadable', 'illegible',
    'blurry', 'difficult', 'cannot', "can't", 'none', 'n/a', 'na',
    'nil', 'null', 'empty',
    # Common noise
    'someone', 'somebody', 'anyone', 'anybody', 'nobody',
    'me', 'you', 'they', 'them', 'it',
    # Positional
    'left', 'right', 'center', 'middle', 'front', 'back', 'here',
    # VRChat specific false positives
    'vrchat', 'vrc', 'world', 'instance', 'lobby', 'room', 'friend',
    # Model meta-responses
    'yes', 'no', 'maybe', 'likely', 'probably', 'possible', 'sorry',
    'see', 'found', 'reading', 'shows', 'appears', 'display', 'shown',
    # Sentence fragments
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'i', 'my', 'your', 'their', 'its',
    # Japanese placeholder
    '不明', '？', '名前',
    # Multi-question marks
    '?!', '!?', '!!', '...',
})


# ---------------------------------------------------------------------------
# Step 58: INVALID_PATTERNS list — regex patterns that flag bad strings
# ---------------------------------------------------------------------------

INVALID_PATTERNS: List[re.Pattern] = [
    re.compile(r'^[\W_]+$'),                  # all punctuation / whitespace
    re.compile(r'^[?*.\-_/\\]+$'),            # only noise chars
    re.compile(r'^\d+$'),                     # only digits
    re.compile(r'^[^a-zA-Z0-9\u00C0-\u024F\u4E00-\u9FFF]+$'),  # no letters at all
    re.compile(r'\bname\b', re.I),            # literally says "name"
    re.compile(r'\bplayer\b', re.I),          # literally says "player"
    re.compile(r'\bunknown\b', re.I),         # literally says "unknown"
    re.compile(r'^[a-z]{1,2}$'),              # too short (1-2 lowercase letters)
    re.compile(r'^\s*$'),                     # empty / whitespace
    re.compile(r'[^\x00-\x7F]{5,}'),         # >5 consecutive non-ASCII (garbled)
    re.compile(r'^(?:no\s+name|no\s+tag|not\s+visible)', re.I),
    re.compile(r'(?:cannot|could not|unable to)\s+read', re.I),
    re.compile(r'(?:not\s+(?:readable|clear|visible|legible))', re.I),
]


# ---------------------------------------------------------------------------
# Step 59: is_valid_name
# ---------------------------------------------------------------------------

_MIN_LEN = 2       # absolute minimum for a real name
_MAX_LEN = 40      # generous maximum (VRChat limit is 32, but allow some buffer)


def is_valid_name(s: str) -> bool:
    """
    Return True if s looks like a legitimate VRChat username.
    Checks:
      - length in [MIN_LEN, MAX_LEN]
      - contains at least one letter
      - not in BLACKLIST (case-insensitive)
      - does not match any INVALID_PATTERNS
    """
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if not (_MIN_LEN <= len(s) <= _MAX_LEN):
        return False
    # Must contain at least one letter
    if not re.search(r'[a-zA-Z\u00C0-\u024F\u4E00-\u9FFF]', s):
        return False
    # Blacklist check (case-insensitive, also checks stripped version)
    if s.lower() in BLACKLIST:
        return False
    if s.lower().strip('?!.…_ ') in BLACKLIST:
        return False
    # Regex pattern checks
    for pat in INVALID_PATTERNS:
        if pat.search(s):
            return False
    # Must not be >90% punctuation
    alpha_count = sum(1 for c in s if c.isalpha() or c.isdigit())
    if len(s) > 0 and alpha_count / len(s) < 0.35:
        return False
    return True


# ---------------------------------------------------------------------------
# Step 60: clean_raw_name — strip noise chars and common prefixes
# ---------------------------------------------------------------------------

# Patterns to strip before validation (applied in order)
_STRIP_PREFIXES = re.compile(
    r'^(?:'
    r'\d+[.)]\s*'           # numbered list: "1. ", "2) "
    r'|[-•·‣▸►*−–—]+\s*'  # bullet chars
    r'|Name:\s*'            # "Name: Alice"
    r'|Player:\s*'          # "Player: Bob"
    r'|Tag:\s*'             # "Tag: Carol"
    r'|Nametag:\s*'        # "Nametag: Dave"
    r'|Label:\s*'           # "Label: Eve"
    r')',
    re.I,
)

_STRIP_SUFFIXES = re.compile(
    r'(?:'
    r'\s*[-–—]\s*.*$'       # trailing " — <description>"
    r'|\s*\(.*\)$'          # trailing "(something)"
    r'|\s*\|\s*.*$'         # trailing " | worlds"
    r')',
    re.I,
)

_STRIP_QUOTES = re.compile(r'^["\'\`\u201C\u201D\u2018\u2019]+|["\'\`\u201C\u201D\u2018\u2019]+$')


def clean_raw_name(s: str) -> str:
    """
    Strip list markers, quotes, descriptive suffixes, extra whitespace.
    Returns the cleaned string (may still be invalid — call is_valid_name after).
    """
    s = s.strip()
    s = _STRIP_PREFIXES.sub('', s)
    s = _STRIP_SUFFIXES.sub('', s)
    s = _STRIP_QUOTES.sub('', s)
    s = s.strip('.,;:!?…')
    s = ' '.join(s.split())   # collapse internal whitespace
    return s.strip()


# ---------------------------------------------------------------------------
# Step 61: extract_names_from_json
# ---------------------------------------------------------------------------

def extract_names_from_json(raw: str) -> List[str]:
    """
    Try to parse structured JSON output from the model.
    Accepts: {"names": [...]} or {"nametags": [...]} or {"players": [...]}
    Also handles a bare JSON list: ["Alice", "Bob"]
    Returns validated names only.
    """
    raw = raw.strip()
    # Sometimes the model wraps JSON in markdown code fences
    raw = re.sub(r'^```(?:json)?\n?', '', raw, flags=re.I)
    raw = re.sub(r'\n?```$', '', raw, flags=re.I)
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract partial JSON dict
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
            except Exception:
                return []
        else:
            return []

    candidates: List[str] = []
    if isinstance(obj, list):
        candidates = [str(item) for item in obj]
    elif isinstance(obj, dict):
        for key in ('names', 'nametags', 'players', 'tags', 'results'):
            if key in obj and isinstance(obj[key], list):
                candidates = [str(item) for item in obj[key]]
                break
        if not candidates:
            # Fallback: grab all string values
            candidates = [str(v) for v in obj.values() if isinstance(v, str)]

    return [n for n in (clean_raw_name(c) for c in candidates) if is_valid_name(n)]


# ---------------------------------------------------------------------------
# Step 62: extract_names_from_lines
# ---------------------------------------------------------------------------

def extract_names_from_lines(raw: str) -> List[str]:
    """
    Parse plain-text model output where each name is on its own line.
    Also handles comma-separated lists on a single line.
    """
    # Handle comma-separated single lines like "Alice, Bob, Carol"
    if '\n' not in raw and ',' in raw:
        parts = raw.split(',')
    else:
        parts = raw.splitlines()

    results: List[str] = []
    for part in parts:
        cleaned = clean_raw_name(part)
        if is_valid_name(cleaned):
            results.append(cleaned)
    return results


# ---------------------------------------------------------------------------
# Step 63: extract_names_from_raw — auto-detect format
# ---------------------------------------------------------------------------

def extract_names_from_raw(raw: str) -> List[str]:
    """
    Try JSON first, then plain-text lines.
    Returns all valid unique names found (order preserved).
    """
    if not raw:
        return []
    raw = raw.strip()

    # Try JSON if the output looks structured
    if raw.startswith(('{', '[', '```')):
        json_names = extract_names_from_json(raw)
        if json_names:
            return _deduplicate_preserve_order(json_names)

    # Fall back to line extraction
    line_names = extract_names_from_lines(raw)
    if line_names:
        return _deduplicate_preserve_order(line_names)

    # Last resort: try JSON anyway even if not starting with correct char
    json_names = extract_names_from_json(raw)
    return _deduplicate_preserve_order(json_names)


def _deduplicate_preserve_order(names: List[str]) -> List[str]:
    seen, out = set(), []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Step 64: levenshtein_distance — pure Python dynamic programming
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # One-row DP to stay O(min(m,n)) space
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Step 65: fuzzy_match_known + deduplicate_names
# ---------------------------------------------------------------------------

def fuzzy_match_known(
    candidate: str,
    known_list: List[str],
    threshold: float = 0.75,
) -> Optional[Tuple[str, float]]:
    """
    Find the closest match in known_list to candidate using normalised edit distance.
    Returns (matched_name, similarity) or None if no match above threshold.
    """
    if not known_list:
        return None
    best_name, best_sim = None, 0.0
    c_lower = candidate.lower()
    for name in known_list:
        n_lower = name.lower()
        dist    = levenshtein_distance(c_lower, n_lower)
        max_len = max(len(c_lower), len(n_lower))
        if max_len == 0:
            continue
        sim = 1.0 - dist / max_len
        if sim > best_sim:
            best_sim  = sim
            best_name = name
    if best_sim >= threshold and best_name:
        return (best_name, best_sim)
    return None


def deduplicate_names(names: List[str], threshold: float = 0.80) -> List[str]:
    """
    Remove near-duplicate names based on normalised edit distance.
    Keeps the first occurrence of each cluster.
    """
    result: List[str] = []
    for candidate in names:
        if not candidate:
            continue
        is_dup = False
        for existing in result:
            dist    = levenshtein_distance(candidate.lower(), existing.lower())
            max_len = max(len(candidate), len(existing))
            sim     = 1.0 - dist / max_len if max_len else 1.0
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(candidate)
    return result
