"""
People Gallery — persistent, multi-session visual memory of VRChat players.
============================================================================
Steps 2–30 of the 100-step recognition plan.

Why this exists: the existing people.json only stores text metadata. This module
adds a visual layer — reference crops, per-person folders, unknown-person tracking,
fuzzy name search, and a full promote-unknown-to-known state machine.

Directory layout created automatically:
  <gallery_dir>/
    gallery.json                ← master index (atomic write-rename)
    known/
      <SanitisedName>/
        ref_000.png             ← reference crops (up to MAX_REF_CROPS each)
        ref_001.png
        ...
    unknown/
      unk_<epoch10>/
        crop_000.png            ← crops from each sighting (up to MAX_SIGHT_CROPS)
        crop_001.png
        ...

Thread safety: every write path acquires _lock. Reads are unlocked for speed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REF_CROPS       = 10   # reference images kept per known person
MAX_SIGHT_CROPS     = 5    # crops kept per unknown-person sighting cluster
MAX_UNKNOWN_PERSONS = 100  # total unknown persons tracked simultaneously
UNKNOWN_EXPIRY_DAYS = 7    # unknown persons not seen within this window are evicted
MAX_SIGHTINGS_LOG   = 100  # sightings stored per known person
MAX_AVATAR_KEYWORDS = 60   # visual keywords per person


# ---------------------------------------------------------------------------
# Step 3: GalleryPerson dataclass
# ---------------------------------------------------------------------------

@dataclass
class GalleryPerson:
    """Full visual+metadata record for a confirmed named VRChat player."""
    name:             str
    first_seen:       str                      # YYYY-MM-DD
    last_seen:        str                      # YYYY-MM-DD
    sighting_count:   int       = 0
    ref_crop_paths:   List[str] = field(default_factory=list)
    avatar_keywords:  List[str] = field(default_factory=list)  # visual desc words
    worlds_met:       List[str] = field(default_factory=list)
    sightings:        List['PersonSighting'] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['sightings'] = [s.to_dict() for s in self.sightings]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'GalleryPerson':
        sightings = [PersonSighting.from_dict(s) for s in d.get('sightings', [])]
        return cls(
            name           = d['name'],
            first_seen     = d.get('first_seen', ''),
            last_seen      = d.get('last_seen', ''),
            sighting_count = d.get('sighting_count', 0),
            ref_crop_paths = d.get('ref_crop_paths', []),
            avatar_keywords= d.get('avatar_keywords', []),
            worlds_met     = d.get('worlds_met', []),
            sightings      = sightings,
        )


# ---------------------------------------------------------------------------
# Step 4: PersonSighting dataclass
# ---------------------------------------------------------------------------

@dataclass
class PersonSighting:
    """A single timestamped sighting of a known person."""
    timestamp:  str
    world:      str  = ''
    crop_path:  str  = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'PersonSighting':
        return cls(
            timestamp = d.get('timestamp', ''),
            world     = d.get('world', ''),
            crop_path = d.get('crop_path', ''),
        )


# ---------------------------------------------------------------------------
# Step 5: UnknownPerson dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnknownPerson:
    """An as-yet-unidentified player tracked by visual similarity."""
    uid:                  str
    first_seen:           str        # ISO 8601 datetime string
    last_seen:            str
    sighting_count:       int  = 0
    crop_paths:           List[str] = field(default_factory=list)
    avatar_desc:          str  = ''  # last known avatar description
    resolution_attempts:  int  = 0   # how many recognition pipeline runs
    resolved_as:          str  = ''  # non-empty once name is confirmed
    next_retry_after:     float = 0.0  # epoch seconds (backoff scheduling)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'UnknownPerson':
        return cls(
            uid                = d['uid'],
            first_seen         = d.get('first_seen', ''),
            last_seen          = d.get('last_seen', ''),
            sighting_count     = d.get('sighting_count', 0),
            crop_paths         = d.get('crop_paths', []),
            avatar_desc        = d.get('avatar_desc', ''),
            resolution_attempts= d.get('resolution_attempts', 0),
            resolved_as        = d.get('resolved_as', ''),
            next_retry_after   = float(d.get('next_retry_after', 0.0)),
        )


# ---------------------------------------------------------------------------
# Step 6: GalleryStats dataclass
# ---------------------------------------------------------------------------

@dataclass
class GalleryStats:
    """Snapshot of gallery metrics."""
    total_known:          int
    total_unknown:        int
    total_resolved:       int
    total_ref_crops:      int
    resolution_rate:      float     # resolved / (unknown + resolved)
    oldest_unknown_days:  float
    most_seen_person:     str
    total_sightings:      int

    def summary(self) -> str:
        return (
            f"Known={self.total_known}  Unknown={self.total_unknown}  "
            f"Resolved={self.total_resolved}  Rate={self.resolution_rate:.0%}  "
            f"Crops={self.total_ref_crops}  Sightings={self.total_sightings}  "
            f"TopPerson={self.most_seen_person}"
        )


# ---------------------------------------------------------------------------
# Steps 7–28: PeopleGallery class
# ---------------------------------------------------------------------------

class PeopleGallery:
    """
    Persistent visual memory of VRChat players across sessions.

    Thread-safe. All writes acquire self._lock.
    Call save() periodically and at shutdown.
    """

    def __init__(self, gallery_dir: str = 'snapshots/gallery') -> None:
        # Step 7: init dirs
        self.gallery_dir  = gallery_dir
        self.known_dir    = os.path.join(gallery_dir, 'known')
        self.unknown_dir  = os.path.join(gallery_dir, 'unknown')
        self.index_path   = os.path.join(gallery_dir, 'gallery.json')

        os.makedirs(self.known_dir,   exist_ok=True)
        os.makedirs(self.unknown_dir, exist_ok=True)

        self._lock:    threading.Lock              = threading.Lock()
        self._known:   Dict[str, GalleryPerson]   = {}
        self._unknown: Dict[str, UnknownPerson]   = {}

        self._load()

    # -----------------------------------------------------------------------
    # Step 8: Load from disk
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data.get('known', []):
                try:
                    p = GalleryPerson.from_dict(item)
                    self._known[p.name] = p
                except Exception:
                    pass
            for item in data.get('unknown', []):
                try:
                    u = UnknownPerson.from_dict(item)
                    self._unknown[u.uid] = u
                except Exception:
                    pass
            print(
                f'[gallery] loaded {len(self._known)} known, '
                f'{len(self._unknown)} unknown persons',
                flush=True,
            )
        except Exception as e:
            print(f'[gallery] load error: {e}', flush=True)

    # -----------------------------------------------------------------------
    # Step 9: Save — atomic write-then-rename
    # -----------------------------------------------------------------------

    def save(self) -> None:
        with self._lock:
            data = {
                'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'known':    [p.to_dict() for p in self._known.values()],
                'unknown':  [u.to_dict() for u in self._unknown.values()],
            }
        tmp_path = self.index_path + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.index_path)
        except Exception as e:
            print(f'[gallery] save error: {e}', flush=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # -----------------------------------------------------------------------
    # Step 10: add_known_person — upsert with full metadata tracking
    # -----------------------------------------------------------------------

    def add_known_person(
        self,
        name:        str,
        world:       str = '',
        crop_path:   str = '',
        avatar_desc: str = '',
    ) -> bool:
        """Upsert a known person. Returns True if brand new."""
        with self._lock:
            today  = time.strftime('%Y-%m-%d')
            is_new = name not in self._known

            if is_new:
                self._known[name] = GalleryPerson(
                    name       = name,
                    first_seen = today,
                    last_seen  = today,
                )
                os.makedirs(self._person_dir(name), exist_ok=True)

            person = self._known[name]
            person.last_seen       = today
            person.sighting_count += 1

            if world and world not in person.worlds_met:
                person.worlds_met.append(world)
                if len(person.worlds_met) > 10:
                    person.worlds_met.pop(0)

            if avatar_desc:
                for kw in re.findall(r'[a-z]{3,}', avatar_desc.lower()):
                    if kw not in person.avatar_keywords:
                        person.avatar_keywords.append(kw)
                if len(person.avatar_keywords) > MAX_AVATAR_KEYWORDS:
                    person.avatar_keywords = person.avatar_keywords[-MAX_AVATAR_KEYWORDS:]

            if crop_path and os.path.exists(crop_path):
                self._add_ref_crop_internal(person, crop_path)

            return is_new

    # -----------------------------------------------------------------------
    # Step 11: get_person
    # -----------------------------------------------------------------------

    def get_person(self, name: str) -> Optional[GalleryPerson]:
        return self._known.get(name)

    # -----------------------------------------------------------------------
    # Step 12: all_known_names
    # -----------------------------------------------------------------------

    def all_known_names(self) -> List[str]:
        return sorted(self._known.keys())

    # -----------------------------------------------------------------------
    # Step 13: search_by_name_fuzzy — Jaro-Winkler
    # -----------------------------------------------------------------------

    def search_by_name_fuzzy(
        self, query: str, threshold: float = 0.70
    ) -> List[Tuple[str, float]]:
        """Return [(name, similarity)] sorted descending, above threshold."""
        q = query.lower().strip()
        results: List[Tuple[str, float]] = []
        for name in self._known:
            sim = _jaro_winkler(q, name.lower())
            if sim >= threshold:
                results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Step 14: _add_ref_crop_internal (must hold lock)
    # -----------------------------------------------------------------------

    def _add_ref_crop_internal(self, person: GalleryPerson, src_path: str) -> None:
        person_dir = self._person_dir(person.name)
        os.makedirs(person_dir, exist_ok=True)
        idx = len(person.ref_crop_paths)
        dst = os.path.join(person_dir, f'ref_{idx:03d}.png')
        try:
            if os.path.abspath(src_path) != os.path.abspath(dst):
                shutil.copy2(src_path, dst)
            if dst not in person.ref_crop_paths:
                person.ref_crop_paths.append(dst)
            while len(person.ref_crop_paths) > MAX_REF_CROPS:
                old = person.ref_crop_paths.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass
        except Exception as e:
            print(f'[gallery] crop copy error ({person.name}): {e}', flush=True)

    # -----------------------------------------------------------------------
    # Step 15: add_reference_crop — public API
    # -----------------------------------------------------------------------

    def add_reference_crop(self, name: str, crop_path: str) -> None:
        with self._lock:
            if name not in self._known or not os.path.exists(crop_path):
                return
            self._add_ref_crop_internal(self._known[name], crop_path)

    # -----------------------------------------------------------------------
    # Step 16: get_best_reference
    # -----------------------------------------------------------------------

    def get_best_reference(self, name: str) -> Optional[str]:
        person = self._known.get(name)
        if not person:
            return None
        for path in reversed(person.ref_crop_paths):
            if os.path.exists(path):
                return path
        return None

    # -----------------------------------------------------------------------
    # Step 17: add_sighting
    # -----------------------------------------------------------------------

    def add_sighting(
        self, name: str, world: str = '', crop_path: str = ''
    ) -> None:
        with self._lock:
            person = self._known.get(name)
            if not person:
                return
            person.sightings.append(PersonSighting(
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S'),
                world     = world,
                crop_path = crop_path,
            ))
            if len(person.sightings) > MAX_SIGHTINGS_LOG:
                person.sightings.pop(0)

    # -----------------------------------------------------------------------
    # Step 18: add_unknown_person — auto-ID with similarity dedup
    # -----------------------------------------------------------------------

    def add_unknown_person(
        self, avatar_desc: str = '', crop_path: str = ''
    ) -> str:
        """Register or update an unknown person. Returns their UID."""
        with self._lock:
            now_str = time.strftime('%Y-%m-%d %H:%M:%S')

            # Step 19: check for existing similar unknown (Jaccard on desc words)
            if avatar_desc:
                existing = self._find_similar_unknown_internal(avatar_desc)
                if existing:
                    u = self._unknown[existing]
                    u.sighting_count += 1
                    u.last_seen       = now_str
                    u.avatar_desc     = avatar_desc  # update desc
                    if crop_path and os.path.exists(crop_path):
                        unk_dir = os.path.join(self.unknown_dir, existing)
                        os.makedirs(unk_dir, exist_ok=True)
                        idx = len(u.crop_paths)
                        dst = os.path.join(unk_dir, f'crop_{idx:03d}.png')
                        try:
                            shutil.copy2(crop_path, dst)
                            u.crop_paths.append(dst)
                            if len(u.crop_paths) > MAX_SIGHT_CROPS:
                                old = u.crop_paths.pop(0)
                                try:
                                    os.remove(old)
                                except OSError:
                                    pass
                        except Exception:
                            pass
                    return existing

            # New unknown — assign a unique timestamp-based UID
            uid = f'unk_{int(time.time()):010d}'
            while uid in self._unknown:
                uid += '_'

            unk_dir = os.path.join(self.unknown_dir, uid)
            os.makedirs(unk_dir, exist_ok=True)

            crop_paths: List[str] = []
            if crop_path and os.path.exists(crop_path):
                dst = os.path.join(unk_dir, 'crop_000.png')
                try:
                    shutil.copy2(crop_path, dst)
                    crop_paths = [dst]
                except Exception:
                    crop_paths = [crop_path]

            self._unknown[uid] = UnknownPerson(
                uid            = uid,
                first_seen     = now_str,
                last_seen      = now_str,
                sighting_count = 1,
                crop_paths     = crop_paths,
                avatar_desc    = avatar_desc,
            )

            # Step 20: cap total unknown persons — evict oldest if over limit
            if len(self._unknown) > MAX_UNKNOWN_PERSONS:
                self._evict_oldest_unknown()

            return uid

    # -----------------------------------------------------------------------
    # Step 19: _find_similar_unknown_internal (Jaccard on avatar_desc words)
    # -----------------------------------------------------------------------

    def _find_similar_unknown_internal(self, avatar_desc: str) -> Optional[str]:
        """Find existing unresolved unknown whose desc overlaps this one. Call with lock."""
        desc_words = set(re.findall(r'[a-z]{3,}', avatar_desc.lower()))
        if not desc_words:
            return None
        best_uid, best_score = None, 0.0
        for uid, u in self._unknown.items():
            if u.resolved_as:
                continue
            stored = set(re.findall(r'[a-z]{3,}', u.avatar_desc.lower()))
            if not stored:
                continue
            score = len(desc_words & stored) / len(desc_words | stored)
            if score > best_score:
                best_score, best_uid = score, uid
        return best_uid if best_score >= 0.45 else None

    # -----------------------------------------------------------------------
    # Step 20: _evict_oldest_unknown
    # -----------------------------------------------------------------------

    def _evict_oldest_unknown(self) -> None:
        unresolved = {uid: u for uid, u in self._unknown.items() if not u.resolved_as}
        if not unresolved:
            return
        oldest = min(unresolved, key=lambda uid: unresolved[uid].first_seen)
        self._unknown.pop(oldest, None)
        evict_dir = os.path.join(self.unknown_dir, oldest)
        shutil.rmtree(evict_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Step 21: get_unknown_persons
    # -----------------------------------------------------------------------

    def get_unknown_persons(
        self, include_resolved: bool = False
    ) -> List[UnknownPerson]:
        with self._lock:
            result = list(self._unknown.values())
        if not include_resolved:
            result = [u for u in result if not u.resolved_as]
        result.sort(key=lambda u: u.last_seen, reverse=True)
        return result

    # -----------------------------------------------------------------------
    # Step 22: promote_unknown_to_known — full state machine
    # -----------------------------------------------------------------------

    def promote_unknown_to_known(
        self,
        uid:          str,
        resolved_name: str,
        world:        str = '',
        avatar_desc:  str = '',
    ) -> bool:
        """Mark uid as resolved; migrate crops to known person's folder.
        Returns True if the known person is brand new.
        """
        with self._lock:
            u = self._unknown.get(uid)
            if not u:
                return False

            u.resolved_as = resolved_name

            # Find best crop from this unknown's history
            best_crop = next(
                (cp for cp in reversed(u.crop_paths) if os.path.exists(cp)),
                '',
            )

            today  = time.strftime('%Y-%m-%d')
            is_new = resolved_name not in self._known

            if is_new:
                self._known[resolved_name] = GalleryPerson(
                    name       = resolved_name,
                    first_seen = today,
                    last_seen  = today,
                )
                os.makedirs(self._person_dir(resolved_name), exist_ok=True)

            person = self._known[resolved_name]
            person.last_seen        = today
            person.sighting_count  += u.sighting_count

            if world and world not in person.worlds_met:
                person.worlds_met.append(world)

            if avatar_desc:
                for kw in re.findall(r'[a-z]{3,}', avatar_desc.lower()):
                    if kw not in person.avatar_keywords:
                        person.avatar_keywords.append(kw)

            if best_crop:
                self._add_ref_crop_internal(person, best_crop)

            print(
                f'[gallery] promoted {uid} → "{resolved_name}" '
                f'({u.sighting_count} sightings migrated)',
                flush=True,
            )
            return is_new

    # -----------------------------------------------------------------------
    # Step 23: expire_old_unknowns — 7-day TTL
    # -----------------------------------------------------------------------

    def expire_old_unknowns(self) -> int:
        cutoff = time.time() - UNKNOWN_EXPIRY_DAYS * 86400
        to_remove: List[str] = []
        with self._lock:
            for uid, u in self._unknown.items():
                if u.resolved_as:
                    continue
                try:
                    last_epoch = time.mktime(
                        time.strptime(u.last_seen, '%Y-%m-%d %H:%M:%S')
                    )
                    if last_epoch < cutoff:
                        to_remove.append(uid)
                except ValueError:
                    pass
            for uid in to_remove:
                self._unknown.pop(uid, None)
                shutil.rmtree(os.path.join(self.unknown_dir, uid), ignore_errors=True)
        if to_remove:
            print(f'[gallery] expired {len(to_remove)} old unknown persons', flush=True)
        return len(to_remove)

    # -----------------------------------------------------------------------
    # Step 24: cleanup
    # -----------------------------------------------------------------------

    def cleanup(self) -> None:
        self.expire_old_unknowns()
        self.save()

    # -----------------------------------------------------------------------
    # Step 25: stats → GalleryStats
    # -----------------------------------------------------------------------

    def stats(self) -> GalleryStats:
        with self._lock:
            known   = list(self._known.values())
            unknown = list(self._unknown.values())

        total_known    = len(known)
        total_unknown  = sum(1 for u in unknown if not u.resolved_as)
        total_resolved = sum(1 for u in unknown if u.resolved_as)
        total_attempts = total_unknown + total_resolved
        rate           = total_resolved / total_attempts if total_attempts > 0 else 0.0
        total_crops    = sum(len(p.ref_crop_paths) for p in known)
        total_sightings= sum(p.sighting_count for p in known)
        most_seen      = max(known, key=lambda p: p.sighting_count, default=None)

        oldest_days = 0.0
        now = time.time()
        for u in unknown:
            if not u.resolved_as:
                try:
                    ts = time.mktime(time.strptime(u.first_seen, '%Y-%m-%d %H:%M:%S'))
                    days = (now - ts) / 86400
                    if days > oldest_days:
                        oldest_days = days
                except ValueError:
                    pass

        return GalleryStats(
            total_known         = total_known,
            total_unknown       = total_unknown,
            total_resolved      = total_resolved,
            total_ref_crops     = total_crops,
            resolution_rate     = rate,
            oldest_unknown_days = oldest_days,
            most_seen_person    = most_seen.name if most_seen else '—',
            total_sightings     = total_sightings,
        )

    # -----------------------------------------------------------------------
    # Step 26: context_for_names — compact block for think model
    # -----------------------------------------------------------------------

    def context_for_names(self, names: List[str]) -> str:
        lines: List[str] = []
        for name in names:
            p = self._known.get(name)
            if not p:
                continue
            worlds = ', '.join(p.worlds_met[-3:]) if p.worlds_met else 'unknown world'
            kws    = ' '.join(p.avatar_keywords[:8])
            ref_ct = len(p.ref_crop_paths)
            line   = (
                f'{name} — seen {p.sighting_count}× across '
                f'{len(p.worlds_met)} world(s) | last: {p.last_seen} | '
                f'{worlds}'
            )
            if kws:
                line += f' | avatar looks: {kws}'
            if ref_ct:
                line += f' | {ref_ct} reference image(s) stored'
            lines.append(line)
        if not lines:
            return ''
        return 'VISUAL GALLERY:\n' + '\n'.join(f'  📷 {ln}' for ln in lines)

    # -----------------------------------------------------------------------
    # Step 27: get_unresolved_crops — feed the resolver queue on startup
    # -----------------------------------------------------------------------

    def get_unresolved_crops(self) -> List[Tuple[str, str]]:
        """Return [(uid, best_crop_path)] for all unresolved unknowns with crops."""
        result: List[Tuple[str, str]] = []
        with self._lock:
            unknowns = list(self._unknown.values())
        for u in unknowns:
            if u.resolved_as:
                continue
            for cp in reversed(u.crop_paths):
                if os.path.exists(cp):
                    result.append((u.uid, cp))
                    break
        return result

    # -----------------------------------------------------------------------
    # Step 28: rebuild_index — disaster recovery
    # -----------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """Re-scan the known/ and unknown/ directories to rebuild the index."""
        print('[gallery] rebuilding index from disk…', flush=True)
        with self._lock:
            # Scan known/
            for person_dir_name in os.listdir(self.known_dir):
                full = os.path.join(self.known_dir, person_dir_name)
                if not os.path.isdir(full):
                    continue
                # Collect existing ref pngs
                ref_pngs = sorted(
                    os.path.join(full, f) for f in os.listdir(full)
                    if f.startswith('ref_') and f.endswith('.png')
                )
                if ref_pngs and person_dir_name not in self._known:
                    today = time.strftime('%Y-%m-%d')
                    p = GalleryPerson(
                        name           = person_dir_name,
                        first_seen     = today,
                        last_seen      = today,
                        ref_crop_paths = ref_pngs,
                    )
                    self._known[person_dir_name] = p
                    print(f'  [gallery/reindex] found known: {person_dir_name}', flush=True)

            # Scan unknown/
            for uid_dir_name in os.listdir(self.unknown_dir):
                full = os.path.join(self.unknown_dir, uid_dir_name)
                if not os.path.isdir(full):
                    continue
                crop_pngs = sorted(
                    os.path.join(full, f) for f in os.listdir(full)
                    if f.startswith('crop_') and f.endswith('.png')
                )
                if uid_dir_name not in self._unknown:
                    now_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    u = UnknownPerson(
                        uid        = uid_dir_name,
                        first_seen = now_str,
                        last_seen  = now_str,
                        crop_paths = crop_pngs,
                    )
                    self._unknown[uid_dir_name] = u
                    print(f'  [gallery/reindex] found unknown: {uid_dir_name}', flush=True)

        print(
            f'[gallery] rebuild complete: {len(self._known)} known, '
            f'{len(self._unknown)} unknown',
            flush=True,
        )
        self.save()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _person_dir(self, name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:60]
        return os.path.join(self.known_dir, safe)

    def get_recent_resolutions(self, limit: int = 10) -> List[dict]:
        """Return the most recently resolved unknown persons (for dashboard)."""
        with self._lock:
            resolved = [u for u in self._unknown.values() if u.resolved_as]
        resolved.sort(key=lambda u: u.last_seen, reverse=True)
        return [
            {
                'uid':         u.uid,
                'resolved_as': u.resolved_as,
                'sightings':   u.sighting_count,
                'attempts':    u.resolution_attempts,
                'last_seen':   u.last_seen,
            }
            for u in resolved[:limit]
        ]


# ---------------------------------------------------------------------------
# Step 29: _jaro_winkler — pure Python, zero dependencies
# ---------------------------------------------------------------------------

def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler string similarity (0.0 = totally different, 1.0 = identical)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    match_dist = max(0, match_dist)

    s1_match = [False] * len1
    s2_match = [False] * len2
    matches = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(len2, i + match_dist + 1)
        for j in range(lo, hi):
            if s2_match[j] or s1[i] != s2[j]:
                continue
            s1_match[i] = True
            s2_match[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_match[i]:
            continue
        while not s2_match[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 +
        matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3.0

    # Winkler prefix bonus (up to 4 chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * p * (1.0 - jaro)
