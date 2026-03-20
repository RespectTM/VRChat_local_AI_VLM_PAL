"""
Nametag Resolution Engine — Steps 92–99 of the 100-step recognition plan.
==========================================================================
Complete rewrite using the full recognition stack:
  image_enhance → prompt_library → name_validator
  → recognition_pipeline → people_gallery.

Public interface (backward compatible with main.py):
  resolver = NametegResolver(vision_fn, snapshots_dir)
  resolver.start()
  resolver.stop()
  resolver.enqueue_frame(frame_path)        # queue a raw frame for analysis
  resolver.save_known_person(name, frame)   # save reference crop for known player
  resolver.resolved_q                       # Queue[ResolvedName] — drain each cycle

What's new vs the original:
  • Uses all 14+ crop variants (not just a single top-strip crop)
  • Tries all 10+ prompts per variant (not just 3 prompts)
  • Integrates with PeopleGallery — unknowns auto-added, resolved → promoted
  • Cross-frame tracking — same unknown sighted multiple times updates their record
  • Smart retry queue with exponential backoff (1s → 2s → 4s → 8s → 32s)
  • Per-frame deduplication — same frame won't be queued twice within 5 minutes
  • Variant quality filtering — skips low-quality crops to save time
  • Thread-safe, daemon thread, zero impact on main loop latency
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Import new recognition stack (Steps 92–96)
# ---------------------------------------------------------------------------

import image_enhance        # crop strategies + enhancement
import name_validator       # is_valid_name, extract_names_from_raw, deduplicate
import prompt_library       # get_all_prompts, PROMPT_* constants
import people_gallery       # PeopleGallery, GalleryPerson, UnknownPerson
import recognition_pipeline # NameRecognizer, RecognitionResult

# ---------------------------------------------------------------------------
# Constants — tunable via config (Step 97)
# ---------------------------------------------------------------------------

MAX_QUEUE_SIZE        = 12      # max frames queued at once
MAX_RETRY_ATTEMPTS    = 5       # max retry rounds per frame before giving up
BACKOFF_SECS          = [1, 2, 4, 8, 32]   # exponential backoff between retries
FRAME_DEDUP_WINDOW    = 300     # seconds — same frame won't be re-queued in this window
MIN_VARIANT_QUALITY   = 0.15    # skip variants below this quality score

# Placeholder patterns — names that must not be saved (same as main.py)
_PLACEHOLDER_RE = re.compile(
    r'^(?:unknown|player\d*|user\d*|person|avatar|nametag|name\s*tag|'
    r'n/?a|none|unnamed|unreadable|unclear|visible|private|someone|\?+)$',
    re.IGNORECASE,
)


def _is_real_name(s: str) -> bool:
    """Thin wrapper kept for back-compat with main.py callers."""
    return name_validator.is_valid_name(str(s or '').strip())


# ---------------------------------------------------------------------------
# ResolvedName result type (preserved from original for backward compat)
# ---------------------------------------------------------------------------

class ResolvedName(NamedTuple):
    name:         str    # the resolved username
    crop_id:      str    # UID of the unknown person or '' for fast-resolve
    frame_path:   str    # original frame that contained the unknown player
    prompt_index: int    # which prompt index finally worked (-1 if multi-attempt)
    confidence:   float = 0.0    # 0.0–1.0 recognition confidence


# ---------------------------------------------------------------------------
# RetryEntry — tracks per-frame retry state
# ---------------------------------------------------------------------------

@dataclass
class RetryEntry:
    """One frame waiting for re-attempt after a failed resolution."""
    frame_path:    str
    crop_id:       str = ''
    attempt:       int = 0
    next_retry_at: float = 0.0
    avatar_desc:   str = ''


# ---------------------------------------------------------------------------
# NametegResolver (Step 93 — preserves original class name for main.py compat)
# ---------------------------------------------------------------------------

class NametegResolver:
    """
    Background thread that resolves unknown nametags from saved frame crops.

    Architecture
    ------------
    Main thread                  Resolver thread
    ──────────────               ──────────────────────────────────────────
    enqueue_frame()  ─────────►  _run() loop
                                   │
                                   ├─ NameRecognizer.recognize()
                                   │     ├─ build_all_variants()  (14+ crops)
                                   │     ├─ get_all_prompts()     (10+ prompts)
                                   │     └─ vote + score
                                   │
                                   ├─ if success →
                                   │     gallery.promote_unknown_to_known()
                                   │     resolved_q.put(ResolvedName)
                                   │
                                   └─ if fail → retry_list with backoff
    """

    def __init__(
        self,
        vision_fn:            Callable,
        snapshots_dir:        str   = 'snapshots',
        gallery:              Optional[people_gallery.PeopleGallery] = None,
        scale_factor:         int   = 4,
        max_prompts:          int   = 10,
        confidence_threshold: float = 0.55,
        nametag_crops_dir:    str   = '',
    ) -> None:
        self._vision_fn            = vision_fn
        self._snapshots_dir        = snapshots_dir
        self._scale_factor         = scale_factor
        self._max_prompts          = max_prompts
        self._confidence_threshold = confidence_threshold
        self._nametag_crops_dir    = nametag_crops_dir

        # Directories
        self._unknown_dir  = os.path.join(snapshots_dir, 'unknown_names')
        self._known_dir    = os.path.join(snapshots_dir, 'known_people')
        os.makedirs(self._unknown_dir, exist_ok=True)
        os.makedirs(self._known_dir,   exist_ok=True)
        if nametag_crops_dir:
            os.makedirs(nametag_crops_dir, exist_ok=True)

        # Gallery integration (Step 96)
        self._gallery = gallery

        # Build the recognition engine
        self._recognizer = recognition_pipeline.NameRecognizer(
            vision_fn            = self._call_vision,
            scale_factor         = scale_factor,
            max_prompts          = max_prompts,
            confidence_threshold = confidence_threshold,
            save_variants_dir    = '',
        )

        # Queues
        self._queue:     queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.resolved_q: queue.Queue = queue.Queue()   # ResolvedName for main loop

        # Retry list — frames that need back-off retry (Step 98)
        self._retry_list: List[RetryEntry] = []
        self._retry_lock  = threading.Lock()

        # Per-frame dedup window (Step 96)
        self._recent_frames: dict = {}   # frame_path → last_queued timestamp

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None

        print(f'[nametag] resolver initialised '
              f'(scale={scale_factor}x, max_prompts={max_prompts}, '
              f'crops→{nametag_crops_dir or "(disabled)"})', flush=True)

    # ------------------------------------------------------------------
    # Vision function adapter (Step 94)
    # ------------------------------------------------------------------

    def _call_vision(self, image_path: str, prompt: str) -> str:
        """
        Adapter: bridges between recognition_pipeline's
        (image_path, prompt) → str
        and main.py's vision_fn(prompt, [image_path]) → str convention.
        """
        try:
            return self._vision_fn(prompt, [image_path])
        except TypeError:
            try:
                return self._vision_fn(image_path, prompt)
            except Exception as e:
                print(f'[nametag] _call_vision error: {e}', flush=True)
                return ''

    # ------------------------------------------------------------------
    # Thread lifecycle (Step 95)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name='nametag-resolver'
        )
        self._thread.start()
        print('[nametag] resolver thread started', flush=True)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._gallery:
            try:
                self._gallery.save()
            except Exception:
                pass
        print('[nametag] resolver thread stopped', flush=True)

    # ------------------------------------------------------------------
    # Public API — enqueue_frame (Step 96)
    # ------------------------------------------------------------------

    def enqueue_frame(
        self,
        frame_path:  str,
        avatar_desc: str = '',
    ) -> None:
        """
        Queue a frame for nametag resolution. Non-blocking; drops if full.
        Same frame path won't be queued again within FRAME_DEDUP_WINDOW seconds.
        """
        now = time.time()
        last = self._recent_frames.get(frame_path, 0)
        if now - last < FRAME_DEDUP_WINDOW:
            return

        self._recent_frames[frame_path] = now
        if len(self._recent_frames) > 500:
            cutoff = now - FRAME_DEDUP_WINDOW
            self._recent_frames = {k: v for k, v in self._recent_frames.items()
                                   if v > cutoff}
        try:
            self._queue.put_nowait((frame_path, avatar_desc))
        except queue.Full:
            pass  # silently drop — resolver is busy

    # ------------------------------------------------------------------
    # Public API — save_known_person (Step 97)
    # ------------------------------------------------------------------

    def save_known_person(self, name: str, frame_path: str) -> None:
        """
        Save a high-quality enhanced crop for a confirmed named player.
        Runs in a separate daemon thread to avoid blocking the main loop.
        Also registers the person in the gallery if available.
        """
        threading.Thread(
            target=self._save_known_crop_background,
            args=(name, frame_path),
            daemon=True,
        ).start()

    def _save_known_crop_background(self, name: str, frame_path: str) -> None:
        try:
            variants = image_enhance.build_all_variants(
                frame_path, scale_factor=self._scale_factor
            )
            if not variants:
                return
            best = image_enhance.choose_best_variant(variants)
            if not best:
                return
            safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:60]
            dst = os.path.join(self._known_dir, f'{safe_name}.png')
            best.image.save(dst)
            if self._gallery:
                self._gallery.add_known_person(name=name, crop_path=dst)
        except Exception as e:
            print(f'[nametag] save_known_crop error for {name!r}: {e}', flush=True)

    # ------------------------------------------------------------------
    # Background worker loop (Step 98)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                frame_path, avatar_desc = self._queue.get(timeout=0.5)
                self._resolve(frame_path, avatar_desc)
            except queue.Empty:
                pass
            except Exception as e:
                print(f'[nametag] worker error: {e}', flush=True)
            self._service_retries()
        print('[nametag] worker loop ended', flush=True)

    def _service_retries(self) -> None:
        """Process any RetryEntry items whose backoff time has elapsed."""
        now = time.time()
        with self._retry_lock:
            due = [e for e in self._retry_list if e.next_retry_at <= now]
            for e in due:
                self._retry_list.remove(e)
        for entry in due:
            try:
                self._resolve(entry.frame_path, entry.avatar_desc, retry_entry=entry)
            except Exception as e:
                print(f'[nametag] retry error: {e}', flush=True)

    # ------------------------------------------------------------------
    # Core resolution logic (Step 99)
    # ------------------------------------------------------------------

    def _resolve(
        self,
        frame_path:  str,
        avatar_desc: str = '',
        retry_entry: Optional[RetryEntry] = None,
    ) -> None:
        """
        Core resolution:
          1. Register unknown in gallery with initial crop snapshot
          2. Run NameRecognizer.recognize() — 14+ variant × 10+ prompt pipeline
          3. If success → promote unknown → known, emit to resolved_q
          4. If failure → schedule exponential backoff retry
        """
        attempt_num = retry_entry.attempt if retry_entry else 0
        print(
            f'[nametag] resolving {os.path.basename(frame_path)} '
            f'(attempt {attempt_num + 1}/{MAX_RETRY_ATTEMPTS})',
            flush=True,
        )

        # Register unknown in gallery
        crop_id = retry_entry.crop_id if retry_entry else ''
        if self._gallery and not crop_id:
            raw_crop = self._save_raw_nametag_crop(frame_path)
            crop_id  = self._gallery.add_unknown_person(
                avatar_desc=avatar_desc, crop_path=raw_crop
            )
            print(f'[nametag] registered {crop_id}', flush=True)
        elif not crop_id:
            crop_id = f'unk_{int(time.time())}'

        known_names = self._gallery.all_known_names() if self._gallery else []

        # Run recognition pipeline
        result = self._recognizer.recognize(
            frame_path=frame_path, extra_known=known_names
        )

        if result.success and result.confidence >= self._confidence_threshold:
            print(
                f'[nametag] ✔ resolved {result.names!r} '
                f'conf={result.confidence:.2f}',
                flush=True,
            )
            for name in result.names:
                if self._gallery and crop_id:
                    try:
                        self._gallery.promote_unknown_to_known(
                            uid=crop_id, resolved_name=name,
                            avatar_desc=avatar_desc
                        )
                    except Exception as e:
                        print(f'[nametag] gallery promote error: {e}', flush=True)
                # Save resolved crop to nametag_crops_dir for player analysis
                if self._nametag_crops_dir:
                    try:
                        variants = image_enhance.build_all_variants(
                            frame_path, scale_factor=self._scale_factor
                        )
                        best = image_enhance.choose_best_variant(variants)
                        if best:
                            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:60]
                            ts_str = time.strftime('%Y%m%d_%H%M%S')
                            dst = os.path.join(
                                self._nametag_crops_dir,
                                f'{safe}__{ts_str}.png'
                            )
                            best.image.save(dst)
                            print(f'[nametag] 📸 saved crop → {dst}', flush=True)
                    except Exception as e:
                        print(f'[nametag] crop save error for {name!r}: {e}', flush=True)
                self.resolved_q.put(ResolvedName(
                    name=name, crop_id=crop_id,
                    frame_path=frame_path,
                    prompt_index=result.winning_prompt_idx,
                    confidence=result.confidence,
                ))
            if self._gallery:
                try:
                    self._gallery.save()
                except Exception:
                    pass
        else:
            next_attempt = attempt_num + 1
            if next_attempt < MAX_RETRY_ATTEMPTS:
                delay = BACKOFF_SECS[min(next_attempt, len(BACKOFF_SECS) - 1)]
                print(
                    f'[nametag] ✗ no name '
                    f'(conf={result.confidence:.2f}, retry in {delay}s)',
                    flush=True,
                )
                entry = RetryEntry(
                    frame_path=frame_path, crop_id=crop_id,
                    attempt=next_attempt,
                    next_retry_at=time.time() + delay,
                    avatar_desc=avatar_desc,
                )
                with self._retry_lock:
                    self._retry_list.append(entry)
            else:
                print(
                    f'[nametag] ✗ gave up on '
                    f'{os.path.basename(frame_path)} after '
                    f'{MAX_RETRY_ATTEMPTS} attempts',
                    flush=True,
                )

    def _save_raw_nametag_crop(self, frame_path: str) -> str:
        """Quick nametag-strip crop for gallery record-keeping. Returns saved path."""
        try:
            from PIL import Image, ImageFilter, ImageEnhance as IE
            img  = Image.open(frame_path).convert('RGB')
            w, h = img.size
            crop = img.crop((0, 0, w, int(h * 0.55)))
            crop = crop.filter(ImageFilter.SHARPEN)
            crop = IE.Contrast(crop).enhance(1.4)
            uid  = f'unk_{int(time.time() * 1000)}'
            dst  = os.path.join(self._unknown_dir, f'{uid}.png')
            crop.save(dst)
            return dst
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return a snapshot of the recognizer's performance metrics."""
        return self._recognizer.metrics.summary_dict()

    def get_pending_count(self) -> int:
        """Number of frames queued + waiting in retry list."""
        return self._queue.qsize() + len(self._retry_list)
