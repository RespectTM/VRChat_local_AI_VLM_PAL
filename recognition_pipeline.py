"""
Recognition Pipeline — Steps 78–91 of the 100-step recognition plan.
======================================================================
Drives a multi-stage vision-model pipeline to resolve VRChat nametag text
from cropped images.

Architecture
------------
NameRecognizer.recognize(frame_path)
  │
  ├─ build_all_variants(frame_path)          # 14+ crop × enhancement variants
  │     └─ CropVariant[]  (quality-sorted)
  │
  ├─ for each CropVariant:
  │     _process_variant(variant, known_names)
  │       └─ _query_until_success(crop_path, known_names)
  │             ├─ prompt[0]: PROMPT_JSON_STRUCTURED   → extract + validate
  │             ├─ prompt[1]: PROMPT_OCR_ENGINE         → extract + validate
  │             ├─ ...
  │             └─ prompt[N]: PROMPT_DESPERATE         → extract + validate
  │
  ├─ _vote_on_candidates(all_attempts)        # majority vote across variants
  │
  └─ RecognitionResult(names, confidence, ...)

Confidence scoring
------------------
  - +0.35 for each additional variant that agrees
  - +0.20 if result matches a known gallery name (fuzzy)
  - +0.10 for each prompt that succeeded (capped at 0.20)
  - −0.05 for each disagreement across variants
  - Capped to [0.0, 1.0]

Early exit
----------
If a high-confidence result (>= 0.80) is found within the first 3 variants,
skip remaining variants to save time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import image_enhance
import name_validator
import prompt_library


# ---------------------------------------------------------------------------
# Step 79: RecognitionAttempt dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecognitionAttempt:
    """One attempt: single crop variant + single prompt."""
    variant_name:    str          # e.g. 'nametag_strip_4x'
    prompt_index:    int          # index into get_all_prompts()
    raw_response:    str          # raw model text
    extracted_names: List[str]    # validated names parsed from raw_response
    confidence:      float = 0.0  # per-attempt confidence (0.0 – 1.0)
    success:         bool  = False # True if at least one valid name extracted
    elapsed_secs:    float = 0.0  # how long the query took


# ---------------------------------------------------------------------------
# Step 80: RecognitionResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    """Final aggregated result of the full recognition pipeline."""
    names:               List[str]             # final deduplicated name list
    confidence:          float                 # 0.0 – 1.0
    winning_variant:     str    = ''           # crop strategy that gave the result
    winning_prompt_idx:  int    = -1           # prompt index that first succeeded
    total_attempts:      int    = 0            # total model queries made
    total_variants:      int    = 0            # number of crop variants processed
    all_attempts:        List[RecognitionAttempt] = field(default_factory=list)
    elapsed_secs:        float  = 0.0
    source_path:         str    = ''

    @property
    def success(self) -> bool:
        return bool(self.names) and self.confidence > 0.0

    def summary(self) -> str:
        if not self.names:
            return (f'[pipeline] No names found '
                    f'(variants={self.total_variants}, '
                    f'attempts={self.total_attempts}, '
                    f'elapsed={self.elapsed_secs:.1f}s)')
        return (f'[pipeline] Found: {self.names!r} '
                f'conf={self.confidence:.2f} '
                f'via {self.winning_variant!r}[p{self.winning_prompt_idx}] '
                f'attempts={self.total_attempts} '
                f'elapsed={self.elapsed_secs:.1f}s')


# ---------------------------------------------------------------------------
# Step 81: PipelineMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    """Accumulated statistics across all recognition attempts this session."""
    total_frames:   int   = 0
    total_success:  int   = 0
    total_failed:   int   = 0
    total_queries:  int   = 0
    total_secs:     float = 0.0
    # Distribution of which crop variant / prompt index succeeded most
    variant_wins:   Dict[str, int] = field(default_factory=dict)
    prompt_wins:    Dict[int,  int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.total_success + self.total_failed
        return self.total_success / total if total else 0.0

    @property
    def avg_queries_per_frame(self) -> float:
        return self.total_queries / max(self.total_frames, 1)

    def summary_dict(self) -> dict:
        return {
            'total_frames':       self.total_frames,
            'total_success':      self.total_success,
            'total_failed':       self.total_failed,
            'success_rate':       round(self.success_rate, 3),
            'avg_queries':        round(self.avg_queries_per_frame, 1),
            'top_variant':        _top_key(self.variant_wins),
            'top_prompt':         _top_key(self.prompt_wins),
        }


def _top_key(d: dict) -> str:
    if not d:
        return '—'
    return str(max(d, key=d.get))


# ---------------------------------------------------------------------------
# Step 82–83: NameRecognizer class
# ---------------------------------------------------------------------------

# Type alias for the vision model callable
VisionFn = Callable[[str, str], str]     # (image_path, prompt) → response_text


class NameRecognizer:
    """
    Multi-strategy nametag recognition engine.

    Parameters
    ----------
    vision_fn : callable
        Function that accepts (image_path: str, prompt: str) → str.
        Should call the local Ollama vision model (minicpm-v:8b).
    known_names : list, optional
        List of known player names used to generate hint prompts and
        for fuzzy-matching confirmation.
    scale_factor : int
        Upscale factor for all crop variants (default 4).
    max_prompts : int
        Maximum number of prompts to try per variant before giving up (default 10).
    confidence_threshold : float
        Minimum confidence to emit a result (default 0.55).
    early_exit_threshold : float
        If confidence >= this after ≤3 variants, skip remaining (default 0.80).
    save_variants_dir : str
        If non-empty, save all generated crop images here for debugging.
    """

    def __init__(
        self,
        vision_fn:            VisionFn,
        known_names:          Optional[List[str]] = None,
        scale_factor:         int   = 4,
        max_prompts:          int   = 10,
        confidence_threshold: float = 0.55,
        early_exit_threshold: float = 0.80,
        save_variants_dir:    str   = '',
    ) -> None:
        self._vision_fn            = vision_fn
        self._known_names          = list(known_names or [])
        self._scale_factor         = scale_factor
        self._max_prompts          = max_prompts
        self._confidence_threshold = confidence_threshold
        self._early_exit_threshold = early_exit_threshold
        self._save_variants_dir    = save_variants_dir
        self._metrics              = PipelineMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_known_names(self, names: List[str]) -> None:
        self._known_names = list(names)

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = PipelineMetrics()

    # ------------------------------------------------------------------
    # Step 84: recognize — main pipeline entry point
    # ------------------------------------------------------------------

    def recognize(
        self,
        frame_path:  str,
        extra_known: Optional[List[str]] = None,
    ) -> RecognitionResult:
        """
        Run the full recognition pipeline against a single frame.

        Strategy:
          1. Build 14+ crop variants (quality-sorted)
          2. Try each variant with increasingly desperate prompts
          3. Aggregate results via majority vote
          4. Compute confidence score
          5. Return RecognitionResult
        """
        t_start      = time.time()
        all_attempts: List[RecognitionAttempt] = []

        known = list(self._known_names)
        if extra_known:
            known = list({*known, *extra_known})

        # Build all enhanced crop variants
        variants = image_enhance.build_all_variants(
            frame_path,
            scale_factor = self._scale_factor,
            save_dir     = self._save_variants_dir,
        )

        if not variants:
            return RecognitionResult(
                names=[], confidence=0.0, source_path=frame_path,
                elapsed_secs=time.time() - t_start,
            )

        # Try each variant; allow early exit if confidence is already high
        for variant in variants:
            if not variant.saved_path and not os.path.exists(str(getattr(variant, 'saved_path', ''))):
                # Save temporarily to a named temp path for vision model
                tmp_path = f'{frame_path}_variant_{variant.name}.png'
                try:
                    variant.image.save(tmp_path)
                    variant.saved_path = tmp_path
                except Exception:
                    continue

            attempt = self._process_variant(variant, known)
            if attempt:
                all_attempts.append(attempt)

            # Check for early exit
            if self._should_stop_early(all_attempts, len(variants)):
                break

        # Aggregate results
        final_names, confidence = self._vote_on_candidates(all_attempts, known)

        # Build result
        winning_attempt = next(
            (a for a in all_attempts if a.success and a.extracted_names == final_names),
            next((a for a in all_attempts if a.success), None),
        )

        elapsed = time.time() - t_start

        result = RecognitionResult(
            names              = final_names,
            confidence         = confidence,
            winning_variant    = winning_attempt.variant_name if winning_attempt else '',
            winning_prompt_idx = winning_attempt.prompt_index if winning_attempt else -1,
            total_attempts     = len(all_attempts),
            total_variants     = len(variants),
            all_attempts       = all_attempts,
            elapsed_secs       = elapsed,
            source_path        = frame_path,
        )

        # Update metrics
        self._metrics.total_frames += 1
        self._metrics.total_queries += len(all_attempts)
        self._metrics.total_secs    += elapsed
        if result.success:
            self._metrics.total_success += 1
            if result.winning_variant:
                self._metrics.variant_wins[result.winning_variant] = (
                    self._metrics.variant_wins.get(result.winning_variant, 0) + 1
                )
            if result.winning_prompt_idx >= 0:
                self._metrics.prompt_wins[result.winning_prompt_idx] = (
                    self._metrics.prompt_wins.get(result.winning_prompt_idx, 0) + 1
                )
        else:
            self._metrics.total_failed += 1

        # Clean up temp files
        self._cleanup_temp_variants(variants)

        print(result.summary(), flush=True)
        return result

    # ------------------------------------------------------------------
    # Step 85: _process_variant — try one crop variant with all prompts
    # ------------------------------------------------------------------

    def _process_variant(
        self,
        variant:     image_enhance.CropVariant,
        known_names: List[str],
    ) -> Optional[RecognitionAttempt]:
        """
        Try prompts against a single crop variant until one succeeds.
        Returns the successful (or best-effort) RecognitionAttempt.
        """
        path = variant.saved_path
        if not path or not os.path.exists(path):
            return None

        names, prompt_idx, raw_resp, elapsed = self._query_until_success(
            path, known_names
        )
        confidence = self._score_attempt(names, variant)

        return RecognitionAttempt(
            variant_name    = variant.name,
            prompt_index    = prompt_idx,
            raw_response    = raw_resp,
            extracted_names = names,
            confidence      = confidence,
            success         = bool(names),
            elapsed_secs    = elapsed,
        )

    # ------------------------------------------------------------------
    # Step 86: _query_until_success — iterate prompts until extraction works
    # ------------------------------------------------------------------

    def _query_until_success(
        self,
        crop_path:   str,
        known_names: List[str],
    ) -> Tuple[List[str], int, str, float]:
        """
        Try each prompt in order until we extract at least one valid name.
        Returns (names, prompt_index, raw_response, elapsed_secs).
        Falls back to ([], last_prompt_index, last_raw, elapsed) if none succeed.
        """
        prompts       = prompt_library.get_all_prompts(known_names or None)
        prompts       = prompts[:self._max_prompts]
        last_raw      = ''
        last_idx      = 0
        t_start       = time.time()

        for idx, prompt in enumerate(prompts):
            last_idx = idx
            try:
                raw = self._vision_fn(crop_path, prompt)
            except Exception as e:
                print(f'[pipeline] vision query error (idx={idx}): {e}', flush=True)
                continue

            last_raw = raw
            names    = name_validator.extract_names_from_raw(raw)
            names    = name_validator.deduplicate_names(names)

            if names:
                elapsed = time.time() - t_start
                return (names, idx, raw, elapsed)

        elapsed = time.time() - t_start
        return ([], last_idx, last_raw, elapsed)

    # ------------------------------------------------------------------
    # Step 87: _vote_on_candidates — majority vote across all attempts
    # ------------------------------------------------------------------

    def _vote_on_candidates(
        self,
        attempts:    List[RecognitionAttempt],
        known_names: List[str],
    ) -> Tuple[List[str], float]:
        """
        Aggregate all extracted names across all variants via weighted vote.

        Scoring per candidate name:
          - +1.0  for each attempt that contains this name
          - +0.5  if the name fuzzy-matches a known player (gallery confirmation)
          - −0.3  for each attempt that differs (disagreement penalty)
        Final confidence = score of winner / (total successful attempts or 1)
        capped to [0.0, 1.0].
        """
        successful = [a for a in attempts if a.success]
        if not successful:
            return ([], 0.0)

        # Tally votes across all successful attempts
        vote_tally: Dict[str, float] = {}
        for attempt in successful:
            for name in attempt.extracted_names:
                key = name.lower()
                vote_tally[key] = vote_tally.get(key, 0.0) + 1.0

        if not vote_tally:
            return ([], 0.0)

        # Apply gallery bonus
        for key in list(vote_tally.keys()):
            match = name_validator.fuzzy_match_known(key, [n.lower() for n in known_names], 0.80)
            if match:
                vote_tally[key] += 0.5

        # Find best candidates (names with more than 0.25 votes)
        threshold_votes = max(0.25, len(successful) * 0.25)
        winners = {k: v for k, v in vote_tally.items() if v >= threshold_votes}

        if not winners:
            # Fall back: take whatever got the most votes, even if just 1
            best_key = max(vote_tally, key=vote_tally.get)
            winners  = {best_key: vote_tally[best_key]}

        # Map back to original casing (take from first attempt that had it)
        key_to_display: Dict[str, str] = {}
        for attempt in successful:
            for name in attempt.extracted_names:
                key = name.lower()
                if key in winners and key not in key_to_display:
                    key_to_display[key] = name

        final_names = [key_to_display[k] for k in winners if k in key_to_display]
        final_names = name_validator.deduplicate_names(final_names)

        # Compute final confidence
        if not final_names:
            return ([], 0.0)

        best_votes = max(winners.values())
        confidence = self._score_result(final_names, attempts, best_votes)

        return (final_names, confidence)

    # ------------------------------------------------------------------
    # Step 88: _score_result — multi-factor confidence calculation
    # ------------------------------------------------------------------

    def _score_attempt(
        self,
        names:   List[str],
        variant: image_enhance.CropVariant,
    ) -> float:
        """Quick per-attempt confidence based on quality score + name presence."""
        if not names:
            return 0.0
        # Base confidence from image quality (max 0.5)
        base = variant.quality_score * 0.5
        # Bonus for finding names (up to 0.5)
        name_bonus = min(0.5, len(names) * 0.2)
        return min(1.0, base + name_bonus)

    def _score_result(
        self,
        names:    List[str],
        attempts: List[RecognitionAttempt],
        votes:    float,
    ) -> float:
        """
        Multi-factor confidence:
          - Agreement ratio (how many attempts agreed)
          - Prompt index penalty (later prompts = less confident)
          - Name quality (plausible username structure)
          - Number of names found (low = more confident)
        """
        successful   = [a for a in attempts if a.success]
        total        = max(len(successful), 1)
        agree_ratio  = min(1.0, votes / total)

        # Penalty for using late prompts (index > 4 = desperate)
        best_prompt  = min((a.prompt_index for a in successful), default=9)
        prompt_score = max(0.3, 1.0 - best_prompt * 0.07)

        # Name structure bonus — VRChat names tend to be 3-20 chars
        name_score  = 1.0
        for name in names:
            if len(name) < 3 or len(name) > 25:
                name_score *= 0.8

        # Fewer names = more confident (1 name = simple case)
        count_score = 1.0 if len(names) <= 2 else max(0.5, 1.0 - (len(names) - 2) * 0.1)

        confidence = (
            agree_ratio  * 0.45 +
            prompt_score * 0.25 +
            name_score   * 0.15 +
            count_score  * 0.15
        )
        return min(1.0, max(0.0, confidence))

    # ------------------------------------------------------------------
    # Step 89: _should_stop_early — fast-exit if already confident
    # ------------------------------------------------------------------

    def _should_stop_early(
        self,
        attempts:      List[RecognitionAttempt],
        total_variants: int,
    ) -> bool:
        """
        Return True if we should stop processing more variants.
        Conditions:
          - We have ≥ 3 successful attempts AND
          - All of them agree on the same name AND
          - Their confidence average is above early_exit_threshold
        """
        successful = [a for a in attempts if a.success]
        if len(successful) < 3:
            return False

        # Check if all recent successful attempts agree
        # Use the last 3 successes
        recent = successful[-3:]
        all_names = [frozenset(a.extracted_names) for a in recent]
        if len(set(all_names)) > 1:
            return False   # disagreement

        avg_conf = sum(a.confidence for a in recent) / len(recent)
        return avg_conf >= self._early_exit_threshold

    # ------------------------------------------------------------------
    # Step 90: metrics property (already defined above) + reset
    # ------------------------------------------------------------------

    # (metrics property defined inside __init__ block — see above)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _cleanup_temp_variants(self, variants: List[image_enhance.CropVariant]) -> None:
        """Remove temporary per-frame variant PNGs created during recognition."""
        for v in variants:
            if (v.saved_path
                    and '_variant_' in v.saved_path
                    and os.path.exists(v.saved_path)):
                try:
                    os.remove(v.saved_path)
                except OSError:
                    pass
