"""
Prompt Library — Steps 66–77 of the 100-step recognition plan.
================================================================
Ten carefully crafted prompts for extracting VRChat nametag text from
cropped images. Each prompt uses a different strategy to coax the
vision model into reading the text accurately.

Reading nametag text from VRChat frames is hard because:
  - Tags are often at the edge of legibility (motion blur, low res)
  - Multiple players may overlap
  - Lighting / HDR effects desaturate or oversaturate text
  - The model may default to describing the scene rather than OCR'ing it

The prompts below progressively escalate from gentle to forceful,
from structured to free-text, from abstract to hyper-specific.
The recognition pipeline will try them in order and stop when one succeeds.

Usage:
    prompts = get_all_prompts(known_names=['Alice', 'Bob'])
    for prompt in prompts:
        response = vision_model.query(image, prompt)
        names = extract_names_from_raw(response)
        if names:
            break
"""

from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Step 67: PROMPT_JSON_STRUCTURED
# ---------------------------------------------------------------------------

PROMPT_JSON_STRUCTURED: str = (
    'Look at this image and find every floating nametag above a player avatar. '
    'Nametags in VRChat appear as text floating above a character\'s head, '
    'usually white or colored, and show the player\'s username. '
    'Read each nametag carefully, character by character. '
    'Respond ONLY with a JSON object in exactly this format: '
    '{"names": ["Name1", "Name2"]}  '
    'If you find no readable nametags respond with {"names": []}. '
    'Do not include any other text, description, or explanation. '
    'Output only the JSON object.'
)


# ---------------------------------------------------------------------------
# Step 68: PROMPT_PLAIN_TEXT
# ---------------------------------------------------------------------------

PROMPT_PLAIN_TEXT: str = (
    'List every VRChat player nametag visible in this image. '
    'Nametags are username labels that float above player avatars. '
    'They are usually displayed in white or bright text. '
    'Write one username per line. Nothing else. '
    'If no nametag is visible, write: none'
)


# ---------------------------------------------------------------------------
# Step 69: PROMPT_OCR_ENGINE
# ---------------------------------------------------------------------------

PROMPT_OCR_ENGINE: str = (
    'You are an OCR engine specialized in reading text from screenshots. '
    'Your task: locate and transcribe every piece of text floating above '
    'avatar heads in this VRChat screenshot crop. '
    'These texts are player usernames. '
    'Focus on bright/white floating text near the top of the image. '
    'Return ONLY the extracted text strings, one per line. '
    'Preserve capitalization and special characters exactly as they appear. '
    'Include underscores, hyphens, dots if they are part of the name. '
    'Do not describe shapes, colors, or anything else.'
)


# ---------------------------------------------------------------------------
# Step 70: PROMPT_CHAR_BY_CHAR
# ---------------------------------------------------------------------------

PROMPT_CHAR_BY_CHAR: str = (
    'Read the characters in this image one by one. '
    'Focus on the text floating near the top of the image. '
    'In VRChat these are username nametags. '
    'Go letter by letter, digit by digit, symbol by symbol. '
    'Pay attention to uppercase vs lowercase. '
    'Output each complete username you find on its own line. '
    'Nothing else — just the raw text you see.'
)


# ---------------------------------------------------------------------------
# Step 71: PROMPT_FLOATING_WHITE_TEXT
# ---------------------------------------------------------------------------

PROMPT_FLOATING_WHITE_TEXT: str = (
    'What white, light-colored, or bright text appears floating '
    'in the upper half of this image? '
    'In VRChat games, player names float above avatars as white or colored '
    'text labels. Look specifically for that floating text. '
    'Read each name carefully. '
    'List the names you find, one per line.'
)


# ---------------------------------------------------------------------------
# Step 72: PROMPT_PARTIAL_OK
# ---------------------------------------------------------------------------

PROMPT_PARTIAL_OK: str = (
    'Examine this image for VRChat player username tags. '
    'Even if only part of a username is visible or the image is blurry, '
    'write down whatever partial text you can make out. '
    'Partial text is acceptable — do not skip a name just because it is cut off. '
    'Format: one username (or partial username) per line. '
    'If truly nothing can be read, write: unreadable'
)


# ---------------------------------------------------------------------------
# Step 73: PROMPT_TOP_REGION
# ---------------------------------------------------------------------------

PROMPT_TOP_REGION: str = (
    'Ignore everything in the lower portion of this image. '
    'Focus exclusively on the very top section. '
    'In that top section, identify any floating text labels. '
    'These are VRChat player usernames. '
    'Transcribe them exactly character by character. '
    'Output format: one name per line, no other text.'
)


# ---------------------------------------------------------------------------
# Step 74: PROMPT_MULTI_PLAYER
# ---------------------------------------------------------------------------

PROMPT_MULTI_PLAYER: str = (
    'There may be multiple players in this image, each with a floating '
    'nametag username above their head. '
    'Find ALL of them. Check left side, center, and right side of the image. '
    'Read every nametag you can see. '
    'Return them as a JSON array: {"names": ["Alice", "Bob", "Carol"]} '
    'If none are visible return {"names": []}.'
)


# ---------------------------------------------------------------------------
# Step 75: PROMPT_DESPERATE
# ---------------------------------------------------------------------------

PROMPT_DESPERATE: str = (
    'CRITICAL TASK: Read the floating username text in this image. '
    'This is real-time VRChat social software. Accurately identifying '
    'player names is essential. '
    'The usernames are bright text floating above avatar heads. '
    'Use every capability you have to read them. '
    'Look at edges, contrast patterns, pixel clusters near the image top. '
    'Even if the text is blurry, make your best determination. '
    'A wrong answer is better than no answer. '
    'Respond with ONLY the names you see, one per line. '
    'First name on the first line, second on the second line, etc.'
)


# ---------------------------------------------------------------------------
# Step 76: make_hint_prompt — injects known names for context
# ---------------------------------------------------------------------------

def make_hint_prompt(known_names: List[str]) -> str:
    """
    Build a prompt that hints at known names the model might recognise,
    reducing the search space and boosting accuracy.
    """
    if not known_names:
        # Fall back to a plain prompt if we have no hints
        return PROMPT_PLAIN_TEXT

    # Limit to 15 names to avoid prompt bloat
    hints = known_names[:15]
    hints_str = ', '.join(f'"{n}"' for n in hints)

    return (
        'You are reading VRChat nametags. '
        f'You have previously seen these players: {hints_str}. '
        'If you see any of these names (or something close to them) in '
        'the image, write that name. You may also see new names — write '
        'those too. '
        'Output each username you find, one per line. '
        'Be exact about spelling and capitalisation.'
    )


# ---------------------------------------------------------------------------
# Step 77: get_all_prompts — returns the full ordered prompt list
# ---------------------------------------------------------------------------

def get_all_prompts(known_names: Optional[List[str]] = None) -> List[str]:
    """
    Return all prompts in the recommended trial order.
    The pipeline should try them in order and stop at the first success.
    JSON-structured prompts are tried first because they give the most
    machine-parseable output. Free-text prompts act as fallbacks.
    If known_names is provided, a hint prompt is appended near the end.
    """
    prompts: List[str] = [
        PROMPT_JSON_STRUCTURED,      # 0 — structured JSON, easiest to parse
        PROMPT_OCR_ENGINE,           # 1 — OCR mode
        PROMPT_PLAIN_TEXT,           # 2 — simple directive
        PROMPT_CHAR_BY_CHAR,         # 3 — character-level reading
        PROMPT_FLOATING_WHITE_TEXT,  # 4 — color/spatial hint
        PROMPT_TOP_REGION,           # 5 — spatial focus
        PROMPT_MULTI_PLAYER,         # 6 — multi-player JSON
        PROMPT_PARTIAL_OK,           # 7 — accept partial
        PROMPT_DESPERATE,            # 8 — last-resort maximum effort
    ]

    # Insert the hint prompt second-to-last if we have known names
    if known_names:
        hint = make_hint_prompt(known_names)
        prompts.insert(-1, hint)   # before PROMPT_DESPERATE

    return prompts
