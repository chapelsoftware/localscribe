"""Build caption tracks from word-level Whisper timestamps.

Whisper emits per-word `(start, end, text)` triples for each segment;
we re-group those into short rolling phrases (Canva / Submagic style)
and emit an ASS subtitle file that ffmpeg's libass filter can burn into
the rendered clip.

The grouping rules favor readability:
- Hard break at strong punctuation (. ! ?) -- captions never bridge
  sentence boundaries.
- Soft break at clause punctuation (, ; :) once a phrase is reasonably
  full.
- Break if the gap to the next word exceeds `gap_break_s` seconds
  (a natural pause).
- Cap each phrase at `max_words` words and `max_duration_s` seconds.

Output `Phrase` objects are ALWAYS expressed in clip-relative time
(0 = start of the clip), so they can be burned directly into a freshly
trimmed mp4.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Phrase:
    start: float  # seconds, clip-relative
    end: float    # seconds, clip-relative
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Word loading + filtering
# ---------------------------------------------------------------------------

def _load_words(transcript_raw: Path) -> list[dict]:
    """Flatten `transcript.raw.json` into a single list of word dicts.

    Each word has `start`, `end`, `word` (with leading space stripped),
    and `prob`.
    """
    data = json.loads(transcript_raw.read_text())
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if not text:
                continue
            words.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "text": text,
                "prob": float(w.get("prob", 1.0)),
            })
    return words


def words_in_range(words: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Return words whose midpoint falls inside [start_s, end_s].

    Using the midpoint (rather than start) avoids dropping a word whose
    start is slightly before the cut but is mostly within the clip.
    """
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if start_s <= mid <= end_s:
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Phrase grouping
# ---------------------------------------------------------------------------

# Punctuation that always ends a phrase
_HARD_PUNCT = re.compile(r"[.!?…]+[\"')\]]*\s*$")
# Punctuation that ends a phrase if it's already long-ish
_SOFT_PUNCT = re.compile(r"[,;:][\"')\]]*\s*$")


def group_phrases(
    words: list[dict],
    *,
    clip_start: float = 0.0,
    max_words: int = 3,
    max_duration_s: float = 2.2,
    gap_break_s: float = 0.45,
    soft_min_words: int = 2,
) -> list[Phrase]:
    """Group consecutive Whisper words into rolling phrases.

    All `Phrase.start` / `.end` are translated to clip-relative time
    (subtracting `clip_start`), so the result can drive a subtitle file
    that overlays the trimmed clip.
    """
    if not words:
        return []

    phrases: list[Phrase] = []
    bucket: list[dict] = []

    def flush() -> None:
        if not bucket:
            return
        text = " ".join(w["text"] for w in bucket)
        phrases.append(Phrase(
            start=max(0.0, bucket[0]["start"] - clip_start),
            end=max(0.0, bucket[-1]["end"] - clip_start),
            text=text,
        ))
        bucket.clear()

    for i, w in enumerate(words):
        bucket.append(w)
        last_text = w["text"]
        cur_dur = bucket[-1]["end"] - bucket[0]["start"]
        next_gap = (
            words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 0.0
        )

        hard_punct = bool(_HARD_PUNCT.search(last_text))
        soft_punct = bool(_SOFT_PUNCT.search(last_text))
        long_gap = next_gap >= gap_break_s
        too_long = len(bucket) >= max_words or cur_dur >= max_duration_s

        # Hard break: sentence-ending punctuation
        if hard_punct:
            flush()
            continue
        # Soft break: clause punctuation, but only if we have enough text
        if soft_punct and len(bucket) >= soft_min_words:
            flush()
            continue
        # Pause break
        if long_gap and len(bucket) >= soft_min_words:
            flush()
            continue
        # Length cap
        if too_long:
            flush()
            continue

    flush()
    return phrases


def phrases_for_clip(transcript_raw: Path, start_s: float, end_s: float,
                     **group_opts) -> list[Phrase]:
    """Convenience: load + filter + group in one call."""
    words = _load_words(transcript_raw)
    in_range = words_in_range(words, start_s, end_s)
    return group_phrases(in_range, clip_start=start_s, **group_opts)


# ---------------------------------------------------------------------------
# ASS rendering
# ---------------------------------------------------------------------------

ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {playres_x}
PlayResY: {playres_y}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H00{accent_bgr},&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,{outline},{shadow},2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(t: float) -> str:
    """Format seconds as ASS timestamp `H:MM:SS.cc` (centiseconds)."""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs >= 100:  # rounding can push us to 100
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_bgr(rgb_hex: str) -> str:
    """Convert '#RRGGBB' -> 'BBGGRR' (ASS uses BGR)."""
    s = rgb_hex.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #RRGGBB, got {rgb_hex!r}")
    r, g, b = s[0:2], s[2:4], s[4:6]
    return (b + g + r).upper()


def _escape_ass_text(text: str) -> str:
    """ASS dialogue text needs braces escaped (override-tag delimiters)."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def render_ass(
    phrases: list[Phrase],
    *,
    playres_x: int = 1080,
    playres_y: int = 1920,
    font: str = "DejaVu Sans",
    font_size: int = 78,
    accent_color: str = "#fbb53c",  # gold (matches mailer)
    outline: int = 4,
    shadow: int = 1,
    margin_v: int = 320,  # distance from bottom in PlayRes units
    fade_ms: int = 80,
) -> str:
    """Build an ASS subtitle file as a string.

    The default style is bold sans, white text, thick black outline,
    soft drop shadow, slight fade in/out, positioned in the lower third.
    """
    accent_bgr = _hex_bgr(accent_color)
    header = ASS_HEADER_TEMPLATE.format(
        playres_x=playres_x,
        playres_y=playres_y,
        font=font,
        font_size=font_size,
        accent_bgr=accent_bgr,
        outline=outline,
        shadow=shadow,
        margin_v=margin_v,
    )

    lines = []
    for ph in phrases:
        # Slight fade in/out keeps captions from snapping
        prefix = f"{{\\fad({fade_ms},{fade_ms})}}"
        text = prefix + _escape_ass_text(ph.text)
        lines.append(
            f"Dialogue: 0,{_ts(ph.start)},{_ts(ph.end)},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(lines) + "\n"


def write_ass(phrases: list[Phrase], out_path: Path, **render_opts) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_ass(phrases, **render_opts), encoding="utf-8")
    return out_path
