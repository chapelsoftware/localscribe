"""Search the word-level transcript for phrase boundaries.

Lets you define a clip by what was SAID rather than by hunting for
timestamps in the transcript yourself. The CLI's `cut --start-phrase
"..."` and `cut-batch` both flow through `resolve_boundaries` here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _load_words(transcript_raw: Path) -> list[dict]:
    """Flatten transcript.raw.json into a list of `{start, end, word}`."""
    data = json.loads(transcript_raw.read_text())
    words: list[dict] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            text = w.get("word", "").strip()
            if not text:
                continue
            words.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": text,
            })
    return words


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Matters because Whisper renders contractions inconsistently
    ("we're" vs "we are") and embeds punctuation in word tokens
    ("world."). Normalizing both sides lets a phrase match regardless.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def find_phrase(
    words: list[dict],
    phrase: str,
    start_idx: int = 0,
    max_window: int = 35,
) -> tuple[float, float, int] | None:
    """Search for `phrase` as the LEADING content of a window of
    consecutive words starting at `words[i]` for some `i >= start_idx`.

    The phrase must align with the start of the window (so a phrase that
    appears mid-window doesn't anchor `word_start` to an earlier,
    unrelated word). Window size grows up to `max_window` to handle
    transcript punctuation that the normalizer collapses but Whisper's
    tokenization keeps as separate words.

    Returns `(word_start, word_end, idx_after_match)` for the FIRST
    aligned match — `idx_after_match` is the index right past the
    match, ready to feed back as `start_idx` for a subsequent search.

    Returns None if the phrase isn't found.
    """
    target = _normalize(phrase)
    if not target:
        return None
    target_words = target.split()
    n = len(target_words)
    for i in range(start_idx, len(words)):
        # Quick reject: if words[i] alone doesn't begin with the
        # phrase's first word, no window starting here can align.
        first = _normalize(words[i]["word"]).split()
        if not first or not target_words[0].startswith(first[0]) and not first[0].startswith(target_words[0]):
            continue
        for span in range(n, max_window + 1):
            if i + span > len(words):
                break
            joined = _normalize(" ".join(w["word"] for w in words[i:i+span]))
            if joined.startswith(target):
                return (words[i]["start"], words[i+span-1]["end"], i + span)
    return None


def resolve_boundaries(
    transcript_raw: Path,
    start_phrase: str,
    end_phrase: str,
    *,
    head_pad: float = 0.1,
    tail_pad: float = 0.3,
) -> tuple[float, float]:
    """Find clip boundaries from a pair of phrases.

    Returns `(start_s, end_s)` in source seconds. `head_pad` is
    subtracted from the matched start (catching the first word's onset)
    and `tail_pad` is added to the matched end (preserving trailing
    consonants). Snap-on-cut will tighten further.

    Raises `ValueError` if either phrase isn't found.
    """
    words = _load_words(transcript_raw)
    sm = find_phrase(words, start_phrase)
    if sm is None:
        raise ValueError(f"start phrase not found: {start_phrase!r}")
    s_start, _, s_idx_after = sm
    em = find_phrase(words, end_phrase, start_idx=s_idx_after - 1)
    if em is None:
        raise ValueError(f"end phrase not found: {end_phrase!r}")
    _, e_end, _ = em
    if e_end <= s_start:
        raise ValueError(
            f"end phrase resolves before start: "
            f"{start_phrase!r}->{s_start:.2f}, {end_phrase!r}->{e_end:.2f}"
        )
    return max(0.0, s_start - head_pad), e_end + tail_pad
