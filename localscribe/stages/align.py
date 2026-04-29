"""Stage 4: Align whisper words with pyannote speaker turns.

For each transcribed word, find the speaker whose turn contains the word's
midpoint. Words that fall outside all turns get the nearest speaker. Then
coalesce consecutive same-speaker words back into speaker segments.
"""
from __future__ import annotations

import bisect
import logging

from ..config import Paths, cached, load_json, save_json

log = logging.getLogger("align")


def _speaker_for_time(turns: list[dict], starts: list[float], t: float) -> str:
    """Find the speaker turn containing time t, or the closest one."""
    idx = bisect.bisect_right(starts, t) - 1
    # Check the candidate and its neighbor in case t falls in a gap
    best = None
    best_dist = float("inf")
    for i in (idx, idx + 1):
        if 0 <= i < len(turns):
            turn = turns[i]
            if turn["start"] <= t <= turn["end"]:
                return turn["speaker"]
            dist = min(abs(t - turn["start"]), abs(t - turn["end"]))
            if dist < best_dist:
                best_dist = dist
                best = turn["speaker"]
    return best or "SPEAKER_00"


def run(paths: Paths, force: bool = False) -> list[dict]:
    if cached(paths.transcript_aligned, force):
        log.info("cached")
        return load_json(paths.transcript_aligned)

    transcript = load_json(paths.transcript_raw)
    turns = load_json(paths.diarization)
    if not turns:
        raise RuntimeError("Empty diarization -- nothing to align")

    turns_sorted = sorted(turns, key=lambda t: t["start"])
    starts = [t["start"] for t in turns_sorted]

    # Walk every word, assign speaker
    aligned_words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if w["start"] is None or w["end"] is None:
                continue
            mid = (w["start"] + w["end"]) / 2
            speaker = _speaker_for_time(turns_sorted, starts, mid)
            aligned_words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"],
                "speaker": speaker,
            })

    # Coalesce consecutive same-speaker words into segments.
    # Start a new segment whenever the speaker changes OR there's a gap > 1.5s.
    GAP_THRESHOLD = 1.5
    segments: list[dict] = []
    for w in aligned_words:
        if (
            segments
            and segments[-1]["speaker"] == w["speaker"]
            and w["start"] - segments[-1]["end"] <= GAP_THRESHOLD
        ):
            segments[-1]["end"] = w["end"]
            segments[-1]["text"] += w["word"]
        else:
            segments.append({
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"],
            })

    # Clean up leading spaces from whisper word tokens
    for s in segments:
        s["text"] = s["text"].strip()

    save_json(paths.transcript_aligned, segments)
    log.info("ok: %d speaker-coalesced segments", len(segments))
    return segments
