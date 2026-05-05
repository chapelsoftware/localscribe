"""Snap clip boundaries to the actual content.

Two complementary signals tell us where a clip "really" starts and
ends:

1. Caption boundaries — the auto-generated captions are derived from
   the word-level transcript, so the FIRST caption's start and the
   LAST caption's end mark the speaker's first and last meaningful
   word in the clip. If the user's curated boundaries include
   pre-content speech ("And so what we say is...") or post-content
   speech ("...and that's the point. Now moving on..."), snapping to
   captions trims that warmup/coast.

2. Audio RMS — for clips with no captions, or as a refinement on top
   of caption-based snapping, the actual speech onset / offset can be
   found by probing audio energy.

`snap_to_content` uses captions as the primary signal and falls back
to RMS when no captions are available. RMS-only `snap_to_silence` is
also exposed for the rare case where you want to snap a clip whose
captions you don't trust (e.g. before manual caption editing).
"""
from __future__ import annotations

import logging
import struct
import subprocess
from pathlib import Path

log = logging.getLogger("clips.silence")

_SR = 16000  # 16 kHz mono PCM is plenty for energy detection
_HOP_S = 0.010  # 10 ms hop = 160 samples at 16 kHz
_WIN_S = 0.030  # 30 ms RMS window


def _extract_pcm(
    src_video: Path, t0: float, t1: float,
) -> bytes:
    """Decode `[t0, t1]` of `src_video` to mono 16 kHz s16le PCM."""
    if t1 <= t0:
        return b""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin",
            "-ss", f"{max(0.0, t0):.3f}",
            "-to", f"{t1:.3f}",
            "-i", str(src_video),
            "-vn",
            "-ac", "1",
            "-ar", str(_SR),
            "-f", "s16le", "-",
        ],
        capture_output=True, check=False, timeout=60,
    )
    return proc.stdout if proc.returncode == 0 else b""


def _rms_envelope(pcm: bytes) -> list[float]:
    """RMS energy at fixed-hop frames over `pcm`. Returns floats in [0,1]."""
    n_samples = len(pcm) // 2
    if n_samples == 0:
        return []
    hop = int(_SR * _HOP_S)
    win = int(_SR * _WIN_S)
    samples = struct.unpack(f"<{n_samples}h", pcm)
    out: list[float] = []
    for i in range(0, n_samples - win, hop):
        chunk = samples[i:i + win]
        # Integer RMS (avoid building a float list per chunk)
        s = 0
        for v in chunk:
            s += v * v
        out.append((s / win) ** 0.5 / 32768.0)
    return out


def _noise_floor(env: list[float]) -> float:
    """Estimate the room-tone level.

    Uses the 5th percentile RMS (true silence frames in the window) so
    the threshold doesn't chase speech levels in segments dominated by
    talking. Clamped to a floor so we never divide by zero.
    """
    if not env:
        return 0.0
    s = sorted(env)
    p05 = s[max(0, len(s) // 20)]
    return max(p05, 0.0008)  # ~-62 dBFS


def snap_to_silence(
    src_video: Path,
    start_s: float,
    end_s: float,
    *,
    head_search: float = 1.5,
    tail_search: float = 1.5,
    head_lead: float = 0.05,
    tail_lead: float = 0.12,
    speech_factor: float = 3.0,
    min_speech_s: float = 0.04,
) -> tuple[float, float]:
    """Tighten `(start_s, end_s)` to the actual speech boundaries.

    This only ever moves boundaries INWARD (never extends the clip).

    HEAD: probes RMS in `[start_s, start_s + head_search]`. The first
    sustained speech frame defines the speech onset; the returned start
    is `head_lead` seconds before that, but never earlier than `start_s`
    (so user-curated boundaries that cut into a word are preserved).

    TAIL: same logic mirrored for `[end_s - tail_search, end_s]`.

    Speech is defined as RMS above `speech_factor * noise_floor` for at
    least `min_speech_s` — the noise floor is taken from the 25th
    percentile of the search window itself, so the threshold adapts to
    whatever recording this is.

    If detection is inconclusive (the entire window is silence, or the
    entire window is loud) the boundary is left unchanged.

    Args:
        src_video: Path to the source video.
        start_s, end_s: Current clip boundaries in source seconds.
        head_search: Forward-search depth from start_s. 1.2s is enough
            to span the longest plausible breath / room-tone gap before
            a speaker resumes.
        tail_search: Backward-search depth from end_s.
        head_lead: Silence kept before the detected first word.
        tail_lead: Silence kept after the detected last word. Generous
            because Whisper word-end timestamps run early and trailing
            consonants need room.
        speech_factor: RMS multiple above the noise floor that counts
            as speech. 4x ≈ +12 dB.
        min_speech_s: Sustained-speech duration required to confirm a
            real onset (filters out clicks / coughs).

    Returns:
        (refined_start, refined_end), guaranteed to satisfy
        `start_s <= refined_start < refined_end <= end_s`. Falls back
        to the inputs on any decode error or if tightening would
        invert the range.
    """
    if not src_video.exists() or end_s <= start_s:
        return start_s, end_s

    new_start = start_s
    new_end = end_s

    # ---- HEAD: tighten forward to first speech onset ----
    h0 = start_s
    h1 = min(end_s, start_s + head_search)
    pcm = _extract_pcm(src_video, h0, h1)
    env = _rms_envelope(pcm)
    if env:
        floor = _noise_floor(env)
        thresh = floor * speech_factor
        min_frames = max(1, int(min_speech_s / _HOP_S))
        run = 0
        first = -1
        for i, v in enumerate(env):
            if v > thresh:
                run += 1
                if run >= min_frames:
                    first = i - (min_frames - 1)
                    break
            else:
                run = 0
        if first >= 0:
            onset = h0 + first * _HOP_S
            # Tighten only: max with start_s ensures we never move the
            # boundary backward (e.g. when start_s already cuts into a
            # word the onset is detected at frame 0 → onset == start_s).
            new_start = max(start_s, onset - head_lead)

    # ---- TAIL: tighten backward to last speech offset ----
    t0 = max(start_s, end_s - tail_search)
    t1 = end_s
    pcm = _extract_pcm(src_video, t0, t1)
    env = _rms_envelope(pcm)
    if env:
        floor = _noise_floor(env)
        thresh = floor * speech_factor
        min_frames = max(1, int(min_speech_s / _HOP_S))
        run = 0
        last = -1
        for i in range(len(env) - 1, -1, -1):
            if env[i] > thresh:
                run += 1
                if run >= min_frames:
                    last = i + (min_frames - 1)
                    break
            else:
                run = 0
        if last >= 0:
            offset = t0 + last * _HOP_S
            new_end = min(end_s, offset + tail_lead)

    if new_end <= new_start:
        return start_s, end_s
    return round(new_start, 3), round(new_end, 3)


def snap_to_content(
    src_video: Path,
    start_s: float,
    end_s: float,
    captions: list[dict] | None = None,
    *,
    head_lead: float = 0.05,
    tail_lead: float = 0.18,
    rms_fallback: bool = True,
) -> tuple[float, float]:
    """Tighten `(start_s, end_s)` to the clip's actual speech content.

    Primary signal: caption boundaries. Each caption has clip-relative
    `start` and `end` keys. The first caption's start marks the desired
    clip onset; the last caption's end marks the desired clip offset.
    The returned source-time bounds are
        new_start = clip.start + max(0, first_cap.start - head_lead)
        new_end   = clip.start + min(clip_dur, last_cap.end + tail_lead)
    Boundaries are never extended outward.

    If no captions are provided (or captions are empty) and
    `rms_fallback` is true, falls through to `snap_to_silence`.

    Args:
        src_video: Source mp4. Used for the RMS fallback path.
        start_s, end_s: Current clip boundaries in source seconds.
        captions: List of caption dicts, each with `start` and `end`
            in CLIP-relative seconds. Pass the manifest entry's
            `captions` field directly.
        head_lead: Silence kept before the first caption (default 50ms).
        tail_lead: Silence kept after the last caption (default 180ms).
            Whisper word-end timestamps run early, so this needs to be
            roomier than head_lead.
        rms_fallback: If captions are empty/missing, use RMS-based
            silence detection instead. Set False to keep boundaries
            unchanged when there's no caption signal.

    Returns:
        (refined_start, refined_end) such that
        `start_s <= refined_start < refined_end <= end_s`.
    """
    if end_s <= start_s:
        return start_s, end_s

    valid = []
    for c in (captions or []):
        try:
            cs = float(c.get("start"))
            ce = float(c.get("end"))
        except (TypeError, ValueError):
            continue
        if ce > cs:
            valid.append((cs, ce))
    if not valid:
        if rms_fallback:
            return snap_to_silence(src_video, start_s, end_s)
        return start_s, end_s

    valid.sort(key=lambda p: p[0])
    first_cap_start = valid[0][0]
    last_cap_end = valid[-1][1]

    clip_dur = end_s - start_s
    new_start_offset = max(0.0, first_cap_start - head_lead)
    new_end_offset = min(clip_dur, last_cap_end + tail_lead)

    new_start = start_s + new_start_offset
    new_end = start_s + new_end_offset
    if new_end <= new_start:
        return start_s, end_s
    return round(new_start, 3), round(new_end, 3)
