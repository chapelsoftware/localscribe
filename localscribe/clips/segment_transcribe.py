"""Run faster-whisper on a single clip's audio segment.

Used by the web UI's "Resync" button: instead of re-grouping the
full-video transcript's word timestamps, this re-runs whisper on just
the clip's [start, end] audio range. That gives a fresh transcription
focused on the segment, which can be more accurate for short clips
(whisper picks better contextual prompts when it isn't drowning in an
hour of surrounding speech).

The returned word list has timestamps in CLIP-RELATIVE seconds (0.0 =
start of the clip), so it can drop straight into a clip's `captions`
array.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

from ..config import pick_faster_whisper_backend

log = logging.getLogger("clips.segment-transcribe")


# A single cached WhisperModel per (model_size, device, compute_type)
# tuple. Loading the model is the slow part (~5-10s on GPU); after that
# transcription of a 30s clip is sub-second. The lock prevents two
# simultaneous resyncs from racing on model construction.
_MODEL_CACHE: dict[tuple[str, str, str], "object"] = {}
_MODEL_LOCK = Lock()


def _get_model(model_size: str):
    device, compute = pick_faster_whisper_backend()
    key = (model_size, device, compute)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel
        log.info("loading whisper model: %s on %s/%s", model_size, device, compute)
        model = WhisperModel(model_size, device=device, compute_type=compute)
        _MODEL_CACHE[key] = model
        return model


def _cut_audio(audio_wav: Path, start_s: float, end_s: float, dest: Path) -> None:
    """Use ffmpeg to slice [start_s, end_s] of `audio_wav` into `dest`.

    Re-encodes (rather than stream-copying) so the output starts at a
    clean keyframe -- whisper is sensitive to leading silence/noise.
    """
    duration = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(audio_wav),
        "-t", f"{duration:.3f}",
        "-ac", "1", "-ar", "16000",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def transcribe_segment(
    audio_wav: Path,
    start_s: float,
    end_s: float,
    *,
    model_size: str = "large-v3",
) -> list[dict]:
    """Re-transcribe just `[start_s, end_s]` of an audio file.

    Returns a list of word dicts:
      [{"start": float, "end": float, "text": str, "prob": float}, ...]
    with timestamps in CLIP-RELATIVE seconds (so the first word's start
    is somewhere near 0, not near `start_s`).
    """
    if not audio_wav.exists():
        raise FileNotFoundError(audio_wav)
    if end_s <= start_s:
        return []

    model = _get_model(model_size)

    with tempfile.TemporaryDirectory() as td:
        seg = Path(td) / "seg.wav"
        _cut_audio(audio_wav, start_s, end_s, seg)
        log.info("transcribing %.2fs segment...", end_s - start_s)
        segments, _info = model.transcribe(
            str(seg),
            word_timestamps=True,
            # Don't use VAD: the segment was already chosen, we want
            # everything in it transcribed even if it's a quiet bit.
            vad_filter=False,
            beam_size=5,
        )
        words: list[dict] = []
        for seg_obj in segments:
            for w in (seg_obj.words or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": text,
                    "prob": float(getattr(w, "probability", 1.0) or 1.0),
                })
    log.info("transcribed %d words", len(words))
    return words
