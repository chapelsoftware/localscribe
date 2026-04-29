"""Stage 2: Transcribe audio to a JSON of segments + word timestamps.

Two engines are supported:
  - faster-whisper (ctranslate2): CPU + CUDA. Fastest on NVIDIA. Default
    everywhere CUDA is available, and the CPU fallback otherwise.
  - whisper-cpp (via pywhispercpp): CPU + Metal. Picked automatically on
    Apple Silicon when pywhispercpp is installed -- Metal gives a large
    speedup over CPU faster-whisper on M-series hardware.

Both engines produce the same output JSON shape so downstream stages
(align, summarize, report) don't care which one ran.
"""
from __future__ import annotations

import logging

from ..config import (
    Paths,
    cached,
    load_json,
    pick_faster_whisper_backend,
    pick_whisper_engine,
    save_json,
)

log = logging.getLogger("transcribe")


# Controls how many audio chunks faster-whisper decodes in parallel.
# Higher -> faster but more VRAM. 8 is comfortable on a 12 GB 3060.
BATCH_SIZE = 8

# Group whisper.cpp's per-word output back into pseudo-segments roughly
# this many seconds long. The exact grouping doesn't matter -- align.py
# rebuilds segments from word timestamps + speaker turns -- but it keeps
# the cached transcript_raw.json human-readable.
WHISPER_CPP_SEGMENT_SECONDS = 30.0


def run(paths: Paths, model_size: str = "large-v3", force: bool = False) -> dict:
    if cached(paths.transcript_raw, force):
        log.info("cached")
        return load_json(paths.transcript_raw)

    engine = pick_whisper_engine()
    if engine == "faster-whisper":
        result = _transcribe_faster_whisper(paths, model_size)
    elif engine == "whisper-cpp":
        result = _transcribe_whisper_cpp(paths, model_size)
    else:
        raise RuntimeError(f"Unknown whisper engine: {engine}")

    save_json(paths.transcript_raw, result)
    log.info("ok: %d segments", len(result['segments']))
    return result


def _transcribe_faster_whisper(paths: Paths, model_size: str) -> dict:
    # Lazy imports -- torch/ctranslate2 are heavy
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device, compute_type = pick_faster_whisper_backend()
    log.info("engine=faster-whisper model=%s device=%s compute=%s batched",
             model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    batched = BatchedInferencePipeline(model=model)

    log.info("running on %s (batch_size=%d)...", paths.audio, BATCH_SIZE)
    segments, info = batched.transcribe(
        str(paths.audio),
        language="en",
        word_timestamps=True,
        batch_size=BATCH_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Materialize the streaming generator
    out_segments = []
    for seg in segments:
        out_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [
                {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                for w in (seg.words or [])
            ],
        })

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments,
    }


def _transcribe_whisper_cpp(paths: Paths, model_size: str) -> dict:
    # Lazy import: only available when the [whisper-cpp] extra is installed
    from pywhispercpp.model import Model

    log.info("engine=whisper-cpp model=%s (Metal on Apple Silicon, Vulkan on Linux+GPU, "
             "CPU otherwise)", model_size)
    # split_on_word + max_len=1 makes whisper.cpp emit one segment per word,
    # which is the most reliable way to get word-level timestamps from the
    # current pywhispercpp API.
    model = Model(
        model=model_size,
        print_progress=False,
        print_realtime=False,
    )

    log.info("running on %s...", paths.audio)
    word_segs = model.transcribe(
        str(paths.audio),
        language="en",
        token_timestamps=True,
        max_len=1,
        split_on_word=True,
    )

    # whisper.cpp timestamps are in centiseconds (1/100 sec)
    words = []
    for s in word_segs:
        words.append({
            "start": s.t0 / 100.0,
            "end":   s.t1 / 100.0,
            # whisper.cpp tends to omit the leading space that faster-whisper
            # emits; align.py concatenates words and then strips, so adding
            # one keeps spacing right in the joined transcript.
            "word":  (" " + s.text.lstrip()),
            # whisper.cpp doesn't expose a per-word probability via this API
            "prob":  None,
        })

    # Group words into ~30s pseudo-segments for the on-disk JSON
    out_segments: list[dict] = []
    cur: list[dict] = []
    for w in words:
        if cur and (w["start"] - cur[0]["start"]) > WHISPER_CPP_SEGMENT_SECONDS:
            out_segments.append(_pack_segment(cur))
            cur = []
        cur.append(w)
    if cur:
        out_segments.append(_pack_segment(cur))

    duration = words[-1]["end"] if words else 0.0
    return {
        "language": "en",
        "duration": duration,
        "segments": out_segments,
    }


def _pack_segment(words: list[dict]) -> dict:
    return {
        "start": words[0]["start"],
        "end":   words[-1]["end"],
        "text":  "".join(w["word"] for w in words),
        "words": words,
    }
