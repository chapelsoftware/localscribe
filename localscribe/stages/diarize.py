"""Stage 3: Speaker diarization with pyannote.audio 3.1."""
from __future__ import annotations

import logging
import os

from ..config import Paths, cached, hf_token, load_json, pick_torch_device, save_json

log = logging.getLogger("diarize")


def run(paths: Paths, force: bool = False) -> list[dict]:
    if cached(paths.diarization, force):
        log.info("cached")
        return load_json(paths.diarization)

    import torch

    # torch 2.6 flipped torch.load's default to weights_only=True for safety,
    # but pyannote's checkpoints contain many Python globals (TorchVersion,
    # Specifications, etc.) that the safe-unpickler rejects. Allowlisting
    # each class is whack-a-mole; instead, force weights_only=False just
    # while we load the pipeline. Pyannote checkpoints come from a gated,
    # authenticated HF repo, so trusting them is acceptable.
    _orig_torch_load = torch.load

    def _trusting_load(*args, **kwargs):
        kwargs["weights_only"] = False  # force, not setdefault
        return _orig_torch_load(*args, **kwargs)

    torch.load = _trusting_load
    try:
        from pyannote.audio import Pipeline
        log.info("loading pyannote/speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token(),
        )
    finally:
        torch.load = _orig_torch_load

    device = pick_torch_device()
    if device.type == "mps":
        # Some pyannote ops (cdist on certain dtypes) lack MPS kernels;
        # let them fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    pipeline.to(device)

    log.info("running on %s (device=%s)...", paths.audio, device.type)
    diarization = pipeline(str(paths.audio))

    turns = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })

    save_json(paths.diarization, turns)
    n_speakers = len({t["speaker"] for t in turns})
    log.info("ok: %d turns, %d speakers", len(turns), n_speakers)
    return turns
