"""Configuration, paths, and stage-cache helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "output"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cached(path: Path, force: bool = False) -> bool:
    """Return True if the cached artifact exists and should be reused."""
    return path.exists() and path.stat().st_size > 0 and not force


def pick_torch_device():
    """Pick the best torch device for general inference (pyannote).

    Order: CUDA > MPS (Apple Silicon) > CPU. Imports torch lazily so the
    CLI startup path stays cheap.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_faster_whisper_backend() -> tuple[str, str]:
    """Pick (device, compute_type) for faster-whisper / ctranslate2.

    ctranslate2 only supports "cuda" and "cpu" -- no MPS, no ROCm.
    int8 is the standard fast CPU compute type.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def pick_whisper_engine() -> str:
    """Pick the whisper engine: 'faster-whisper' or 'whisper-cpp'.

    NVIDIA users always get faster-whisper -- ctranslate2 + CUDA outperforms
    whisper.cpp + CUDA. For everyone else, prefer whisper-cpp if pywhispercpp
    is installed: setup.sh only installs it when whisper.cpp can be GPU-
    accelerated (Metal on Apple Silicon, Vulkan on Linux+AMD/Intel).
    """
    import torch
    if torch.cuda.is_available():
        return "faster-whisper"
    try:
        import pywhispercpp  # noqa: F401
        return "whisper-cpp"
    except ImportError:
        return "faster-whisper"


@dataclass
class Paths:
    """All per-video paths. Create with Paths.for_video(video_id)."""
    video_id: str
    root: Path

    @classmethod
    def for_video(cls, video_id: str) -> "Paths":
        root = OUTPUT_ROOT / video_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "chunks").mkdir(exist_ok=True)
        return cls(video_id=video_id, root=root)

    # Stage artifacts
    @property
    def audio(self) -> Path: return self.root / "audio.wav"
    @property
    def metadata(self) -> Path: return self.root / "metadata.json"
    @property
    def transcript_raw(self) -> Path: return self.root / "transcript.raw.json"
    @property
    def diarization(self) -> Path: return self.root / "diarization.json"
    @property
    def transcript_aligned(self) -> Path: return self.root / "transcript.aligned.json"
    @property
    def speakers(self) -> Path: return self.root / "speakers.json"
    @property
    def chunks_dir(self) -> Path: return self.root / "chunks"

    # Final outputs
    @property
    def transcript_md(self) -> Path: return self.root / "transcript.md"
    @property
    def tldr_md(self) -> Path: return self.root / "tldr.md"
    @property
    def per_speaker_md(self) -> Path: return self.root / "per_speaker.md"
    @property
    def chronological_md(self) -> Path: return self.root / "chronological.md"
    @property
    def debate_map_md(self) -> Path: return self.root / "debate_map.md"
    # Discussion / interview / panel outputs
    @property
    def topics_md(self) -> Path: return self.root / "topics.md"
    @property
    def highlights_md(self) -> Path: return self.root / "highlights.md"
    # Monologue outputs
    @property
    def key_points_md(self) -> Path: return self.root / "key_points.md"
    @property
    def outline_md(self) -> Path: return self.root / "outline.md"
    @property
    def quotes_md(self) -> Path: return self.root / "quotes.md"
    @property
    def report_html(self) -> Path: return self.root / "report.html"
    # Snapshot state
    @property
    def manifest(self) -> Path: return self.root / ".manifest.json"
    @property
    def snapshots_dir(self) -> Path: return self.root / "snapshots"


def get_env(key: str, default: str | None = None) -> str | None:
    """Read an env var, falling back to the .env file in the repo root.

    Process env wins over .env. Returns `default` if neither has the key.
    Used for both runtime config (LOCALSCRIBE_LLM_BACKEND, OPENAI_*) and
    secrets (HF_TOKEN, OPENAI_API_KEY).
    """
    val = os.environ.get(key)
    if val is not None:
        return val
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        prefix = f"{key}="
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                return line[len(prefix):].strip().strip('"').strip("'")
    return default


def hf_token() -> str:
    """Find a HuggingFace token.

    Checks, in order:
      1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars
      2. .env file in repo root
      3. ~/.cache/huggingface/token (written by `huggingface-cli login`)
    """
    token = get_env("HF_TOKEN") or get_env("HUGGING_FACE_HUB_TOKEN")
    if not token:
        hf_cache_token = Path.home() / ".cache" / "huggingface" / "token"
        if hf_cache_token.exists():
            token = hf_cache_token.read_text().strip()
    if not token:
        raise RuntimeError(
            "No HuggingFace token found. Either run `huggingface-cli login` "
            "or export HF_TOKEN=hf_xxx. You also need to accept the gated "
            "model terms at https://hf.co/pyannote/speaker-diarization-community-1"
        )
    return token
