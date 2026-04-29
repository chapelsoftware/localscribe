"""Light tests for pure-logic pieces -- no torch/yt-dlp/llm calls needed."""
from pathlib import Path
from unittest import mock

import pytest

import localscribe.llm as llm
from localscribe.llm import claude_cli as _claude_cli  # noqa: F401
from localscribe.llm import codex_cli as _codex_cli    # noqa: F401
from localscribe.llm import gemini_cli as _gemini_cli  # noqa: F401
from localscribe.llm import openai_api as _openai_api  # noqa: F401
from localscribe.stages.align import _speaker_for_time
from localscribe.stages.download import (
    _local_file_id,
    _looks_like_local_path,
    extract_video_id,
)
from localscribe.stages.summarize import _chunk_transcript, _format_transcript, _hms
from localscribe.text import to_ascii


def test_extract_video_id():
    cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ]
    for url, expected in cases:
        assert extract_video_id(url) == expected, url


def test_looks_like_local_path():
    yes = [
        "/abs/path/audio.mp3",
        "~/Downloads/talk.m4a",
        "./local.wav",
        "../other/clip.webm",
        "audio.mp3",
        "talk.MP4",
        "weird name.flac",
    ]
    no = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtu.be/dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
        "just-some-string",
    ]
    for s in yes:
        assert _looks_like_local_path(s), s
    for s in no:
        assert not _looks_like_local_path(s), s


def test_local_file_id_stable_and_safe():
    p = Path("/tmp/some weird:name!.mp3")
    a = _local_file_id(p)
    b = _local_file_id(p)
    assert a == b, "id should be stable for the same path"
    c = _local_file_id(Path("/other/dir/some weird:name!.mp3"))
    assert a != c, "id should differ when the absolute path differs"
    assert all(ch.isalnum() or ch in "_-" for ch in a), f"unsafe chars: {a!r}"
    fallback = _local_file_id(Path("/tmp/!!!.wav"))
    assert fallback and "-" in fallback


def test_speaker_for_time():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.5, "end": 10.0, "speaker": "B"},
        {"start": 12.0, "end": 15.0, "speaker": "A"},
    ]
    starts = [t["start"] for t in turns]
    assert _speaker_for_time(turns, starts, 2.5) == "A"
    assert _speaker_for_time(turns, starts, 7.0) == "B"
    assert _speaker_for_time(turns, starts, 13.0) == "A"
    # Gap between 10 and 12 -- should pick the nearest
    assert _speaker_for_time(turns, starts, 10.5) == "B"
    assert _speaker_for_time(turns, starts, 11.9) == "A"


def test_chunk_transcript_respects_budget():
    segs = [
        {"speaker": "A", "start": float(i), "end": float(i + 1),
         "text": "x" * 200}
        for i in range(1000)
    ]
    chunks = _chunk_transcript(segs)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 1000


def test_hms():
    assert _hms(0) == "00:00:00"
    assert _hms(61) == "00:01:01"
    assert _hms(3661) == "01:01:01"


def test_format_transcript():
    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "hi"},
    ]
    out = _format_transcript(segs)
    assert "[00:00:00] Alice: hello" in out
    assert "[00:00:02] Bob: hi" in out


def test_llm_dispatch_default_is_claude_cli():
    captured = {}
    def fake_ask(prompt, *, system=None, timeout=900):
        captured["called"] = "claude_cli"
        captured["prompt"] = prompt
        return "ok"
    # llm/__init__.py imports get_env once at module load, so patch the
    # already-bound reference inside the llm package, not config.
    with mock.patch("localscribe.llm.get_env", return_value=None), \
         mock.patch.object(llm.claude_cli, "ask", fake_ask):
        out = llm.ask("hi")
    assert out == "ok"
    assert captured["called"] == "claude_cli"
    assert captured["prompt"] == "hi"


def test_llm_dispatch_routes_to_openai_api():
    captured = {}
    def fake_ask(prompt, *, system=None, timeout=900):
        captured["called"] = "openai_api"
        return "ok"
    with mock.patch("localscribe.llm.get_env", return_value="openai_api"), \
         mock.patch.object(llm.openai_api, "ask", fake_ask):
        out = llm.ask("hi")
    assert out == "ok"
    assert captured["called"] == "openai_api"


@pytest.mark.parametrize("backend,module", [
    ("codex_cli",  "codex_cli"),
    ("gemini_cli", "gemini_cli"),
])
def test_llm_dispatch_routes_to_subscription_clis(backend, module):
    captured = {}
    def fake_ask(prompt, *, system=None, timeout=900):
        captured["called"] = backend
        return "ok"
    with mock.patch("localscribe.llm.get_env", return_value=backend), \
         mock.patch.object(getattr(llm, module), "ask", fake_ask):
        out = llm.ask("hi")
    assert out == "ok"
    assert captured["called"] == backend


def test_llm_dispatch_unknown_backend_errors():
    with mock.patch("localscribe.llm.get_env", return_value="bogus"):
        with pytest.raises(llm.LLMError, match="bogus"):
            llm.ask("hi")


def test_to_ascii():
    cases = [
        ("em — dash",    "em -- dash"),
        ("en – dash",    "en - dash"),
        ("smart “quotes”", 'smart "quotes"'),
        ("apostrophe’s",  "apostrophe's"),
        ("ellipsis…",    "ellipsis..."),
        ("arrow → there", "arrow -> there"),
        ("café",         "cafe"),
        ("résumé",  "resume"),
        ("nbsp here",    "nbsp here"),
        ("bullet • pt",  "bullet * pt"),
        ("plain ascii",       "plain ascii"),
        ("",                  ""),
    ]
    for src, expected in cases:
        assert to_ascii(src) == expected, src
    dropped = to_ascii("emoji \U0001F600 and chinese 中文")
    assert all(ord(c) < 128 for c in dropped)
