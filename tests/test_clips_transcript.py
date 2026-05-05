"""Pure-logic tests for clips.transcript phrase resolution."""
import json
from pathlib import Path

import pytest

from localscribe.clips import transcript as tr


def _words(*tokens: tuple[str, float, float]) -> list[dict]:
    """Build a minimal word list. Each token: (text, start, end)."""
    return [
        {"start": s, "end": e, "word": w}
        for w, s, e in tokens
    ]


def test_normalize_strips_punct_and_lowercases():
    assert tr._normalize("We're not trying.") == "we re not trying"
    assert tr._normalize("World,") == "world"
    assert tr._normalize("  HELLO   World  ") == "hello world"


def test_find_phrase_basic_match():
    words = _words(
        ("Hello", 0.0, 0.4),
        ("there", 0.4, 0.8),
        ("friend", 0.8, 1.2),
    )
    m = tr.find_phrase(words, "hello there")
    assert m is not None
    s, e, idx_after = m
    assert s == pytest.approx(0.0)
    assert e == pytest.approx(0.8)
    assert idx_after == 2


def test_find_phrase_returns_none_when_missing():
    words = _words(("only", 0.0, 0.5))
    assert tr.find_phrase(words, "not here") is None


def test_find_phrase_handles_punctuation_in_tokens():
    """Whisper embeds punctuation in word tokens (e.g. 'world.'). The
    normalizer must strip it so a clean search phrase still matches."""
    words = _words(
        ("Hello", 0.0, 0.3),
        ("world.", 0.3, 0.6),
        ("Then", 0.6, 0.9),
    )
    m = tr.find_phrase(words, "Hello world")
    assert m is not None and m[0] == 0.0 and m[1] == 0.6


def test_find_phrase_normalizes_contractions():
    """`we are` should match `we're`."""
    words = _words(
        ("We're", 0.0, 0.3),
        ("not", 0.3, 0.5),
        ("done.", 0.5, 0.9),
    )
    m = tr.find_phrase(words, "we re not")
    assert m is not None


def test_find_phrase_respects_start_idx():
    """A second occurrence is found if start_idx skips past the first."""
    words = _words(
        ("repeat", 0.0, 0.5),
        ("middle", 0.5, 1.0),
        ("repeat", 1.0, 1.5),
    )
    first = tr.find_phrase(words, "repeat")
    assert first[0] == pytest.approx(0.0)
    second = tr.find_phrase(words, "repeat", start_idx=first[2])
    assert second is not None
    assert second[0] == pytest.approx(1.0)


def test_find_phrase_max_window_caps_search():
    """A phrase that spans more than max_window words is not matched."""
    words = _words(*(("w", float(i), float(i) + 0.5) for i in range(50)))
    assert tr.find_phrase(words, "w " * 40, max_window=5) is None


def test_resolve_boundaries_end_to_end(tmp_path: Path):
    """End-to-end: write a fake transcript.raw.json and resolve a
    pair of phrases against it."""
    seg = {
        "start": 0.0, "end": 5.0, "text": "...",
        "words": [
            {"start": 1.0, "end": 1.4, "word": "The"},
            {"start": 1.4, "end": 1.9, "word": "quick"},
            {"start": 1.9, "end": 2.6, "word": "brown"},
            {"start": 2.6, "end": 3.0, "word": "fox"},
            {"start": 3.0, "end": 3.3, "word": "jumps"},
            {"start": 3.3, "end": 3.9, "word": "over"},
            {"start": 3.9, "end": 4.5, "word": "everything."},
        ],
    }
    p = tmp_path / "transcript.raw.json"
    p.write_text(json.dumps({"segments": [seg]}))
    s, e = tr.resolve_boundaries(p, "The quick brown fox", "jumps over everything")
    # Start padded back by 0.1, end padded forward by 0.3 (defaults).
    assert s == pytest.approx(1.0 - 0.1)
    assert e == pytest.approx(4.5 + 0.3)


def test_resolve_boundaries_searches_end_after_start(tmp_path: Path):
    """If the end phrase recurs earlier, resolve_boundaries should
    pick the occurrence AFTER the start match, not before."""
    seg = {
        "start": 0.0, "end": 10.0, "text": "...",
        "words": [
            {"start": 0.0, "end": 0.3, "word": "fox"},  # earlier!
            {"start": 0.3, "end": 1.0, "word": "first"},
            {"start": 1.0, "end": 1.4, "word": "The"},
            {"start": 1.4, "end": 1.9, "word": "quick"},
            {"start": 1.9, "end": 2.6, "word": "brown"},
            {"start": 2.6, "end": 3.0, "word": "fox"},
        ],
    }
    p = tmp_path / "transcript.raw.json"
    p.write_text(json.dumps({"segments": [seg]}))
    s, e = tr.resolve_boundaries(p, "The quick brown", "fox")
    # End must come from the second "fox" at 3.0, not the first at 0.3.
    assert e == pytest.approx(3.0 + 0.3)
    assert s == pytest.approx(1.0 - 0.1)


def test_resolve_boundaries_raises_on_missing_start(tmp_path: Path):
    p = tmp_path / "transcript.raw.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "words": [
            {"start": 0.0, "end": 0.5, "word": "hi"},
        ]}
    ]}))
    with pytest.raises(ValueError, match="start phrase not found"):
        tr.resolve_boundaries(p, "missing", "hi")


def test_resolve_boundaries_raises_on_missing_end(tmp_path: Path):
    p = tmp_path / "transcript.raw.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "words": [
            {"start": 0.0, "end": 0.5, "word": "hi"},
            {"start": 0.5, "end": 1.0, "word": "there"},
        ]}
    ]}))
    with pytest.raises(ValueError, match="end phrase not found"):
        tr.resolve_boundaries(p, "hi", "missing")
