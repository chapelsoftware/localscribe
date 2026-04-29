"""Stage 5: Identify speakers AND classify the content type.

One Claude call produces:
  1. A label map SPEAKER_XX -> display name (names accepted at high/medium
     confidence; low stays anonymous).
  2. A content_type classification in
     {debate, interview, panel, discussion, monologue}
     with a confidence and one-sentence rationale.

Both go into speakers.json. Downstream stages (summarize, report) read
content_type to pick their output recipe.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from ..config import Paths, cached, load_json, save_json
from ..llm import ask

log = logging.getLogger("identify")


SAMPLE_CHARS_PER_SPEAKER = 2500

CONTENT_TYPES = ("debate", "interview", "panel", "discussion", "monologue")

SYSTEM = (
    "You analyze transcripts of multi-speaker videos. "
    "You identify speakers when the transcript or metadata supports it "
    "(self-introductions, being addressed by name, title/description mentions), "
    "and you classify the overall content type. "
    "Never invent names from outside knowledge. "
    "Prefer SPEAKER_XX over a wrong guess."
)


def _gather_samples(segments: list[dict]) -> dict[str, str]:
    """For each speaker, collect their first ~SAMPLE_CHARS_PER_SPEAKER of speech."""
    buckets: dict[str, list[str]] = defaultdict(list)
    lens: dict[str, int] = defaultdict(int)
    for seg in segments:
        spk = seg["speaker"]
        if lens[spk] >= SAMPLE_CHARS_PER_SPEAKER:
            continue
        text = seg["text"]
        buckets[spk].append(text)
        lens[spk] += len(text)
    return {spk: " ".join(parts)[:SAMPLE_CHARS_PER_SPEAKER]
            for spk, parts in buckets.items()}


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of Claude's response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object in Claude response:\n{text}")


def _build_label_map(speakers_data: dict) -> dict[str, str]:
    """SPEAKER_XX -> display name. Accept high and medium confidence."""
    label_map: dict[str, str] = {}
    for spk, info in speakers_data.items():
        name = info.get("name") if isinstance(info, dict) else None
        conf = (info.get("confidence") or "").lower() if isinstance(info, dict) else ""
        if name and conf in ("high", "medium"):
            label_map[spk] = name
        else:
            label_map[spk] = spk
    return label_map


def _normalize_content_type(raw: dict | None, n_speakers: int) -> dict:
    """Normalize Claude's content_type block; fall back sensibly on bad output."""
    default_type = "monologue" if n_speakers <= 1 else "discussion"
    if not isinstance(raw, dict):
        return {"type": default_type, "confidence": "low",
                "reason": "classifier returned no content_type; fell back"}
    t = (raw.get("type") or "").lower().strip()
    if t not in CONTENT_TYPES:
        return {"type": default_type, "confidence": "low",
                "reason": f"unknown type '{t}'; fell back"}
    return {
        "type": t,
        "confidence": (raw.get("confidence") or "low").lower(),
        "reason": raw.get("reason") or "",
    }


def run(paths: Paths, force: bool = False,
        content_type_override: str | None = None) -> dict:
    """Run identification + classification.

    If content_type_override is set (e.g. from --content-type CLI flag), the
    classifier's answer is ignored for the final type, but we still ask Claude
    so we get speaker names.
    """
    if cached(paths.speakers, force):
        result = load_json(paths.speakers)
        # If the cached file is the old schema (flat speakers at top level,
        # no content_type), we need to re-run.
        if isinstance(result, dict) and "content_type" in result:
            if content_type_override:
                current_type = result.get("content_type", {}).get("type")
                # Only rewrite if the override actually changes something.
                # Otherwise we'd bump speakers.json mtime and invalidate
                # all the downstream summary files for no reason.
                if current_type != content_type_override:
                    result["content_type"] = {
                        "type": content_type_override,
                        "confidence": "override",
                        "reason": "forced by --content-type flag",
                    }
                    save_json(paths.speakers, result)
            log.info("cached (type=%s)", result['content_type']['type'])
            return result
        log.info("cache is old schema, re-running")

    metadata = load_json(paths.metadata)
    segments = load_json(paths.transcript_aligned)
    samples = _gather_samples(segments)
    n_speakers = len(samples)

    speakers_block = "\n\n".join(
        f"### {spk}\n{sample}" for spk, sample in sorted(samples.items())
    )

    prompt = f"""I need you to analyze a video transcript and return a single JSON object.

## Video metadata
- Title: {metadata.get("title")}
- Channel: {metadata.get("channel") or metadata.get("uploader")}
- Description (first 2000 chars):
{(metadata.get("description") or "")[:2000]}

## Speaker samples (early speech from each speaker)

{speakers_block}

## Your task

Return a JSON object with exactly two top-level keys: "content_type" and "speakers".

### content_type
Classify the overall format of the video. Pick ONE of these five types:

- "debate": Formal debate with opposing positions. Structured rounds (openings, rebuttals, cross-ex, closings). Usually a moderator. There is a resolution/proposition being argued for and against.
- "interview": One interviewer asking questions of one or more guests. Asymmetric: the interviewer prompts and the guest(s) respond at length. Typical of podcasts where a host interviews an expert, author, or public figure.
- "panel": Three or more participants discussing shared topics, usually with a moderator. Multiple perspectives offered cooperatively (not adversarially). Typical of conference panels, roundtables, news shows with multiple guests.
- "discussion": Two to a few speakers conversing without formal structure. Symmetric turn-taking. Casual podcast, co-hosted show, two friends chatting. Not adversarial, not a formal interview.
- "monologue": A single speaker delivering content: lecture, talk, tutorial, essay, solo podcast episode. May include brief audience Q&A but the body is one person.

Return:
{{
  "type": "one of the five above",
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence citing specific transcript features"
}}

### speakers
For each SPEAKER_XX label, return:
  - "name": the person's name if identifiable, otherwise null
  - "role": short role/description if inferable (e.g. "host", "guest", "moderator", "debater", "panelist"), else null
  - "confidence": "high" | "medium" | "low"
  - "reason": one-sentence justification

What counts as "identifiable":
- Self-introduction ("I'm Jane Smith...")
- Another speaker addressing or referring to them by name ("Thanks, Mr. Knapp")
- The title or description clearly listing them in a role this speaker fills
- Partial names (last or first only) are acceptable

Confidence levels:
- "high": self-introduction, or title/description names the role and content matches
- "medium": clear in-transcript reference (addressed by name) matching role
- "low": only circumstantial hints, or conflicting signals -- leave name as null

Do not guess a name based on general knowledge.

## Example output format

{{
  "content_type": {{
    "type": "interview",
    "confidence": "high",
    "reason": "Host asks probing questions for 90+ minutes while the guest gives long first-person responses about their book."
  }},
  "speakers": {{
    "SPEAKER_00": {{"name": "Jane Smith", "role": "host", "confidence": "high", "reason": "introduces herself at 0:05"}},
    "SPEAKER_01": {{"name": null, "role": "guest", "confidence": "low", "reason": "never named in transcript"}}
  }}
}}

Respond with ONLY the JSON object, no prose before or after.
"""

    log.info("asking claude to identify %d speakers + classify content type...", n_speakers)
    raw = ask(prompt, system=SYSTEM)
    parsed = _extract_json(raw)

    # Back-compat: if Claude ignored the new schema and returned a flat
    # SPEAKER_XX dict, treat that as the speakers block.
    if "speakers" in parsed or "content_type" in parsed:
        speakers_data = parsed.get("speakers", {}) or {}
        ct_data = parsed.get("content_type")
    else:
        speakers_data = parsed
        ct_data = None

    label_map = _build_label_map(speakers_data)
    content_type = _normalize_content_type(ct_data, n_speakers)

    if content_type_override:
        content_type = {
            "type": content_type_override,
            "confidence": "override",
            "reason": "forced by --content-type flag",
        }

    result = {
        "label_map": label_map,
        "content_type": content_type,
        "raw": speakers_data,
    }
    save_json(paths.speakers, result)

    named = sum(1 for k, v in label_map.items() if v != k)
    log.info("ok: %d/%d speakers named, type=%s (%s)",
             named, len(label_map), content_type['type'], content_type['confidence'])
    return result
