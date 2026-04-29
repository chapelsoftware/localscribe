"""Stage 6: Chunked summarization + content-type-aware final synthesis.

Pipeline:
  1. Format the aligned, speaker-labeled transcript.
  2. Chunk by character budget, respecting speaker-turn boundaries.
  3. Summarize each chunk into a content-neutral structured note (shared
     across all content types -- topics, who said what, claims, evidence,
     disagreements-if-any).
  4. Read content_type from speakers.json (set by the identify stage).
  5. Run the synthesis prompts for that content type. Each content type has
     its own set of final output files.

Content types and their outputs:
  debate      -> tldr, per_speaker, chronological, debate_map
  interview   -> tldr, topics, per_speaker, chronological, highlights
  panel       -> tldr, topics, per_speaker, chronological, highlights
  discussion  -> tldr, topics, per_speaker, chronological, highlights
  monologue   -> tldr, key_points, outline, quotes

Also emits transcript.md (a pretty-print of the aligned transcript -- no
LLM involved for that one).
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from ..config import Paths, cached, load_json
from ..llm import ask
from ..text import to_ascii, write_ascii

log = logging.getLogger("summarize")


# ---------- snapshot + manifest helpers ----------

def _read_manifest(paths: Paths) -> dict | None:
    if not paths.manifest.exists():
        return None
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(paths: Paths, content_type: str, outputs: list[str]) -> None:
    data = {
        "content_type": content_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": outputs,
    }
    paths.manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _take_snapshot(paths: Paths, label: str) -> Path | None:
    """Copy all current top-level summary files into snapshots/<ts>-<label>/.

    Returns the snapshot dir if anything was copied, else None.
    """
    # Only snapshot the human-facing outputs; skip audio/diarization/etc.
    items: list[Path] = sorted(paths.root.glob("*.md"))
    if paths.report_html.exists() and paths.report_html.stat().st_size > 0:
        items.append(paths.report_html)
    if paths.manifest.exists():
        items.append(paths.manifest)
    items = [p for p in items if p.exists() and p.stat().st_size > 0]
    if not items:
        return None

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    snap_dir = paths.snapshots_dir / f"{ts}-{safe_label}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    for p in items:
        shutil.copy2(p, snap_dir / p.name)

    log.info("snapshot: %s (%d files)", snap_dir.relative_to(paths.root), len(items))
    return snap_dir


def _fresh(target: Path, speakers_path: Path, force: bool) -> bool:
    """Cached AND not invalidated by speakers.json being newer.

    If the content type or speaker labels change, speakers.json gets a newer
    mtime and we treat any previously-written summary file as stale.
    """
    if force or not target.exists() or target.stat().st_size == 0:
        return False
    if speakers_path.exists() and speakers_path.stat().st_mtime > target.stat().st_mtime:
        return False
    return True


# ~60k chars ~= ~15k tokens -- leaves plenty of room for the prompt scaffolding
CHUNK_CHAR_BUDGET = 60_000

SYSTEM_SUMMARIZE = (
    "You summarize transcripts of multi-speaker videos. "
    "Be faithful to what speakers actually said. "
    "Do not invent facts, names, or claims. "
    "Preserve disagreements and nuance -- do not flatten opposing views into a neutral blur. "
    "Write in plain ASCII only: use '--' for em dashes, straight quotes (\" and '), "
    "'...' for ellipses, and '->' for arrows. Do not use smart quotes or Unicode punctuation."
)


# ---------- helpers ----------

def _hms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _apply_labels(segments: list[dict], label_map: dict[str, str]) -> list[dict]:
    return [{**s, "speaker": label_map.get(s["speaker"], s["speaker"])}
            for s in segments]


def _format_transcript(segments: list[dict]) -> str:
    return "\n".join(
        f"[{_hms(s['start'])}] {s['speaker']}: {s['text']}" for s in segments
    )


def _chunk_transcript(segments: list[dict]) -> list[list[dict]]:
    """Split at speaker-turn boundaries, aiming for CHUNK_CHAR_BUDGET per chunk."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for seg in segments:
        seg_len = len(seg["text"]) + len(seg["speaker"]) + 20
        if current and current_len + seg_len > CHUNK_CHAR_BUDGET:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(seg)
        current_len += seg_len
    if current:
        chunks.append(current)
    return chunks


def _format_speaker_roster(speakers: dict) -> str:
    roster_lines = []
    raw = speakers.get("raw", {})
    label_map = speakers["label_map"]
    for spk, display in sorted(label_map.items()):
        info = raw.get(spk, {}) if isinstance(raw, dict) else {}
        role = info.get("role") if isinstance(info, dict) else None
        if display != spk and role:
            roster_lines.append(f"- **{display}** ({role})  [was {spk}]")
        elif display != spk:
            roster_lines.append(f"- **{display}**  [was {spk}]")
        elif role:
            roster_lines.append(f"- **{spk}** ({role})")
        else:
            roster_lines.append(f"- **{spk}** (unidentified)")
    return "\n".join(roster_lines)


# ---------- chunk summary (shared across all content types) ----------

CHUNK_PROMPT = """You are summarizing ONE chunk of a longer multi-speaker video transcript.
This chunk-level note will be combined with others to produce a final summary.

## Video title
{title}

## Content type
{content_type}

## Speakers in this video
{speaker_roster}

## This chunk (timestamps in HH:MM:SS)
{chunk}

## Your task
Produce a structured note about THIS chunk only. Use this exact markdown format:

### Topics in this chunk
- (bullet list of topics/themes actually discussed)

### Per-speaker contributions
- SPEAKER_NAME: (specifically what they said, asked, argued, described, or brought up)

### Notable statements / claims / stories
- SPEAKER_NAME @ HH:MM:SS -- "short paraphrase or quote of something interesting, strongly asserted, or memorable"

### Disagreements / pushback (if any)
- SPEAKER_A @ HH:MM:SS disagreed with SPEAKER_B's claim that X, saying Y
- (write "(none)" if the conversation is cooperative)

### Evidence, sources, examples cited
- SPEAKER_NAME cited: (study, book, person, statistic, anecdote, example)

If a section has nothing to report for this chunk, write "- (none)".
Do not add any prose outside this structure.
"""


def _chunk_summary(chunk_text: str, title: str, content_type: str,
                   speaker_roster: str) -> str:
    prompt = CHUNK_PROMPT.format(
        title=title,
        content_type=content_type,
        speaker_roster=speaker_roster,
        chunk=chunk_text,
    )
    return ask(prompt, system=SYSTEM_SUMMARIZE, timeout=900)


# ---------- common preamble used in every final-synthesis prompt ----------

_HEADER = """Below are structured notes from every chunk of a multi-speaker video.

## Video
Title: {title}
Channel: {channel}
Duration: {duration}

## Speakers
{speaker_roster}

## Chunk notes
{notes}

## Task
"""


# ---------- DEBATE prompts ----------

DEBATE_TLDR = _HEADER + """Write a short TL;DR of the entire debate in markdown.

Format:
# TL;DR -- {title}

**What it is:** (1 sentence, name the resolution/topic)

**Main thread:** (2-4 sentences: the central disagreement and how each side frames it)

**Key takeaways:**
- (3-6 bullets, the most important points)

**Who said what (one line each):**
- SPEAKER: their central position in one line
"""

DEBATE_PER_SPEAKER = _HEADER + """Write a per-speaker breakdown of the debate.

Format:
# Per-speaker -- {title}

## SPEAKER_NAME
**Overall position:** (1-2 sentences)

**Main arguments:**
- (argument 1, with brief support)
- (argument 2, ...)

**Evidence they cited:**
- (item, or "none provided")

**Where they pushed back on others:**
- (who, on what, briefly -- or "did not")

(Repeat for each speaker including moderator if present.)
"""

DEBATE_CHRONOLOGICAL = _HEADER + """Write a chronological walkthrough of how the debate unfolded.

Format:
# Chronological -- {title}

## HH:MM:SS -- (section heading, e.g. "Bolt's opening: transcendental argument")
- SPEAKER: what they argued/did in this moment
- SPEAKER: response / counterpoint
(5-15 sections depending on length.)
"""

DEBATE_MAP = _HEADER + """Build a debate map: the structure of the disagreements in
this video, with your evaluative judgment of who made the stronger case
on each contested claim.

## How to evaluate a claim

Weigh each contested claim on three axes:

1. **Logical structure** -- does the argument follow validly from its
   premises? Are there unstated assumptions doing load-bearing work?
2. **Evidence quality** -- is the support concrete (specific examples,
   cited sources, verifiable facts) or merely asserted / appeal to
   authority / hand-waved?
3. **Responsiveness** -- did the speaker actually address the opposing
   argument, or did they talk past it, strawman it, or change the subject?

You MUST make a judgment on every claim. "Both sides made good points" is
a cop-out unless genuinely warranted by evenly-matched engagement. If one
side ignored a key objection, say so. If one side's "evidence" was just
assertion, say so. If they were truly evenly matched, say so and explain
what each brought.

Your judgments should be rooted in the specific transcript content, not
in your own views on the topic. You are judging who argued better in THIS
debate, not who is objectively right about the subject.

## Format

# Debate map -- {title}

## Claim 1: (short statement of a contested claim)
- **Asserted by:** SPEAKER(s)
- **Supporting arguments / evidence:**
  - (bullet)
- **Counter-claims / rebuttals:**
  - SPEAKER: (their counter, briefly)
- **Resolved in the debate?** (yes / no / partially -- did either side
  concede, or did the debate actually reach a resolution?)
- **Stronger case:** SPEAKER_NAME -- OR -- "neither -- evenly matched"
  -- OR -- "inconclusive -- insufficient engagement"
- **Why:** (1-3 sentences citing the specific strength or weakness:
  which axis above they won on, or where the opposing side failed. Be
  concrete -- name the argument or evidence that tipped it.)

## Claim 2: ...

(3-10 claims. If the debate has no real disagreement, write a single
section explaining that instead of this format.)

---

## Final tally

- **SPEAKER_A -- stronger case on N claims:** (list the claim numbers)
- **SPEAKER_B -- stronger case on M claims:** (list the claim numbers)
- **Evenly matched / inconclusive -- K claims:** (list the claim numbers)

## Overall assessment

(3-5 sentences: who, on balance, made the better case for their overall
position, and why. This must be rooted in the specific per-claim
judgments above -- reference the claims by number or by short title.
Note any claims where the weaker-arguing side was nonetheless closer to
a commonly-held view, and any claims where the stronger-arguing side
still failed to carry their broader thesis.)
"""


# ---------- DISCUSSION / INTERVIEW / PANEL shared prompts ----------

def _conversation_tldr(kind_desc: str) -> str:
    return _HEADER + f"""Write a short TL;DR of this {kind_desc} in markdown.

Format:
# TL;DR -- {{title}}

**What it is:** (1 sentence)

**Main threads:** (2-5 sentences describing the central themes explored)

**Key takeaways:**
- (3-6 bullets, the most interesting/important things that came out of the conversation)

**Who said what (one line each):**
- SPEAKER: their main role or contribution in one line
"""


TOPICS_PROMPT = _HEADER + """Write a topic-by-topic summary of what was discussed.
Do NOT go in chronological order -- group by topic instead.

Format:
# Topics -- {title}

## (Topic 1)
(1-3 sentences summarizing what was said about this topic and by whom.)

## (Topic 2)
...

(5-15 topics depending on length. Group related sub-topics together.)
"""


INTERVIEW_PER_SPEAKER = _HEADER + """Write a per-speaker breakdown of this interview.

For the host/interviewer: a short section describing what lines of
questioning they pursued, not a full treatment.

For the guest(s): a fuller section covering their background (as
revealed), the main things they said, notable views, stories, and any
claims/opinions they strongly asserted.

Format:
# Per-speaker -- {title}

## INTERVIEWER_NAME (interviewer)
**What they probed for:**
- (line of questioning 1)
- (line of questioning 2)

## GUEST_NAME (guest)
**Background (as revealed in the conversation):**
(1-3 sentences)

**Main views and claims:**
- (view 1)
- (view 2)

**Notable stories or examples:**
- (story 1, briefly)

**Opinions strongly asserted:**
- (opinion, briefly)
"""


PANEL_PER_SPEAKER = _HEADER + """Write a per-panelist breakdown.

For each panelist, capture their distinct perspective on the shared
topics. For a moderator, note what they steered the conversation toward.

Format:
# Per-panelist -- {title}

## PANELIST_NAME
**Distinctive perspective:** (1-2 sentences on what makes their view different from the others)

**Main contributions:**
- (contribution 1)
- (contribution 2)

**Where they agreed or disagreed with others:**
- (who, on what, briefly)
"""


DISCUSSION_PER_SPEAKER = _HEADER + """Write a per-speaker breakdown of this conversation.

Format:
# Per-speaker -- {title}

## SPEAKER_NAME
**Role in the conversation:** (host / co-host / guest / etc.)

**Main things they brought up:**
- (contribution 1)
- (contribution 2)

**Opinions or views they shared:**
- (view 1)
- (view 2)

**Notable stories or examples:**
- (story/example, briefly -- or "none")
"""


CONVERSATION_CHRONOLOGICAL = _HEADER + """Write a chronological walkthrough of the conversation.

Format:
# Chronological -- {title}

## HH:MM:SS -- (topic / section heading)
- SPEAKER: what they said/did in this section
- SPEAKER: response / follow-up
(5-15 sections depending on length. Capture how the conversation actually flowed.)
"""


HIGHLIGHTS_PROMPT = _HEADER + """Pull out the highlights from this conversation: memorable
moments, quotable lines, surprising claims, strong opinions, interesting
stories. Not a summary -- a curated collection.

Format:
# Highlights -- {title}

## (Short label for the moment)
**Speaker:** NAME @ HH:MM:SS
**Quote or paraphrase:** "..."
**Why it stood out:** (1 sentence)

(8-15 highlights. Mix quotable lines, surprising claims, good stories,
and strong opinions. Attribute accurately.)
"""


# ---------- MONOLOGUE prompts ----------

MONOLOGUE_TLDR = _HEADER + """Write a short TL;DR of this single-speaker piece.

Format:
# TL;DR -- {title}

**What it is:** (1 sentence, e.g. "A 45-minute lecture on X by Y")

**Thesis / main message:** (1-2 sentences)

**Key takeaways:**
- (3-6 bullets, the most important points)
"""


MONOLOGUE_KEY_POINTS = _HEADER + """Extract the key points the speaker makes.

Format:
# Key points -- {title}

## 1. (Key point)
(1-3 sentences explaining it in the speaker's own framing, with a brief quote or paraphrase.)

## 2. (Key point)
...

(5-12 key points depending on length. Order by importance, not chronologically.)
"""


MONOLOGUE_OUTLINE = _HEADER + """Write a structured outline of the talk.

Format:
# Outline -- {title}

## HH:MM:SS -- Section 1 title
- Main point
- Sub-point
- Sub-point

## HH:MM:SS -- Section 2 title
...

(Mirror the talk's own structure. Use the speaker's timestamps.)
"""


MONOLOGUE_QUOTES = _HEADER + """Pull out the most quotable, interesting, or memorable lines.

Format:
# Notable quotes -- {title}

## HH:MM:SS
> "Exact or near-exact quote."

**Context:** (1 sentence about what prompted the line.)

(6-15 quotes. Pick lines that are quotable on their own merits -- pithy,
provocative, illuminating, or core to the thesis.)
"""


# ---------- the recipes ----------

# Each entry: content_type -> list of (output_key, prompt_template, path_attr)
RECIPES: dict[str, list[tuple[str, str, str]]] = {
    "debate": [
        ("tldr",          DEBATE_TLDR,          "tldr_md"),
        ("per_speaker",   DEBATE_PER_SPEAKER,   "per_speaker_md"),
        ("chronological", DEBATE_CHRONOLOGICAL, "chronological_md"),
        ("debate_map",    DEBATE_MAP,           "debate_map_md"),
    ],
    "interview": [
        ("tldr",          _conversation_tldr("interview"), "tldr_md"),
        ("topics",        TOPICS_PROMPT,                   "topics_md"),
        ("per_speaker",   INTERVIEW_PER_SPEAKER,           "per_speaker_md"),
        ("chronological", CONVERSATION_CHRONOLOGICAL,      "chronological_md"),
        ("highlights",    HIGHLIGHTS_PROMPT,               "highlights_md"),
    ],
    "panel": [
        ("tldr",          _conversation_tldr("panel discussion"), "tldr_md"),
        ("topics",        TOPICS_PROMPT,                          "topics_md"),
        ("per_speaker",   PANEL_PER_SPEAKER,                      "per_speaker_md"),
        ("chronological", CONVERSATION_CHRONOLOGICAL,             "chronological_md"),
        ("highlights",    HIGHLIGHTS_PROMPT,                      "highlights_md"),
    ],
    "discussion": [
        ("tldr",          _conversation_tldr("discussion"), "tldr_md"),
        ("topics",        TOPICS_PROMPT,                    "topics_md"),
        ("per_speaker",   DISCUSSION_PER_SPEAKER,           "per_speaker_md"),
        ("chronological", CONVERSATION_CHRONOLOGICAL,       "chronological_md"),
        ("highlights",    HIGHLIGHTS_PROMPT,                "highlights_md"),
    ],
    "monologue": [
        ("tldr",       MONOLOGUE_TLDR,       "tldr_md"),
        ("key_points", MONOLOGUE_KEY_POINTS, "key_points_md"),
        ("outline",    MONOLOGUE_OUTLINE,    "outline_md"),
        ("quotes",     MONOLOGUE_QUOTES,     "quotes_md"),
    ],
}


# ---------- main entry ----------

def run(paths: Paths, force: bool = False) -> None:
    metadata = load_json(paths.metadata)
    speakers = load_json(paths.speakers)
    raw_segments = load_json(paths.transcript_aligned)

    content_type = speakers.get("content_type", {}).get("type", "discussion")
    if content_type not in RECIPES:
        log.warning("unknown content_type %r, falling back to discussion", content_type)
        content_type = "discussion"

    recipe = RECIPES[content_type]
    log.info("content_type=%s, will generate %d outputs: %s",
             content_type, len(recipe), ', '.join(key for key, _, _ in recipe))

    # --- snapshot previous outputs if we're about to regenerate anything ---
    # The _fresh check uses speakers.json mtime to decide staleness. We'll
    # regen if: any recipe file is not fresh, OR the content_type differs
    # from what's recorded in .manifest.json.
    prev_manifest = _read_manifest(paths)
    prev_type = prev_manifest.get("content_type") if prev_manifest else None
    will_regen = any(
        not _fresh(getattr(paths, path_attr), paths.speakers, force)
        for _, _, path_attr in recipe
    ) or (prev_type is not None and prev_type != content_type)

    if will_regen:
        snap_label = prev_type or "pre_manifest"
        _take_snapshot(paths, snap_label)

        # Delete any .md files not in the current recipe so the output dir
        # cleanly reflects what we're about to generate. transcript.md is
        # shared and always kept. The snapshot preserves everything we
        # remove, so this is non-destructive.
        current_files = {
            getattr(paths, path_attr).name
            for _, _, path_attr in recipe
        }
        current_files.add("transcript.md")
        for md in paths.root.glob("*.md"):
            if md.name not in current_files:
                md.unlink()
                log.info("removed orphan from prev recipe: %s", md.name)

    # Remap SPEAKER_XX -> display names throughout
    segments = _apply_labels(raw_segments, speakers["label_map"])

    # --- transcript.md (speaker labels depend on speakers.json) ---
    if not _fresh(paths.transcript_md, paths.speakers, force):
        write_ascii(
            paths.transcript_md,
            f"# Transcript -- {metadata.get('title', '(untitled)')}\n\n"
            f"Source: {metadata.get('webpage_url')}\n\n"
            + _format_transcript(segments)
            + "\n",
        )
        log.info("wrote %s", paths.transcript_md.name)

    # --- chunk summaries (shared) ---
    chunks = _chunk_transcript(segments)
    speaker_roster = _format_speaker_roster(speakers)
    title = metadata.get("title") or "(untitled)"

    chunk_notes: list[str] = []
    for i, chunk in enumerate(chunks):
        note_path = paths.chunks_dir / f"chunk_{i:03d}.md"
        if _fresh(note_path, paths.speakers, force):
            chunk_notes.append(to_ascii(note_path.read_text(encoding="utf-8")))
            log.info("chunk %d/%d cached", i+1, len(chunks))
            continue
        log.info("chunk %d/%d (%d segs)...", i+1, len(chunks), len(chunk))
        chunk_text = _format_transcript(chunk)
        note = to_ascii(_chunk_summary(chunk_text, title, content_type, speaker_roster))
        write_ascii(note_path, note)
        chunk_notes.append(note)

    combined_notes = "\n\n---\n\n".join(
        f"### Chunk {i+1}\n\n{note}" for i, note in enumerate(chunk_notes)
    )

    # --- final synthesis passes (per-content-type recipe) ---
    duration_s = metadata.get("duration") or 0
    duration_str = _hms(duration_s) if duration_s else "unknown"

    for output_key, prompt_template, path_attr in recipe:
        out_path = getattr(paths, path_attr)
        if _fresh(out_path, paths.speakers, force):
            log.info("%s cached", out_path.name)
            continue
        prompt = prompt_template.format(
            title=title,
            channel=metadata.get("channel") or metadata.get("uploader") or "",
            duration=duration_str,
            speaker_roster=speaker_roster,
            notes=combined_notes,
        )
        log.info("generating %s...", out_path.name)
        result = ask(prompt, system=SYSTEM_SUMMARIZE, timeout=900)
        write_ascii(out_path, result + "\n")

    # Record what we generated so the next run can snapshot correctly
    _write_manifest(
        paths,
        content_type=content_type,
        outputs=[getattr(paths, attr).name for _, _, attr in recipe],
    )
    log.info("all outputs ready")
