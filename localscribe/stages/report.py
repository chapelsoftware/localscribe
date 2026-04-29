"""Stage 7: Render a single self-contained HTML report from the markdown outputs.

Combines tldr, per_speaker, debate_map, chronological, and the transcript
into one styled document with a table of contents, video metadata header,
collapsible transcript, and YouTube timestamp deep-links.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import markdown

from ..config import Paths, load_json
from ..text import to_ascii

log = logging.getLogger("report")


# ---------- helpers ----------

def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hms_to_seconds(hms: str) -> int:
    parts = [int(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return h * 3600 + m * 60 + s


_TIMESTAMP_BRACKET_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")


def _linkify_timestamps(md_text: str, youtube_url: str | None) -> str:
    """Turn [HH:MM:SS] tokens into markdown links to the video at that time.

    Works on the raw markdown before it's converted to HTML.
    """
    if not youtube_url or "youtube.com/watch" not in youtube_url:
        return md_text

    def repl(m: re.Match) -> str:
        hms = m.group(1)
        secs = _hms_to_seconds(hms)
        sep = "&" if "?" in youtube_url else "?"
        return f"[[{hms}]]({youtube_url}{sep}t={secs}s)"

    return _TIMESTAMP_BRACKET_RE.sub(repl, md_text)


def _md_to_html(md_text: str) -> str:
    """Render markdown with the extensions we use."""
    return markdown.markdown(
        md_text,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )


def _read_opt(path: Path) -> str:
    """Read a markdown file, or return empty string if missing."""
    if not path.exists() or path.stat().st_size == 0:
        return ""
    return path.read_text(encoding="utf-8")


# ---------- CSS ----------

CSS = """
:root {
  --bg: #fafaf7;
  --fg: #1a1a1a;
  --muted: #666;
  --accent: #2c5aa0;
  --border: #e0ddd4;
  --card: #ffffff;
  --code-bg: #f2efe6;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  font-family: Charter, Georgia, "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.6;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 0;
}
.container {
  max-width: 760px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}
header.report-header {
  border-bottom: 2px solid var(--border);
  padding-bottom: 1.5rem;
  margin-bottom: 2rem;
}
header.report-header h1 {
  font-size: 1.9rem;
  line-height: 1.2;
  margin: 0 0 0.6rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-weight: 700;
}
header.report-header .meta {
  color: var(--muted);
  font-size: 0.95rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header.report-header .meta a { color: var(--accent); text-decoration: none; }
header.report-header .meta a:hover { text-decoration: underline; }
.type-badge {
  display: inline-block;
  padding: 0.12rem 0.5rem;
  border: 1px solid var(--accent);
  border-radius: 3px;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
nav.toc {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.2rem;
  margin-bottom: 2.5rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 0.95rem;
}
nav.toc strong {
  display: block;
  margin-bottom: 0.4rem;
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
nav.toc ul { list-style: none; margin: 0; padding: 0; }
nav.toc li { display: inline-block; margin-right: 1.2rem; }
nav.toc a { color: var(--accent); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section.report-section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.8rem 2rem;
  margin-bottom: 2rem;
}
section.report-section h1:first-child,
section.report-section h2:first-child {
  margin-top: 0;
}
section.report-section h1 {
  font-size: 1.5rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin-top: 2rem;
}
section.report-section h2 {
  font-size: 1.2rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin-top: 1.8rem;
}
section.report-section h3 {
  font-size: 1.05rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--muted);
  margin-top: 1.4rem;
}
section.report-section ul, section.report-section ol { padding-left: 1.4rem; }
section.report-section li { margin-bottom: 0.3rem; }
section.report-section p { margin: 0.8rem 0; }
section.report-section strong { color: #000; }
section.report-section a { color: var(--accent); }
section.report-section code {
  background: var(--code-bg);
  padding: 0.1em 0.35em;
  border-radius: 3px;
  font-size: 0.9em;
}
details.transcript summary {
  cursor: pointer;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-weight: 600;
  font-size: 1.1rem;
  padding: 0.5rem 0;
  color: var(--accent);
}
details.transcript summary:hover { text-decoration: underline; }
details.transcript[open] summary { margin-bottom: 1rem; }
.transcript-body p {
  margin: 0.35rem 0;
  font-size: 0.94rem;
  line-height: 1.5;
}
.transcript-body a {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.85em;
  color: var(--muted);
  text-decoration: none;
  margin-right: 0.3em;
}
.transcript-body a:hover { color: var(--accent); text-decoration: underline; }
footer {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.85rem;
  color: var(--muted);
  text-align: center;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
@media print {
  body { background: white; }
  section.report-section, nav.toc { border: none; page-break-inside: avoid; }
  details.transcript { display: none; }
  nav.toc { display: none; }
}
"""


# ---------- transcript rendering ----------

_TRANSCRIPT_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+([^:]+):\s*(.*)$")


def _render_transcript_body(transcript_md: str, youtube_url: str | None) -> str:
    """Render the transcript section as HTML paragraphs with timestamp deep-links.

    We bypass the markdown library for this because the transcript is just
    line-by-line "[HH:MM:SS] Speaker: text" and we want tight control.
    """
    lines = transcript_md.splitlines()
    # Skip the header line and the source line at the top
    body_lines = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("#") or line.startswith("Source:"):
            continue
        body_lines.append(line)

    out = []
    for line in body_lines:
        m = _TRANSCRIPT_LINE_RE.match(line)
        if not m:
            out.append(f"<p>{html.escape(line)}</p>")
            continue
        hms, speaker, text = m.group(1), m.group(2), m.group(3)
        if youtube_url and "youtube.com/watch" in youtube_url:
            secs = _hms_to_seconds(hms)
            sep = "&" if "?" in youtube_url else "?"
            link = f'<a href="{html.escape(youtube_url)}{sep}t={secs}s">[{hms}]</a>'
        else:
            link = f'<span class="ts">[{hms}]</span>'
        out.append(
            f'<p>{link} <strong>{html.escape(speaker)}:</strong> '
            f'{html.escape(text)}</p>'
        )
    return "\n".join(out)


# ---------- section registry ----------

# All possible sections across all content types.
# (path_attr, slug, display_name)
SECTION_META: dict[str, tuple[str, str]] = {
    "tldr_md":          ("tldr",          "TL;DR"),
    "topics_md":        ("topics",        "Topics"),
    "per_speaker_md":   ("per-speaker",   "Per-speaker"),
    "debate_map_md":    ("debate-map",    "Debate map"),
    "chronological_md": ("chronological", "Chronological"),
    "highlights_md":    ("highlights",    "Highlights"),
    "key_points_md":    ("key-points",    "Key points"),
    "outline_md":       ("outline",       "Outline"),
    "quotes_md":        ("quotes",        "Notable quotes"),
}

# Section order per content type. Only files that exist on disk are included.
SECTIONS_BY_TYPE: dict[str, list[str]] = {
    "debate":     ["tldr_md", "per_speaker_md", "debate_map_md", "chronological_md"],
    "interview":  ["tldr_md", "topics_md", "per_speaker_md", "highlights_md", "chronological_md"],
    "panel":      ["tldr_md", "topics_md", "per_speaker_md", "highlights_md", "chronological_md"],
    "discussion": ["tldr_md", "topics_md", "per_speaker_md", "highlights_md", "chronological_md"],
    "monologue":  ["tldr_md", "key_points_md", "outline_md", "quotes_md"],
}


# ---------- main entry ----------


def _is_stale(paths: Paths) -> bool:
    """Return True if any upstream .md (or speakers.json) is newer than report.html."""
    if not paths.report_html.exists():
        return True
    report_mtime = paths.report_html.stat().st_mtime
    upstream_attrs = list(SECTION_META.keys()) + ["transcript_md"]
    for attr in upstream_attrs:
        p = getattr(paths, attr)
        if p.exists() and p.stat().st_mtime > report_mtime:
            return True
    # speakers.json also matters (content_type may have changed)
    if paths.speakers.exists() and paths.speakers.stat().st_mtime > report_mtime:
        return True
    return False


def run(paths: Paths, force: bool = False) -> Path:
    if not force and not _is_stale(paths):
        log.info("cached")
        return paths.report_html

    metadata = load_json(paths.metadata)
    speakers = load_json(paths.speakers) if paths.speakers.exists() else {}
    content_type_info = speakers.get("content_type") or {}
    content_type = content_type_info.get("type") or "discussion"
    if content_type not in SECTIONS_BY_TYPE:
        content_type = "discussion"

    title = metadata.get("title") or "(untitled)"
    channel = metadata.get("channel") or metadata.get("uploader") or ""
    duration = metadata.get("duration") or 0
    duration_str = _hms(duration) if duration else "unknown"
    url = metadata.get("webpage_url") or ""

    # Build the header meta line
    meta_parts = []
    if channel:
        meta_parts.append(html.escape(channel))
    if duration:
        meta_parts.append(duration_str)
    # Pretty content-type badge
    type_label = content_type.capitalize()
    meta_parts.append(f'<span class="type-badge">{type_label}</span>')
    if url:
        meta_parts.append(f'<a href="{html.escape(url)}">source</a>')
    meta_line = " &middot; ".join(meta_parts)

    # Pick sections for this content type, dropping any whose file is missing
    section_attrs = [
        attr for attr in SECTIONS_BY_TYPE[content_type]
        if getattr(paths, attr).exists()
        and getattr(paths, attr).stat().st_size > 0
    ]

    # TOC
    toc_items = []
    for attr in section_attrs:
        slug, name = SECTION_META[attr]
        toc_items.append(f'<li><a href="#{slug}">{name}</a></li>')
    toc_items.append('<li><a href="#transcript">Transcript</a></li>')
    toc_html = f"""<nav class="toc">
  <strong>Contents</strong>
  <ul>
    {"".join(toc_items)}
  </ul>
</nav>"""

    # Each main section
    section_htmls = []
    for attr in section_attrs:
        slug, _name = SECTION_META[attr]
        md_text = _read_opt(getattr(paths, attr))
        if not md_text.strip():
            continue
        md_text = _linkify_timestamps(md_text, url)
        body_html = _md_to_html(md_text)
        section_htmls.append(
            f'<section id="{slug}" class="report-section">\n{body_html}\n</section>'
        )

    # Transcript (collapsible)
    transcript_md = _read_opt(paths.transcript_md)
    if transcript_md.strip():
        transcript_body = _render_transcript_body(transcript_md, url)
        section_htmls.append(f"""<section id="transcript" class="report-section">
<details class="transcript">
  <summary>Full transcript ({duration_str})</summary>
  <div class="transcript-body">
{transcript_body}
  </div>
</details>
</section>""")

    # Assemble
    full_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<header class="report-header">
<h1>{html.escape(title)}</h1>
<div class="meta">{meta_line}</div>
</header>
{toc_html}
{"".join(section_htmls)}
<footer>
Generated by localscribe. Transcription: faster-whisper large-v3.
Diarization: pyannote 3.1. Summarization: Claude.
</footer>
</div>
</body>
</html>
"""
    # ASCII-clean the whole thing (belt & suspenders)
    full_html = to_ascii(full_html)
    paths.report_html.write_text(full_html, encoding="utf-8")
    size_kb = paths.report_html.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", paths.report_html.name, size_kb)
    return paths.report_html
