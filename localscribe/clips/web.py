"""Local web UI for reviewing and editing clip captions.

Runs a stdlib `http.server` on localhost. The UI shows each defined
clip's rendered mp4 alongside an editable caption list; the JS overlays
current-phrase captions onto the player as it plays so you can preview
edits before re-rendering.

The server is video-agnostic: at startup it scans `OUTPUT_ROOT` for
subdirectories that contain a `clips.json`, exposes them all via
`GET /api/videos`, and lets the UI switch between them.

Endpoints:
  GET  /                                  -- single-page UI
  GET  /api/videos                        -- list of (id, title, n_clips)
  GET  /api/clips/<video_id>              -- manifest for one video
  POST /api/clips/<video_id>/<clip_id>    -- save captions / re-burn
  GET  /video/<video_id>/<clip_id>.mp4              -- raw preview
  GET  /video/<video_id>/<clip_id>.captioned.mp4    -- burned final
  GET  /favicon.ico                       -- 204 (silences logs)
"""
from __future__ import annotations

import json
import logging
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..config import OUTPUT_ROOT, Paths
from . import captions as cap
from . import cut as cut_mod
from . import download as video_download
from . import manifest as mf
from . import segment_transcribe as seg_tx
from . import waveform as wf

log = logging.getLogger("clips.web")

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CLIP_ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")


# ---------------------------------------------------------------------------
# HTML / JS / CSS
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>The Clip Room</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    /* === Editing-booth palette ===
       Warm-cool neutrals + one desaturated cyan signal. The chrome
       recedes; video and captions are the foreground. */
    --bg: #0f1115;
    --surface: #161922;
    --panel: #161922;
    --elevated: #1d212c;
    --line: #2a2f3a;
    --line-bold: #353b48;
    --text: #d8dde5;
    --text-dim: #8b93a3;
    --text-mute: #5a6273;
    --accent: #5fb3c4;
    --accent-soft: #4a8d99;
    --accent-deep: #2e5b66;
    --saved: #6fa37a;
    --dirty: #c89668;
    --danger: #c4675f;

    --body: 'Inter', -apple-system, "Helvetica Neue", sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, "SF Mono", monospace;
    /* No serif display face — headings use the same Inter family at
       a heavier weight to keep the tool look. --display stays defined
       as an alias so old rules don't break. */
    --display: var(--body);

    /* Legacy aliases — older rules in this stylesheet still reference
       the Editorial palette names. Mapping them here means the
       per-rule edits stay small. */
    --gold: var(--accent);
    --gold-warm: var(--accent);
    --gold-soft: var(--text-dim);
    --gold-deep: var(--accent-deep);
    --cream: var(--text);
    --cream-dim: var(--text-dim);
    --rust: var(--danger);
    --rust-deep: #7a3f3a;
    --green: var(--saved);
    --ink: var(--bg);
    --paper: var(--surface);
    --red: var(--danger);
    --slate: var(--text-dim);
    --muted: var(--text-mute);
  }
  * { box-sizing: border-box; }
  ::selection { background: var(--gold); color: var(--bg); }

  body {
    margin: 0;
    min-height: 100vh;
    background-color: var(--bg);
    color: var(--text);
    font-family: var(--body);
    font-size: 13px;
    line-height: 1.5;
    font-feature-settings: "cv11", "ss01";  /* Inter: tabular alts */
    -webkit-font-smoothing: antialiased;
    display: grid;
    grid-template-columns: 320px 1fr;
  }

  /* === SIDEBAR === */
  #sidebar {
    grid-row: 1 / -1;
    position: sticky;
    top: 0;
    align-self: start;
    height: 100vh;
    display: grid;
    grid-template-rows: auto auto 1fr;
    background: var(--surface);
    border-right: 1px solid var(--line);
    overflow: hidden;
    z-index: 10;
  }
  .sidebar-mast {
    padding: 18px 20px;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  .sidebar-mast .wordmark {
    font-family: var(--body);
    font-weight: 600;
    font-size: 15px;
    letter-spacing: -0.005em;
    color: var(--text);
    line-height: 1;
  }
  .sidebar-mast .section {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1;
  }
  .sidebar-mast .section::before {
    content: "/ ";
    color: var(--text-mute);
    margin-right: 2px;
  }
  .sidebar-mast .count {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-mute);
    margin-left: auto;
    line-height: 1;
  }

  .sidebar-project {
    padding: 14px 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    border-bottom: 1px solid var(--line);
  }
  .sidebar-project label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-mute);
  }
  .sidebar-project select {
    appearance: none;
    -webkit-appearance: none;
    background-color: var(--bg);
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none' stroke='%23d4a64f' stroke-width='1.5'><path d='M1 1l4 4 4-4'/></svg>");
    background-repeat: no-repeat;
    background-position: right 12px center;
    color: var(--cream);
    border: 1px solid var(--line-bold);
    padding: 10px 32px 10px 14px;
    font-family: var(--body);
    font-size: 13px;
    font-weight: 500;
    border-radius: 1px;
    cursor: pointer;
    width: 100%;
    transition: border-color .2s, background-color .2s;
  }
  .sidebar-project select:hover { border-color: var(--gold); }
  .sidebar-project select:focus { outline: none; border-color: var(--gold); }
  .sidebar-project .meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--gold-soft);
    letter-spacing: 0.18em;
  }

  .sidebar-caps {
    overflow-y: auto;
    overflow-x: hidden;
    padding: 0 0 24px;
  }
  .sidebar-caps::-webkit-scrollbar { width: 6px; }
  .sidebar-caps::-webkit-scrollbar-track { background: transparent; }
  .sidebar-caps::-webkit-scrollbar-thumb { background: var(--line-bold); border-radius: 0; }
  .sidebar-caps .pane-header {
    padding: 18px 26px 14px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(95,179,196,.03), transparent);
  }
  .sidebar-caps .pane-eyebrow {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--gold-soft);
    margin-bottom: 6px;
  }
  .sidebar-caps .pane-title {
    font-family: var(--display);
    font-size: 18px;
    font-weight: 500;
    color: var(--cream);
    line-height: 1.2;
  }
  .sidebar-caps .pane-empty {
    padding: 38px 26px;
    color: var(--cream-dim);
    font-family: var(--display);
    font-size: 14px;
    text-align: center;
    line-height: 1.5;
  }
  .sidebar-caps .pane-add {
    width: calc(100% - 52px);
    margin: 14px 26px 0;
    background: transparent;
    color: var(--gold-soft);
    border: 1px dashed var(--gold-deep);
    padding: 10px 14px;
    border-radius: 1px;
    font-family: var(--body);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .15s;
  }
  .sidebar-caps .pane-add:hover {
    color: var(--gold);
    border-color: var(--gold);
    border-style: solid;
    background: rgba(95,179,196,.05);
  }
  .sidebar-caps .pane-resync {
    width: calc(100% - 52px);
    margin: 8px 26px 0;
    background: transparent;
    color: var(--cream-dim);
    border: 1px solid var(--line-bold);
    padding: 9px 14px;
    border-radius: 1px;
    font-family: var(--body);
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .15s;
  }
  .sidebar-caps .pane-resync:hover { color: var(--gold); border-color: var(--gold); }

  /* slim title bar above main content */
  header {
    grid-column: 2;
    position: relative;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    border-bottom: 1px solid var(--line);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }
  header .crumb { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  header .crumb-key { color: var(--text-mute); }
  header .crumb-val { color: var(--text); }
  header .legend {
    display: flex;
    gap: 14px;
    color: var(--text-mute);
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  header .legend kbd {
    font-family: var(--mono);
    background: var(--elevated);
    border: 1px solid var(--line-bold);
    padding: 1px 5px;
    color: var(--text-dim);
    border-radius: 2px;
    font-size: 10px;
    margin-right: 3px;
  }

  main {
    grid-column: 2;
    max-width: none;
    padding: 20px 28px 60px;
  }

  .empty {
    text-align: center;
    color: var(--text-dim);
    padding: 80px 20px;
    font-family: var(--body);
    font-size: 13px;
  }
  .empty code {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
  }

  /* === SECTION HEADINGS === */
  .section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px;
    font-family: var(--mono);
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  .section-heading .count {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-mute);
  }
  .section-heading .count::before { content: "("; }
  .section-heading .count::after { content: ")"; }

  /* === CLIP CARD === */
  .clip {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 2px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .clip-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--line);
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 16px;
  }
  .clip-header .title {
    font-family: var(--body);
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.25;
    letter-spacing: -0.005em;
  }
  .clip-header .id {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-mute);
  }
  .clip-header .left { display: flex; flex-direction: column; gap: 3px; }

  .clip-body {
    display: block;
    padding: 26px 32px 28px;
  }
  .player-col {
    display: block;
    width: 100%;
  }
  .clip.is-active {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent-deep);
  }
  .clip { cursor: pointer; }
  .clip-body, .clip-actions, .range-edit { cursor: auto; }

  /* === RANGE EDIT BAR === */
  .range-edit {
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--cream-dim);
    padding: 14px 16px;
    background: rgba(22,25,34,.45);
    border: 1px solid var(--line);
    border-radius: 1px;
    margin-top: 14px;
    transition: border-color .25s, background .25s;
  }
  .range-edit.dirty {
    border-color: var(--gold);
    background: rgba(95,179,196,.08);
  }
  .range-edit .row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1 1 100%;
    flex-wrap: wrap;
  }
  .range-edit label {
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold-soft);
    font-weight: 600;
    min-width: 36px;
  }
  .range-edit .nudge {
    background: transparent;
    color: var(--cream-dim);
    border: 1px solid var(--line-bold);
    border-radius: 1px;
    padding: 3px 8px;
    font-size: 10px;
    font-family: var(--mono);
    cursor: pointer;
    line-height: 1;
    transition: all .15s;
  }
  .range-edit .nudge:hover {
    color: var(--gold);
    border-color: var(--gold);
    background: rgba(95,179,196,.08);
  }
  .range-edit input[type=number] {
    width: 86px;
    background: var(--bg);
    color: var(--cream);
    border: 1px solid var(--line-bold);
    padding: 5px 8px;
    font-family: var(--mono);
    font-size: 11px;
    border-radius: 1px;
    font-feature-settings: "tnum";
  }
  .range-edit input[type=number]:focus {
    outline: none;
    border-color: var(--gold);
    background: rgba(95,179,196,.05);
  }
  .range-edit .duration {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--gold);
    font-weight: 600;
    font-feature-settings: "tnum";
    margin-left: auto;
  }
  .range-edit .set-here {
    background: var(--gold-deep);
    color: var(--bg);
    border: 0;
    padding: 4px 9px;
    border-radius: 1px;
    font-family: var(--body);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    cursor: pointer;
    margin-left: 2px;
    transition: background .15s;
  }
  .range-edit .set-here:hover { background: var(--gold); }

  /* === VIDEO PLAYER === */
  /* Width is derived from height via the 9:16 ratio so the frame
     stays vertical even though its parent (.player-col) fills the
     full main column. The scrubber lives in a sibling row below and
     spans the full main column independently. */
  .video-wrap {
    position: relative;
    background: #000;
    border: 1px solid var(--line-bold);
    border-radius: 1px;
    overflow: hidden;
    height: min(72vh, 1000px);
    width: calc(min(72vh, 1000px) * 9 / 16);
    max-width: 100%;
    margin: 0 auto;
    box-shadow:
      inset 0 0 0 1px rgba(95,179,196,.18),
      0 24px 40px -16px rgba(0,0,0,.7);
  }
  .video-wrap video {
    width: 100%; height: 100%;
    display: block;
    object-fit: cover;
  }
  .video-wrap.no-video {
    display: grid;
    place-items: center;
    color: var(--gold-soft);
    font-family: var(--display);
    font-size: 16px;
    height: min(72vh, 1000px);
    width: calc(min(72vh, 1000px) * 9 / 16);
    max-width: 100%;
    margin: 0 auto;
    border: 1px dashed var(--line-bold);
    background: var(--surface);
  }
  /* === CUSTOM PLAYER CONTROLS (premiere-style) ===
     Lives as a sibling under .video-wrap inside .player-col, so it
     spans the full main column even though the video itself is narrow. */
  .player-controls {
    display: grid;
    grid-template-columns: 36px 1fr auto auto;
    gap: 16px;
    align-items: center;
    width: 100%;
    margin-top: 14px;
    padding: 14px 18px;
    background: linear-gradient(180deg, var(--elevated), var(--surface));
    border: 1px solid var(--line-bold);
    border-radius: 1px;
    color: var(--cream);
    font-family: var(--mono);
    font-size: 10px;
  }
  .player-controls .trim-stack {
    position: relative;
    outline: none;
  }
  .player-controls .trim-stack:focus-visible .trim-track {
    box-shadow: 0 0 0 1px var(--gold);
  }
  .zoom-ctl {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--gold-soft);
  }
  .zoom-ctl .zoom-btn {
    width: 22px; height: 22px;
    background: rgba(15,17,21,.7);
    color: var(--gold);
    border: 1px solid var(--gold-deep);
    border-radius: 50%;
    font-family: var(--body);
    font-size: 12px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
    display: grid; place-items: center;
    transition: all .15s;
  }
  .zoom-ctl .zoom-btn:hover { background: var(--gold); color: var(--bg); }
  .zoom-ctl .zoom-btn:disabled { opacity: .3; cursor: not-allowed; }
  .zoom-ctl .zoom-level {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--gold);
    letter-spacing: 0.08em;
    min-width: 28px;
    text-align: center;
  }
  .player-controls .play-btn {
    width: 28px; height: 28px;
    border: 1px solid var(--gold-deep);
    background: rgba(15,17,21,.8);
    color: var(--gold);
    border-radius: 50%;
    cursor: pointer;
    display: grid;
    place-items: center;
    font-size: 11px;
    line-height: 1;
    padding: 0;
    transition: all .15s;
  }
  .player-controls .play-btn:hover {
    background: var(--gold);
    color: var(--bg);
    border-color: var(--gold);
    transform: scale(1.05);
  }
  .player-controls .time {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--gold);
    letter-spacing: 0.06em;
    font-feature-settings: "tnum";
    white-space: nowrap;
  }

  /* === TRIM TRACK (scrubber + waveform + markers + in/out) === */
  .trim-track {
    position: relative;
    height: 52px;
    background: rgba(0,0,0,.55);
    border: 1px solid var(--line-bold);
    border-radius: 1px;
    user-select: none;
    cursor: pointer;
    overflow: hidden;
  }
  .trim-track .waveform {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    opacity: .55;
  }
  .trim-track .waveform path {
    fill: var(--gold);
  }
  .trim-track .markers {
    position: absolute;
    left: 0; right: 0;
    pointer-events: none;
    height: 6px;
  }
  .trim-track .cap-markers { top: 0; }
  .trim-track .shot-markers { bottom: 0; }
  .trim-track .markers .tick {
    position: absolute;
    width: 1.5px;
    height: 100%;
    transform: translateX(-50%);
    pointer-events: auto;
    cursor: help;
  }
  .trim-track .cap-markers .tick { background: var(--gold-warm); opacity: .85; }
  .trim-track .cap-markers .tick:hover { opacity: 1; box-shadow: 0 0 6px var(--gold); }
  .trim-track .shot-markers .tick { background: var(--rust); width: 2px; opacity: .9; }
  .trim-track .shot-markers .tick:hover { opacity: 1; box-shadow: 0 0 6px var(--rust); }

  .trim-track .hover-tip {
    position: absolute;
    bottom: calc(100% + 6px);
    left: 0;
    transform: translateX(-50%);
    background: var(--bg);
    color: var(--cream);
    border: 1px solid var(--gold-deep);
    border-radius: 1px;
    padding: 4px 8px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.04em;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity .12s;
    z-index: 6;
    max-width: 320px;
    text-overflow: ellipsis;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0,0,0,.6);
  }
  .trim-track .hover-tip.show { opacity: 1; }
  .trim-track .hover-tip .ht-cap {
    display: block;
    color: var(--gold-warm);
    font-family: var(--display);
    font-size: 11px;
    margin-top: 2px;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .trim-track .snap-flash {
    position: absolute;
    top: -3px; bottom: -3px;
    width: 2px;
    background: var(--gold-warm);
    box-shadow: 0 0 10px var(--gold);
    pointer-events: none;
    opacity: 0;
    transform: translateX(-50%);
    transition: opacity .15s;
    z-index: 5;
  }
  .trim-track .snap-flash.show { opacity: 1; }
  .trim-track .clipped-l,
  .trim-track .clipped-r {
    position: absolute;
    top: 0; bottom: 0;
    background: repeating-linear-gradient(
      -45deg,
      rgba(0,0,0,.75) 0 5px,
      rgba(0,0,0,.55) 5px 10px
    );
    pointer-events: none;
  }
  .trim-track .clipped-l { left: 0; }
  .trim-track .clipped-r { right: 0; }
  .trim-track .selection {
    position: absolute;
    top: 0; bottom: 0;
    background: rgba(95,179,196,.22);
    border-left: 2px solid var(--gold);
    border-right: 2px solid var(--gold);
    pointer-events: none;
  }
  .trim-track .handle {
    position: absolute;
    top: -4px; bottom: -4px;
    width: 11px;
    background: var(--gold);
    border: 1px solid var(--gold-deep);
    border-radius: 1px;
    cursor: ew-resize;
    transform: translateX(-50%);
    box-shadow: 0 1px 6px rgba(0,0,0,.6);
    z-index: 4;
    transition: background .12s, box-shadow .12s;
  }
  .trim-track .handle:hover {
    background: var(--gold-warm);
    box-shadow: 0 1px 10px rgba(95,179,196,.4);
  }
  .trim-track .handle::before {
    content: "";
    position: absolute;
    left: 50%; top: 50%;
    transform: translate(-50%,-50%);
    width: 1px; height: 14px;
    background: var(--bg);
  }
  .trim-track .playhead {
    position: absolute;
    top: -6px; bottom: -6px;
    width: 1px;
    background: var(--rust);
    pointer-events: none;
    z-index: 5;
    box-shadow: 0 0 8px rgba(200,70,40,.85);
  }
  .trim-track .playhead::before {
    content: "";
    position: absolute;
    top: -4px; left: 50%;
    transform: translateX(-50%);
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid var(--rust);
  }
  .trim-track .ticks {
    position: absolute;
    inset: 0;
    pointer-events: none;
    color: var(--cream-dim);
    font-family: var(--mono);
    font-size: 8px;
    opacity: .55;
  }
  .trim-track .tick {
    position: absolute;
    top: 1px;
    transform: translateX(-50%);
  }
  .trim-track .tick::before {
    content: "";
    position: absolute;
    top: -1px; left: 50%;
    width: 1px; height: 4px;
    background: var(--cream-dim);
  }
  .burned-note {
    position: absolute;
    left: 0; right: 0; top: 0;
    background: rgba(15,17,21,.92);
    color: var(--gold);
    font-family: var(--mono);
    font-size: 9px;
    text-align: center;
    padding: 6px 8px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    pointer-events: none;
  }
  .preview-overlay {
    position: absolute;
    left: 8%; right: 8%;
    bottom: 18%;
    text-align: center;
    color: #fff;
    font-family: var(--body);
    font-weight: 800;
    font-size: 22px;
    line-height: 1.18;
    text-shadow: 0 0 10px rgba(0,0,0,.95), 0 2px 0 #000, 2px 0 0 #000, -2px 0 0 #000, 0 -2px 0 #000;
    pointer-events: none;
  }
  /* === CAPTIONS LIST (rendered in sidebar pane) === */
  .captions { display: contents; }
  .cap-row {
    display: grid;
    grid-template-columns: auto auto 1fr auto;
    gap: 8px 10px;
    padding: 11px 26px;
    align-items: center;
    border-bottom: 1px solid var(--line);
    transition: background .15s, border-color .15s;
  }
  .cap-row:last-child { border-bottom: none; }
  .cap-row.active {
    background: linear-gradient(90deg, rgba(95,179,196,.22), rgba(95,179,196,.04));
    box-shadow: inset 2px 0 0 var(--gold);
  }
  .cap-row input[type=text],
  .cap-row input[type=number] {
    background: transparent;
    border: 1px solid transparent;
    border-bottom: 1px solid var(--line);
    color: var(--cream);
    padding: 5px 7px;
    font-family: var(--body);
    font-size: 13px;
    border-radius: 0;
    transition: border-color .15s, background .15s;
  }
  .cap-row input[type=text] { width: 100%; min-width: 0; }
  .cap-row input[type=number] { width: 58px; }
  .cap-row input[type=text]:focus,
  .cap-row input[type=number]:focus {
    outline: none;
    border: 1px solid var(--gold);
    background: rgba(95,179,196,.05);
  }
  .cap-row input[type=number] {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--gold);
    font-feature-settings: "tnum";
    letter-spacing: 0.02em;
  }
  .cap-row .text {
    font-family: var(--display);
    font-size: 14px;
    color: var(--cream);
    line-height: 1.35;
  }
  .cap-row button {
    background: transparent;
    color: var(--cream-dim);
    border: 1px solid var(--line-bold);
    border-radius: 50%;
    width: 22px; height: 22px;
    cursor: pointer;
    font-size: 13px;
    line-height: 1;
    display: grid; place-items: center;
    transition: all .15s;
    padding: 0;
  }
  .cap-row button:hover {
    background: var(--rust);
    border-color: var(--rust);
    color: #fff;
  }

  /* === CLIP ACTIONS === */
  .clip-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    padding: 16px 32px;
    background: rgba(22,25,34,.55);
    border-top: 1px solid var(--line);
  }
  .clip-actions .status { flex: 1 1 auto; }
  .clip-actions button,
  .clip-actions a.btn {
    background: transparent;
    color: var(--cream-dim);
    border: 1px solid var(--line-bold);
    padding: 9px 16px;
    border-radius: 1px;
    cursor: pointer;
    font-family: var(--body);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-decoration: none;
    display: inline-block;
    transition: all .18s;
  }
  .clip-actions button:hover,
  .clip-actions a.btn:hover {
    color: var(--cream);
    border-color: var(--gold);
  }
  .clip-actions button:disabled { opacity: .35; cursor: progress; }

  .clip-actions .save {
    background: var(--gold);
    color: var(--bg);
    border-color: var(--gold);
    font-weight: 700;
  }
  .clip-actions .save:hover {
    background: var(--gold-warm);
    border-color: var(--gold-warm);
    color: var(--bg);
  }
  .clip-actions .archive { color: var(--cream-dim); }
  .clip-actions .archive:hover { color: var(--rust); border-color: var(--rust); }
  .clip-actions .add-row { color: var(--gold-soft); border-color: var(--gold-deep); }
  .clip-actions .add-row:hover { color: var(--gold); border-color: var(--gold); }
  .clip-actions a.download {
    background: transparent;
    color: var(--gold);
    border: 1px solid var(--gold);
  }
  .clip-actions a.download:hover {
    background: var(--gold);
    color: var(--bg);
  }
  .clip-actions .status {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.16em;
    color: var(--cream-dim);
    text-align: left;
    text-transform: uppercase;
  }
  .clip-actions .status.saved { color: var(--green); }
  .clip-actions .status.dirty { color: var(--rust); }

  /* === COMPLETED GRID === */
  .completed-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 20px;
  }
  .completed-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 1px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color .15s;
  }
  .completed-card:hover {
    border-color: var(--accent-deep);
  }
  .completed-card .video-wrap {
    /* Reset the full-size active-clip rules: in the completed grid the
       thumbnail must fit its grid cell, not fill 72vh. */
    height: auto;
    width: 100%;
    aspect-ratio: 9 / 16;
    margin: 0;
    border: none;
    border-radius: 0;
    box-shadow: none;
  }
  .completed-card .meta {
    padding: 13px 14px;
    border-top: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .completed-card .meta .title {
    font-family: var(--display);
    font-weight: 500;
    font-size: 16px;
    color: var(--cream);
    line-height: 1.2;
  }
  .completed-card .meta .id {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--gold-soft);
    letter-spacing: 0.12em;
  }
  .completed-card .actions {
    display: flex;
    gap: 6px;
    padding: 11px 14px;
    border-top: 1px solid var(--line);
  }
  .completed-card a.download,
  .completed-card button.restore {
    flex: 1;
    text-align: center;
    padding: 8px 10px;
    border-radius: 1px;
    font-family: var(--body);
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-decoration: none;
    cursor: pointer;
    border: 1px solid var(--line-bold);
    background: transparent;
    color: var(--cream-dim);
    transition: all .15s;
  }
  .completed-card a.download {
    color: var(--gold);
    border-color: var(--gold-deep);
  }
  .completed-card a.download:hover {
    background: var(--gold);
    color: var(--bg);
    border-color: var(--gold);
  }
  .completed-card button.restore:hover {
    color: var(--cream);
    border-color: var(--cream-dim);
  }
</style>
</head>
<body>
<aside id="sidebar">
  <div class="sidebar-mast">
    <span class="wordmark">localscribe</span>
    <span class="section">clips</span>
    <span class="count" id="clip-count"></span>
  </div>
  <div class="sidebar-project">
    <label for="video-select">Source</label>
    <select id="video-select"></select>
    <span class="meta" id="video-meta"></span>
  </div>
  <div id="sidebar-captions" class="sidebar-caps">
    <div class="pane-empty">Select a clip to edit its captions.</div>
  </div>
</aside>
<header>
  <div class="crumb">
    <span class="crumb-key">Now editing</span>
    <span class="crumb-val" id="crumb-clip">—</span>
  </div>
  <div class="legend">
    <span><kbd>Space</kbd> play/pause</span>
    <span><kbd>I</kbd>/<kbd>O</kbd> set in/out</span>
    <span><kbd>+</kbd>/<kbd>-</kbd>/<kbd>0</kbd> zoom</span>
  </div>
</header>
<main id="root"></main>

<script>
const fmt = (t) => {
  const tt = Math.max(0, t);
  const m = Math.floor(tt / 60), s = (tt - m * 60);
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
};

let manifest = null;
let videos = [];
let currentVideoId = null;
let activeClipId = null;
const dirty = new Set();

async function loadVideos() {
  const r = await fetch("/api/videos");
  videos = await r.json();
  const sel = document.getElementById("video-select");
  sel.innerHTML = "";
  let total = 0;
  for (const v of videos) {
    const opt = document.createElement("option");
    opt.value = v.id;
    const title = v.title || v.id;
    const n = v.n_clips;
    opt.textContent = `${title} (${n} clip${n === 1 ? "" : "s"})`;
    sel.appendChild(opt);
    total += n;
  }
  const cc = document.getElementById("clip-count");
  if (cc) cc.textContent = `${total} clip${total === 1 ? "" : "s"} · ${videos.length} source${videos.length === 1 ? "" : "s"}`;
  // Pick from URL hash if set, else first video
  const hashId = (location.hash.match(/#v=([^&]+)/) || [])[1];
  const initial = hashId && videos.find(v => v.id === hashId)
    ? hashId : (videos[0] && videos[0].id);
  if (initial) {
    sel.value = initial;
    await loadManifest(initial);
  } else {
    document.getElementById("root").innerHTML =
      '<div class="empty">No videos with a clips.json found in OUTPUT_ROOT. ' +
      'Run <code>localscribe-clips cut ...</code> to define a clip first.</div>';
  }
  sel.addEventListener("change", () => {
    if (dirty.size && !confirm("Unsaved caption edits will be lost. Continue?")) {
      sel.value = currentVideoId;
      return;
    }
    dirty.clear();
    loadManifest(sel.value);
  });
}

async function loadManifest(videoId) {
  currentVideoId = videoId;
  location.hash = "v=" + videoId;
  const r = await fetch(`/api/clips/${encodeURIComponent(videoId)}`);
  manifest = await r.json();
  const v = videos.find(v => v.id === videoId);
  document.getElementById("video-meta").textContent =
    v && v.title ? `${v.id}` : "";
  render();
}

function render() {
  const root = document.getElementById("root");
  root.innerHTML = "";
  const active = manifest.clips || [];
  const archived = manifest.archived || [];
  if (!active.length && !archived.length) {
    root.innerHTML = '<div class="empty">No clips defined. Run <code>localscribe-clips cut ...</code> first.</div>';
    setActiveClip(null);
    return;
  }

  // Active section
  if (active.length) {
    const head = document.createElement("div");
    head.className = "section-heading";
    head.innerHTML = `<span>Active</span><span class="count">${active.length}</span>`;
    root.appendChild(head);
    for (const clip of active) root.appendChild(renderClip(clip));
  }
  // Restore previous active clip if it still exists, else pick first.
  const stillActive = active.find(c => c.id === activeClipId);
  setActiveClip(stillActive ? stillActive.id : (active[0] && active[0].id) || null);

  // Completed section
  if (archived.length) {
    const head = document.createElement("div");
    head.className = "section-heading";
    head.innerHTML = `<span>Completed</span><span class="count">${archived.length}</span>`;
    root.appendChild(head);
    const grid = document.createElement("div");
    grid.className = "completed-grid";
    for (const clip of archived) grid.appendChild(renderCompleted(clip));
    root.appendChild(grid);
  }
}

function renderCompleted(clip) {
  const burnedPath = clip.render && clip.render.burned_path;
  const card = document.createElement("div");
  card.className = "completed-card";
  card.dataset.id = clip.id;
  // burned_path is `_completed/<filename>` for clips archived under the
  // universal layout, or `completed/<filename>` (relative to per-video
  // root) for legacy archived clips.
  let videoUrl = null;
  if (burnedPath) {
    if (burnedPath.startsWith("_completed/")) {
      videoUrl = "/completed/" + burnedPath.slice("_completed/".length);
    } else {
      videoUrl = `/file/${encodeURIComponent(currentVideoId)}/${burnedPath}`;
    }
  }
  card.innerHTML = `
    <div class="video-wrap ${videoUrl ? "" : "no-video"}">
      ${videoUrl
        ? `<video src="${videoUrl}" controls preload="metadata"></video>`
        : "no captioned mp4"}
    </div>
    <div class="meta">
      <div class="title">${escapeHtml(clip.title)}</div>
      <div class="id">${clip.id} · ${(clip.end - clip.start).toFixed(1)}s · archived ${clip.archived_at ? clip.archived_at.slice(0,10) : ""}</div>
    </div>
    <div class="actions">
      ${videoUrl ? `<a class="download" href="${videoUrl}" download>Download</a>` : ""}
      <button class="restore" data-action="restore">Restore</button>
    </div>
  `;
  card.querySelector('[data-action=restore]')
    .addEventListener("click", () => restoreClip(clip.id));
  return card;
}

function setActiveClip(clipId) {
  // No-op if already active — avoids stomping on caption inputs the
  // user is typing in.
  if (activeClipId === clipId) return;
  activeClipId = clipId;
  // Highlight the active card in the main column.
  document.querySelectorAll("section.clip").forEach(el => {
    el.classList.toggle("is-active", el.dataset.id === clipId);
  });
  // Update the breadcrumb in the slim header.
  const crumb = document.getElementById("crumb-clip");
  const clip = clipId && manifest && (manifest.clips || []).find(c => c.id === clipId);
  if (crumb) crumb.textContent = clip ? clip.title : "—";
  // Repaint the sidebar caption pane.
  if (clip) renderSidebarCaptions(clip);
  else {
    const pane = document.getElementById("sidebar-captions");
    if (pane) pane.innerHTML = '<div class="pane-empty">Select a clip to edit its captions.</div>';
  }
}

function renderSidebarCaptions(clip) {
  const pane = document.getElementById("sidebar-captions");
  if (!pane) return;
  pane.innerHTML = "";
  const head = document.createElement("div");
  head.className = "pane-header";
  head.innerHTML =
    `<div class="pane-eyebrow">Captions · ${(clip.captions || []).length}</div>` +
    `<div class="pane-title">${escapeHtml(clip.title)}</div>`;
  pane.appendChild(head);
  if (!(clip.captions || []).length) {
    const empty = document.createElement("div");
    empty.className = "pane-empty";
    empty.textContent = "No captions on this clip yet.";
    pane.appendChild(empty);
  } else {
    for (let i = 0; i < clip.captions.length; i++) {
      pane.appendChild(renderRow(clip, i));
    }
  }
  const addBtn = document.createElement("button");
  addBtn.className = "pane-add";
  addBtn.textContent = "+ Add caption";
  addBtn.addEventListener("click", () => addRow(clip.id));
  pane.appendChild(addBtn);
  const resyncBtn = document.createElement("button");
  resyncBtn.className = "pane-resync";
  resyncBtn.textContent = "Resync from transcript";
  resyncBtn.title = "Regenerate captions from word-level transcript";
  resyncBtn.addEventListener("click", () => resyncClip(clip.id));
  pane.appendChild(resyncBtn);
}

function renderClip(clip) {
  const wrap = document.createElement("section");
  wrap.className = "clip";
  wrap.dataset.id = clip.id;

  // Player loads the cut clip (not the source). The trim track lets you
  // shrink the clip from inside; for extending past the cut boundaries
  // use the numeric inputs / nudge buttons -- the player doesn't have
  // those frames to seek through.
  const renderPath = clip.render && clip.render.path;
  const renderedAt = clip.render && clip.render.rendered_at;
  // Cache-bust on rendered_at so a re-cut shows fresh bytes immediately.
  const cb = renderedAt ? `?v=${encodeURIComponent(renderedAt)}` : "";
  const videoSrc = renderPath
    ? `/video/${encodeURIComponent(currentVideoId)}/${clip.id}.mp4${cb}`
    : null;
  const burned = !!(clip.render && clip.render.captions_burned);
  const stale = !!(clip.render && clip.render.captions_stale);

  // If captions are burned into the video, we suppress the JS overlay
  // (otherwise you'd see them twice). The captions list is still
  // editable, but a re-render is required to preview changes.
  const overlayHtml = videoSrc && !burned
    ? '<div class="preview-overlay"></div>' : '';
  const burnedNote = burned
    ? `<div class="burned-note">${stale ? '⚠ Captions changed — re-render to update.' : 'Captions are burned into this preview. Edits won\\u2019t show until you re-render.'}</div>`
    : '';

  const burnedPath = clip.render && clip.render.burned_path;
  const burnedAt = clip.render && clip.render.burned_at;
  const burnCb = burnedAt ? `?v=${encodeURIComponent(burnedAt)}` : "";
  const downloadHtml = burnedPath
    ? `<a class="btn download" href="/video/${encodeURIComponent(currentVideoId)}/${clip.id}.captioned.mp4${burnCb}" download>Download captioned</a>`
    : '';

  wrap.innerHTML = `
    <div class="clip-header">
      <div class="left">
        <div class="title">${escapeHtml(clip.title)}</div>
        <div class="id">${clip.id} · source: ${fmt(clip.start)} – ${fmt(clip.end)}</div>
      </div>
    </div>
    <div class="clip-body">
      <div class="player-col">
        <div class="video-wrap ${videoSrc ? "" : "no-video"}">
          ${videoSrc
            ? `<video src="${videoSrc}" preload="metadata" data-player tabindex="0"></video>${overlayHtml}`
            : "Not rendered yet — run <code>render</code> after saving"}
          ${burnedNote}
        </div>
        ${videoSrc ? `
        <div class="player-controls">
          <button class="play-btn" data-play title="Play / pause selection (Space)">▶</button>
          <div class="trim-stack" tabindex="0" data-stack>
            <div class="trim-track" data-trim>
              <svg class="waveform" data-waveform preserveAspectRatio="none" viewBox="0 0 1000 100"></svg>
              <div class="markers cap-markers" data-cap-markers></div>
              <div class="markers shot-markers" data-shot-markers></div>
              <div class="clipped-l" data-clipped-l></div>
              <div class="clipped-r" data-clipped-r></div>
              <div class="selection" data-selection></div>
              <div class="playhead" data-playhead></div>
              <div class="handle start" data-handle="start" title="Drag to set start (I)"></div>
              <div class="handle end" data-handle="end" title="Drag to set end (O)"></div>
              <div class="hover-tip" data-hover-tip></div>
              <div class="snap-flash" data-snap-flash></div>
            </div>
          </div>
          <span class="time" data-time>0:00 / 0:00</span>
          <div class="zoom-ctl">
            <button class="zoom-btn" data-zoom-out title="Zoom out (−)">−</button>
            <span class="zoom-level" data-zoom-level>1×</span>
            <button class="zoom-btn" data-zoom-in title="Zoom in (+)">+</button>
            <button class="zoom-btn" data-zoom-fit title="Fit (0)">↔</button>
          </div>
        </div>` : ""}
        <div class="range-edit" data-range>
          <div class="row">
            <label>Start</label>
            <button class="nudge" data-nudge-start="-0.5">−.5</button>
            <button class="nudge" data-nudge-start="-0.1">−.1</button>
            <input type="number" step="0.05" data-field="start" value="${clip.start.toFixed(2)}">
            <button class="nudge" data-nudge-start="0.1">+.1</button>
            <button class="nudge" data-nudge-start="0.5">+.5</button>
            <button class="set-here" data-set-start>⤓ Set @ playhead</button>
          </div>
          <div class="row">
            <label>End</label>
            <button class="nudge" data-nudge-end="-0.5">−.5</button>
            <button class="nudge" data-nudge-end="-0.1">−.1</button>
            <input type="number" step="0.05" data-field="end" value="${clip.end.toFixed(2)}">
            <button class="nudge" data-nudge-end="0.1">+.1</button>
            <button class="nudge" data-nudge-end="0.5">+.5</button>
            <button class="set-here" data-set-end>⤒ Set @ playhead</button>
            <span class="duration" data-duration>${(clip.end - clip.start).toFixed(2)}s</span>
          </div>
        </div>
      </div>
    </div>
    <div class="clip-actions">
      <span class="status">${burnedPath ? 'captioned mp4 ready' : 'no captioned mp4 yet'}</span>
      ${downloadHtml}
      <button class="archive" data-action="archive">Archive</button>
      <button class="save" data-action="save">Save &amp; render</button>
    </div>
  `;

  wrap.querySelector('[data-action=save]').addEventListener("click", (e) => {
    e.stopPropagation(); saveClip(clip.id);
  });
  wrap.querySelector('[data-action=archive]').addEventListener("click", (e) => {
    e.stopPropagation(); archiveClip(clip.id);
  });
  // Click anywhere on the clip card → make it the active clip so its
  // captions show in the sidebar.
  wrap.addEventListener("click", () => setActiveClip(clip.id));

  // Wire up start/end nudge buttons + dirty-tracking on range edits.
  const rangeEl = wrap.querySelector('[data-range]');
  const startInput = rangeEl.querySelector('[data-field=start]');
  const endInput = rangeEl.querySelector('[data-field=end]');
  const durationEl = rangeEl.querySelector('[data-duration]');
  const refreshDuration = () => {
    const s = parseFloat(startInput.value);
    const e = parseFloat(endInput.value);
    durationEl.textContent = isFinite(s) && isFinite(e) ? `${(e-s).toFixed(2)}s` : "—";
  };
  const markRangeDirty = () => {
    const startChanged = Math.abs(parseFloat(startInput.value) - clip.start) > 0.005;
    const endChanged = Math.abs(parseFloat(endInput.value) - clip.end) > 0.005;
    rangeEl.classList.toggle("dirty", startChanged || endChanged);
    if (startChanged || endChanged) markDirty(clip.id);
  };
  startInput.addEventListener("input", () => { refreshDuration(); markRangeDirty(); });
  endInput.addEventListener("input", () => { refreshDuration(); markRangeDirty(); });
  rangeEl.querySelectorAll('[data-nudge-start]').forEach(btn => {
    btn.addEventListener("click", () => {
      startInput.value = (parseFloat(startInput.value) + parseFloat(btn.dataset.nudgeStart)).toFixed(2);
      refreshDuration(); markRangeDirty();
      startInput.dispatchEvent(new Event("input"));
    });
  });
  rangeEl.querySelectorAll('[data-nudge-end]').forEach(btn => {
    btn.addEventListener("click", () => {
      endInput.value = (parseFloat(endInput.value) + parseFloat(btn.dataset.nudgeEnd)).toFixed(2);
      refreshDuration(); markRangeDirty();
      endInput.dispatchEvent(new Event("input"));
    });
  });
  // The video element plays the cut clip. Its currentTime is
  // clip-relative (0 = start of clip). The "Set @ playhead" buttons
  // translate that back to source coords by adding clip.start.
  const playerVideo = wrap.querySelector("video");
  const setStartBtn = rangeEl.querySelector('[data-set-start]');
  const setEndBtn = rangeEl.querySelector('[data-set-end]');
  if (playerVideo && setStartBtn) {
    setStartBtn.addEventListener("click", () => {
      const sourceTime = clip.start + playerVideo.currentTime;
      startInput.value = sourceTime.toFixed(2);
      refreshDuration(); markRangeDirty();
      // Programmatic .value = ... does NOT trigger input events; fire
      // it manually so the trim track redraws (syncSelection lives in
      // the trim block's scope).
      startInput.dispatchEvent(new Event("input"));
    });
  }
  if (playerVideo && setEndBtn) {
    setEndBtn.addEventListener("click", () => {
      const sourceTime = clip.start + playerVideo.currentTime;
      endInput.value = sourceTime.toFixed(2);
      refreshDuration(); markRangeDirty();
      endInput.dispatchEvent(new Event("input"));
    });
  }

  // ===================== Scrubber / trim / waveform / zoom =====================
  // The trim track is also the playback scrubber. Internal time is
  // clip-relative (0..clipDur); the in/out inputs hold source-relative
  // times for backend compatibility. A viewport (viewStart..viewEnd in
  // clip-rel seconds) lets the timeline zoom and pan; pctOf() always
  // maps relative to the current viewport.
  const trim = wrap.querySelector('[data-trim]');
  const stack = wrap.querySelector('[data-stack]');
  const playBtn = wrap.querySelector('[data-play]');
  const timeEl = wrap.querySelector('[data-time]');
  if (trim && playerVideo) {
    const sel = trim.querySelector('[data-selection]');
    const playhead = trim.querySelector('[data-playhead]');
    const clippedL = trim.querySelector('[data-clipped-l]');
    const clippedR = trim.querySelector('[data-clipped-r]');
    const startHandle = trim.querySelector('[data-handle="start"]');
    const endHandle = trim.querySelector('[data-handle="end"]');
    const waveformEl = trim.querySelector('[data-waveform]');
    const capMarkers = trim.querySelector('[data-cap-markers]');
    const shotMarkers = trim.querySelector('[data-shot-markers]');
    const hoverTip = trim.querySelector('[data-hover-tip]');
    const snapFlash = trim.querySelector('[data-snap-flash]');
    const zoomLevel = wrap.querySelector('[data-zoom-level]');
    const zoomInBtn = wrap.querySelector('[data-zoom-in]');
    const zoomOutBtn = wrap.querySelector('[data-zoom-out]');
    const zoomFitBtn = wrap.querySelector('[data-zoom-fit]');

    const trackStart = clip.start;
    const trackEnd = clip.end;
    const clipDur = Math.max(0.001, trackEnd - trackStart);

    let viewStart = 0;        // clip-relative seconds
    let viewEnd = clipDur;
    let peaks = null;
    let fps = 30;
    const SHOT_TIMES = (clip.face_track || []).map(s => s.t).filter(t => t > 0.05 && t < clipDur - 0.05);
    const CAP_TIMES = (clip.captions || []).map(c => c.start).filter(t => t > 0.05 && t < clipDur - 0.05);

    const inRel = () => parseFloat(startInput.value) - trackStart;
    const outRel = () => parseFloat(endInput.value) - trackStart;
    const pctOf = (relT) => ((relT - viewStart) / (viewEnd - viewStart)) * 100;
    const relAtPct = (p) => viewStart + (p / 100) * (viewEnd - viewStart);

    function fmtMmss(t) {
      if (!isFinite(t) || t < 0) t = 0;
      const m = Math.floor(t / 60);
      const s = Math.floor(t - m * 60);
      return `${m}:${s.toString().padStart(2, "0")}`;
    }

    function renderWaveform() {
      if (!peaks || !peaks.length) return;
      const N = peaks.length;
      const startIdx = Math.max(0, Math.floor((viewStart / clipDur) * N));
      const endIdx = Math.min(N, Math.ceil((viewEnd / clipDur) * N));
      const visible = Math.max(1, endIdx - startIdx);
      let d = "";
      const barW = Math.max(0.6, 1000 / visible);
      for (let i = startIdx; i < endIdx; i++) {
        const t = (i / N) * clipDur;
        const x = pctOf(t) * 10;
        const h = Math.max(1, peaks[i] * 80);
        const y = 50 - h / 2;
        d += `M${x.toFixed(2)},${y.toFixed(2)}h${barW.toFixed(2)}v${h.toFixed(2)}h-${barW.toFixed(2)}z`;
      }
      let path = waveformEl.querySelector('path');
      if (!path) {
        path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        waveformEl.appendChild(path);
      }
      path.setAttribute('d', d);
    }

    function renderMarkers() {
      capMarkers.innerHTML = "";
      shotMarkers.innerHTML = "";
      for (const c of (clip.captions || [])) {
        const t = c.start;
        if (t < viewStart - 0.01 || t > viewEnd + 0.01) continue;
        const tick = document.createElement("div");
        tick.className = "tick";
        tick.style.left = pctOf(t) + "%";
        tick.title = `caption @ ${fmtMmss(t)}: ${(c.text || "").slice(0, 100)}`;
        capMarkers.appendChild(tick);
      }
      for (const s of (clip.face_track || [])) {
        const t = s.t;
        if (t < viewStart - 0.01 || t > viewEnd + 0.01) continue;
        const tick = document.createElement("div");
        tick.className = "tick";
        tick.style.left = pctOf(t) + "%";
        tick.title = `shot start @ ${fmtMmss(t)} · zoom ${(s.zoom||1).toFixed(2)}× · focus (${(s.focus_x||0).toFixed(2)},${(s.focus_y||0).toFixed(2)})`;
        shotMarkers.appendChild(tick);
      }
    }

    function syncSelection() {
      const sP = pctOf(inRel());
      const eP = pctOf(outRel());
      // Clip handle positions to the visible viewport (-3..103 lets
      // them edge slightly off-screen rather than disappearing entirely)
      const sClamp = Math.max(-3, Math.min(103, sP));
      const eClamp = Math.max(-3, Math.min(103, eP));
      sel.style.left = sClamp + "%";
      sel.style.width = Math.max(0, eClamp - sClamp) + "%";
      startHandle.style.left = sClamp + "%";
      endHandle.style.left = eClamp + "%";
      startHandle.style.opacity = (sP >= -3 && sP <= 103) ? 1 : 0;
      endHandle.style.opacity = (eP >= -3 && eP <= 103) ? 1 : 0;
      clippedL.style.width = Math.max(0, Math.min(100, sP)) + "%";
      clippedR.style.width = Math.max(0, Math.min(100, 100 - eP)) + "%";
    }

    function syncTime() {
      const t = playerVideo.currentTime;
      const dur = Math.max(0, outRel() - inRel());
      const elapsed = Math.max(0, Math.min(dur, t - inRel()));
      const tf = Math.round(elapsed * fps);
      const totalF = Math.round(dur * fps);
      timeEl.textContent = `${fmtMmss(elapsed)} / ${fmtMmss(dur)} · f${tf}/${totalF}`;
    }

    function setView(s, e) {
      const minSpan = 0.4;
      if (e - s < minSpan) {
        const c = (s + e) / 2;
        s = Math.max(0, c - minSpan / 2);
        e = Math.min(clipDur, s + minSpan);
        s = e - minSpan;
      }
      viewStart = Math.max(0, s);
      viewEnd = Math.min(clipDur, e);
      const z = clipDur / (viewEnd - viewStart);
      zoomLevel.textContent = z < 10 ? `${z.toFixed(1)}×` : `${Math.round(z)}×`;
      renderWaveform();
      renderMarkers();
      syncSelection();
      playhead.style.left = pctOf(playerVideo.currentTime) + "%";
    }

    function zoomBy(factor, pivotRel) {
      const span = viewEnd - viewStart;
      const newSpan = Math.max(0.4, Math.min(clipDur, span / factor));
      if (pivotRel == null) pivotRel = (viewStart + viewEnd) / 2;
      let s = pivotRel - (pivotRel - viewStart) * (newSpan / span);
      let e = s + newSpan;
      if (s < 0) { e -= s; s = 0; }
      if (e > clipDur) { s -= (e - clipDur); e = clipDur; }
      setView(s, e);
    }

    // ---- Snap targets ----
    function snapCandidate(t, snapPx) {
      // Convert pixel threshold to clip-rel seconds
      const rect = trim.getBoundingClientRect();
      const tol = (snapPx / Math.max(1, rect.width)) * (viewEnd - viewStart);
      const targets = [
        ...SHOT_TIMES.map(x => ({t: x, w: 0.0})),         // strongest
        ...CAP_TIMES.map(x => ({t: x, w: 0.4})),
      ];
      // Whole-second targets within view
      for (let s = Math.ceil(viewStart); s <= Math.floor(viewEnd); s++) {
        targets.push({t: s, w: 0.7});
      }
      let best = null, bestD = Infinity;
      for (const tgt of targets) {
        const d = Math.abs(tgt.t - t) + tgt.w * 0.001;
        if (d < bestD && Math.abs(tgt.t - t) <= tol) {
          best = tgt.t;
          bestD = d;
        }
      }
      return best;
    }

    function flashSnap(relT) {
      snapFlash.style.left = pctOf(relT) + "%";
      snapFlash.classList.add("show");
      clearTimeout(snapFlash._t);
      snapFlash._t = setTimeout(() => snapFlash.classList.remove("show"), 240);
    }

    // ---- Initial draw ----
    setView(0, clipDur);
    syncTime();

    // ---- Fetch waveform + probe ----
    fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clip.id)}/probe`)
      .then(r => r.ok ? r.json() : null).then(d => {
        if (d && d.fps) { fps = d.fps; syncTime(); }
      }).catch(() => {});
    fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clip.id)}/waveform`)
      .then(r => r.ok ? r.json() : null).then(d => {
        if (d && Array.isArray(d.peaks)) {
          peaks = d.peaks;
          renderWaveform();
        }
      }).catch(() => {});

    // ---- Player events ----
    playerVideo.addEventListener("loadedmetadata", () => {
      try { playerVideo.currentTime = inRel(); } catch (e) {}
      syncTime();
    });
    playerVideo.addEventListener("timeupdate", () => {
      const t = playerVideo.currentTime;
      playhead.style.left = pctOf(t) + "%";
      syncTime();
      if (!playerVideo.paused && t >= outRel()) playerVideo.pause();
    });
    playerVideo.addEventListener("play", () => {
      playBtn.textContent = "❚❚";
      const t = playerVideo.currentTime;
      if (t < inRel() - 0.05 || t > outRel() + 0.05) {
        try { playerVideo.currentTime = inRel(); } catch (e) {}
      }
    });
    playerVideo.addEventListener("pause", () => { playBtn.textContent = "▶"; });

    playBtn.addEventListener("click", () => {
      if (playerVideo.paused) {
        const t = playerVideo.currentTime;
        if (t >= outRel() - 0.01 || t < inRel() - 0.01) {
          try { playerVideo.currentTime = inRel(); } catch (e) {}
        }
        playerVideo.play();
      } else playerVideo.pause();
    });

    // ---- Click-to-seek ----
    trim.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-handle]")) return;
      if (ev.target.closest(".markers .tick")) return;  // marker hover only
      const rect = trim.getBoundingClientRect();
      const p = ((ev.clientX - rect.left) / rect.width) * 100;
      const relT = Math.max(0, Math.min(clipDur, relAtPct(p)));
      playerVideo.currentTime = Math.max(inRel(), Math.min(outRel(), relT));
      stack.focus();
    });

    // ---- Hover tip ----
    trim.addEventListener("mousemove", (ev) => {
      const rect = trim.getBoundingClientRect();
      const p = ((ev.clientX - rect.left) / rect.width) * 100;
      const t = relAtPct(p);
      // Find caption at this time
      let capText = "";
      for (const c of (clip.captions || [])) {
        if (t >= c.start && t <= c.end) { capText = c.text; break; }
      }
      hoverTip.style.left = ((ev.clientX - rect.left)) + "px";
      hoverTip.innerHTML = `${fmtMmss(t)}` + (capText ? `<span class="ht-cap">${escapeHtml(capText)}</span>` : "");
      hoverTip.classList.add("show");
    });
    trim.addEventListener("mouseleave", () => { hoverTip.classList.remove("show"); });

    // ---- Drag handles (with snap) ----
    function startDrag(handleName) {
      return (downEv) => {
        downEv.preventDefault();
        stack.focus();
        const rect = trim.getBoundingClientRect();
        const onMove = (ev) => {
          const x = ('touches' in ev ? ev.touches[0].clientX : ev.clientX) - rect.left;
          const p = Math.max(0, Math.min(100, (x / rect.width) * 100));
          let relT = relAtPct(p);
          // Snap unless shift held
          if (!ev.shiftKey) {
            const snap = snapCandidate(relT, 6);
            if (snap !== null) {
              relT = snap;
              flashSnap(relT);
            }
          }
          const srcT = trackStart + relT;
          if (handleName === "start") {
            const e = parseFloat(endInput.value);
            startInput.value = Math.max(trackStart, Math.min(srcT, e - 0.1)).toFixed(2);
          } else {
            const s = parseFloat(startInput.value);
            endInput.value = Math.min(trackEnd, Math.max(srcT, s + 0.1)).toFixed(2);
          }
          refreshDuration();
          syncSelection();
          syncTime();
          markRangeDirty();
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          window.removeEventListener("touchmove", onMove);
          window.removeEventListener("touchend", onUp);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        window.addEventListener("touchmove", onMove, {passive: false});
        window.addEventListener("touchend", onUp);
      };
    }
    startHandle.addEventListener("mousedown", startDrag("start"));
    startHandle.addEventListener("touchstart", startDrag("start"), {passive: false});
    endHandle.addEventListener("mousedown", startDrag("end"));
    endHandle.addEventListener("touchstart", startDrag("end"), {passive: false});

    // ---- Zoom controls ----
    zoomInBtn.addEventListener("click", () => zoomBy(1.6, playerVideo.currentTime));
    zoomOutBtn.addEventListener("click", () => zoomBy(1 / 1.6, playerVideo.currentTime));
    zoomFitBtn.addEventListener("click", () => setView(0, clipDur));
    trim.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const rect = trim.getBoundingClientRect();
      const p = ((ev.clientX - rect.left) / rect.width) * 100;
      const pivot = relAtPct(p);
      if (ev.shiftKey) {
        // Pan
        const span = viewEnd - viewStart;
        const dx = ev.deltaY * 0.002 * span;  // pan speed
        let s = viewStart + dx, e = viewEnd + dx;
        if (s < 0) { e -= s; s = 0; }
        if (e > clipDur) { s -= (e - clipDur); e = clipDur; }
        setView(s, e);
      } else {
        zoomBy(ev.deltaY < 0 ? 1.25 : 1 / 1.25, pivot);
      }
    }, {passive: false});

    // ---- Keyboard shortcuts (only when stack has focus) ----
    stack.addEventListener("keydown", (ev) => {
      const frameDt = 1 / fps;
      const code = ev.key;
      // Ignore if the user is typing in an input nested somewhere
      if (ev.target.matches('input, textarea')) return;
      if (code === " ") {
        ev.preventDefault(); playBtn.click();
      } else if (code === "ArrowLeft") {
        ev.preventDefault();
        const step = ev.shiftKey ? 1.0 : frameDt;
        playerVideo.currentTime = Math.max(0, playerVideo.currentTime - step);
      } else if (code === "ArrowRight") {
        ev.preventDefault();
        const step = ev.shiftKey ? 1.0 : frameDt;
        playerVideo.currentTime = Math.min(clipDur, playerVideo.currentTime + step);
      } else if (code === "j" || code === "J") {
        ev.preventDefault();
        playerVideo.currentTime = Math.max(0, playerVideo.currentTime - 1.0);
      } else if (code === "k" || code === "K") {
        ev.preventDefault(); playerVideo.pause();
      } else if (code === "l" || code === "L") {
        ev.preventDefault();
        playerVideo.currentTime = Math.min(clipDur, playerVideo.currentTime + 1.0);
      } else if (code === "i" || code === "I") {
        ev.preventDefault();
        startInput.value = (trackStart + playerVideo.currentTime).toFixed(2);
        refreshDuration(); markRangeDirty();
        startInput.dispatchEvent(new Event("input"));
      } else if (code === "o" || code === "O") {
        ev.preventDefault();
        endInput.value = (trackStart + playerVideo.currentTime).toFixed(2);
        refreshDuration(); markRangeDirty();
        endInput.dispatchEvent(new Event("input"));
      } else if (code === "q" || code === "Q") {
        ev.preventDefault();
        playerVideo.currentTime = Math.max(0, inRel());
      } else if (code === "w" || code === "W") {
        ev.preventDefault();
        playerVideo.currentTime = Math.max(0, outRel() - 0.05);
      } else if (code === "[") {
        ev.preventDefault();
        const cur = parseFloat(startInput.value);
        startInput.value = Math.max(trackStart,
          cur + (ev.shiftKey ? frameDt : -frameDt)).toFixed(2);
        refreshDuration(); markRangeDirty();
        startInput.dispatchEvent(new Event("input"));
      } else if (code === "]") {
        ev.preventDefault();
        const cur = parseFloat(endInput.value);
        endInput.value = Math.min(trackEnd,
          cur + (ev.shiftKey ? -frameDt : frameDt)).toFixed(2);
        refreshDuration(); markRangeDirty();
        endInput.dispatchEvent(new Event("input"));
      } else if (code === "+" || code === "=") {
        ev.preventDefault(); zoomBy(1.6, playerVideo.currentTime);
      } else if (code === "-" || code === "_") {
        ev.preventDefault(); zoomBy(1 / 1.6, playerVideo.currentTime);
      } else if (code === "0") {
        ev.preventDefault(); setView(0, clipDur);
      }
    });

    // Re-sync visuals when numeric inputs change
    startInput.addEventListener("input", () => { syncSelection(); syncTime(); });
    endInput.addEventListener("input", () => { syncSelection(); syncTime(); });
  }

  const video = wrap.querySelector("video");
  if (video) {
    const overlay = wrap.querySelector(".preview-overlay");  // null when burned
    video.addEventListener("timeupdate", () => updateOverlay(clip, video, overlay));
  }
  return wrap;
}

// renderRow renders one caption row into the sidebar pane. Inputs
// mutate the in-memory caption (clip.captions[idx]) directly, which is
// the source of truth that saveClip reads from. The clip card has no
// DOM caption rows anymore.
function renderRow(clip, idx) {
  const cap = clip.captions[idx];
  const row = document.createElement("div");
  row.className = "cap-row";
  row.dataset.idx = String(idx);
  row.innerHTML = `
    <input type="number" step="0.05" value="${cap.start.toFixed(2)}" data-field="start">
    <input type="number" step="0.05" value="${cap.end.toFixed(2)}" data-field="end">
    <input type="text" value="${escapeAttr(cap.text)}" data-field="text" class="text">
    <button data-action="del" title="Delete">×</button>
  `;
  const sInp = row.querySelector('[data-field=start]');
  const eInp = row.querySelector('[data-field=end]');
  const tInp = row.querySelector('[data-field=text]');
  sInp.addEventListener("input", () => {
    const v = parseFloat(sInp.value);
    if (isFinite(v)) clip.captions[idx].start = v;
    markDirty(clip.id);
  });
  eInp.addEventListener("input", () => {
    const v = parseFloat(eInp.value);
    if (isFinite(v)) clip.captions[idx].end = v;
    markDirty(clip.id);
  });
  tInp.addEventListener("input", () => {
    clip.captions[idx].text = tInp.value;
    markDirty(clip.id);
  });
  row.querySelector('[data-action=del]').addEventListener("click", () => {
    clip.captions.splice(idx, 1);
    markDirty(clip.id);
    renderSidebarCaptions(clip);
  });
  return row;
}

function addRow(clipId) {
  const clip = manifest.clips.find(c => c.id === clipId);
  if (!clip) return;
  if (!Array.isArray(clip.captions)) clip.captions = [];
  const lastEnd = clip.captions.reduce((m, c) => Math.max(m, c.end || 0), 0);
  clip.captions.push({
    start: +(lastEnd + 0.1).toFixed(2),
    end: +(lastEnd + 1.5).toFixed(2),
    text: "",
  });
  markDirty(clipId);
  renderSidebarCaptions(clip);
}

function markDirty(clipId) {
  dirty.add(clipId);
  const wrap = document.querySelector(`section.clip[data-id="${CSS.escape(clipId)}"]`);
  if (!wrap) return;
  const status = wrap.querySelector(".status");
  if (status) {
    status.textContent = "unsaved changes";
    status.className = "status dirty";
  }
}

async function saveClip(clipId) {
  const wrap = document.querySelector(`section.clip[data-id="${CSS.escape(clipId)}"]`);
  const clipNow = manifest.clips.find(c => c.id === clipId);
  if (!wrap || !clipNow) return;
  // Captions are kept in-memory on clipNow; sidebar inputs mutate them.
  const captions = (clipNow.captions || [])
    .filter(c => isFinite(c.start) && isFinite(c.end))
    .map(c => ({start: c.start, end: c.end, text: c.text || ""}));

  const rangeEl = wrap.querySelector('[data-range]');
  const newStart = parseFloat(rangeEl.querySelector('[data-field=start]').value);
  const newEnd = parseFloat(rangeEl.querySelector('[data-field=end]').value);
  const rangeChanged =
    Math.abs(newStart - clipNow.start) > 0.005 ||
    Math.abs(newEnd - clipNow.end) > 0.005;

  if (rangeChanged) {
    if (newEnd <= newStart) {
      alert("End must be greater than start.");
      return;
    }
    if (captions.length && !confirm(`Range changed (${(newEnd-newStart).toFixed(2)}s). Captions will be regenerated from the transcript for the new range, discarding any manual edits. Continue?`)) {
      return;
    }
  }

  const status = wrap.querySelector(".status");
  const saveBtn = wrap.querySelector('[data-action=save]');
  status.textContent = rangeChanged ? "re-cutting clip…" : "saving + rendering…";
  status.className = "status";
  if (saveBtn) saveBtn.disabled = true;

  const body = rangeChanged
    ? {start: newStart, end: newEnd}
    : {captions};

  const r = await fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clipId)}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (r.ok) {
    dirty.delete(clipId);
    const updated = await r.json();
    const idx = manifest.clips.findIndex(c => c.id === clipId);
    if (idx >= 0) manifest.clips[idx] = updated;
    if (updated._recut_error) {
      status.textContent = `recut failed: ${updated._recut_error}`;
      status.className = "status dirty";
    } else if (updated._burn_error) {
      status.textContent = `saved · burn failed: ${updated._burn_error}`;
      status.className = "status dirty";
    } else if (updated.render && updated.render.burned_path) {
      status.textContent = "saved + captioned mp4 ready";
      status.className = "status saved";
    } else if (rangeChanged) {
      status.textContent = "raw re-cut · click Save & render again to burn captions";
      status.className = "status saved";
    } else {
      status.textContent = "saved";
      status.className = "status saved";
    }
    const fresh = renderClip(updated);
    wrap.replaceWith(fresh);
    if (activeClipId === clipId) renderSidebarCaptions(updated);
  } else {
    const text = await r.text();
    status.textContent = `error: HTTP ${r.status} ${text.slice(0,100)}`;
    status.className = "status dirty";
    if (saveBtn) saveBtn.disabled = false;
  }
}

function updateOverlay(clip, video, overlay) {
  // The video element plays the cut clip, so currentTime is already
  // clip-relative. Captions are stored clip-relative too -- direct match.
  // Read from the in-memory manifest (sidebar inputs mutate it directly)
  // so this stays in sync regardless of which clip is currently in the
  // sidebar pane.
  const t = video.currentTime;
  let activeText = "";
  let activeIdx = -1;
  const caps = clip.captions || [];
  for (let i = 0; i < caps.length; i++) {
    const c = caps[i];
    if (isFinite(c.start) && isFinite(c.end) && t >= c.start && t <= c.end) {
      activeText = c.text || "";
      activeIdx = i;
      break;
    }
  }
  if (overlay) overlay.textContent = activeText;
  // Highlight the active cap-row in the sidebar (only if this clip is
  // the one currently displayed there).
  if (activeClipId === clip.id) {
    const pane = document.getElementById("sidebar-captions");
    if (pane) {
      const rows = pane.querySelectorAll(".cap-row");
      rows.forEach((r, i) => r.classList.toggle("active", i === activeIdx));
    }
  }
}

async function resyncClip(clipId) {
  if (!confirm(`Resync captions for "${clipId}"?\nThis replaces the current captions with a fresh pass from the word-level transcript and discards any manual edits. The captioned mp4 won't change until you click Save & render.`)) return;
  const wrap = document.querySelector(`section.clip[data-id="${CSS.escape(clipId)}"]`);
  const status = wrap && wrap.querySelector(".status");
  if (status) { status.textContent = "resyncing…"; status.className = "status"; }
  const r = await fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clipId)}/resync`, { method: "POST" });
  if (!r.ok) {
    if (status) { status.textContent = `resync failed: HTTP ${r.status}`; status.className = "status dirty"; }
    return;
  }
  const updated = await r.json();
  const idx = manifest.clips.findIndex(c => c.id === clipId);
  if (idx >= 0) manifest.clips[idx] = updated;
  // Re-render this clip's DOM with the fresh captions
  if (wrap) wrap.replaceWith(renderClip(updated));
  if (activeClipId === clipId) renderSidebarCaptions(updated);
  dirty.delete(clipId);
}

async function archiveClip(clipId) {
  if (!confirm(`Archive "${clipId}"?\nIts captioned mp4 moves to completed/, raw + caption data move to _archive/. You can Restore later.`)) return;
  const r = await fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clipId)}/archive`, { method: "POST" });
  if (!r.ok) {
    alert(`Archive failed: HTTP ${r.status}`);
    return;
  }
  await loadManifest(currentVideoId);
}

async function restoreClip(clipId) {
  const r = await fetch(`/api/clips/${encodeURIComponent(currentVideoId)}/${encodeURIComponent(clipId)}/restore`, { method: "POST" });
  if (!r.ok) {
    alert(`Restore failed: HTTP ${r.status}`);
    return;
  }
  await loadManifest(currentVideoId);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s);
}

window.addEventListener("DOMContentLoaded", loadVideos);
window.addEventListener("beforeunload", e => {
  if (dirty.size) {
    e.preventDefault();
    e.returnValue = "You have unsaved caption edits.";
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    # Class-level config; set when the server is constructed
    output_root: Path = OUTPUT_ROOT

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- video discovery -------------------------------------------------

    def _list_videos(self) -> list[dict]:
        out = []
        if not self.output_root.exists():
            return out
        for sub in sorted(self.output_root.iterdir()):
            clips_path = sub / "clips.json"
            if not clips_path.exists():
                continue
            try:
                manifest = json.loads(clips_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            title = ""
            try:
                meta = json.loads((sub / "metadata.json").read_text())
                title = meta.get("title") or ""
            except (OSError, json.JSONDecodeError):
                pass
            out.append({
                "id": sub.name,
                "title": title,
                "n_clips": len(manifest.get("clips", [])),
            })
        return out

    def _paths_for(self, video_id: str) -> Paths | None:
        if not _VIDEO_ID_RE.match(video_id):
            return None
        target = self.output_root / video_id
        if not (target / "clips.json").exists():
            return None
        return Paths.for_video(video_id)

    # -- helpers ---------------------------------------------------------

    def _send_json(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The HTML page embeds the JS inline -- make sure the browser
        # never serves a stale copy after we've updated server code.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"not found: {path.name}")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size

        # Handle Range requests so the browser can seek the video
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                spec = rng[len("bytes="):].strip()
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                length = max(0, end - start + 1)
                with path.open("rb") as f:
                    f.seek(start)
                    body = f.read(length)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
                return
            except (ValueError, OSError):
                pass

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        # Re-cuts overwrite mp4s in place. Without no-store the browser
        # serves the old bytes after a save and "the slider didn't cut
        # anything" is the visible symptom.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        with path.open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path

        if path in ("/", "/index.html"):
            self._send_bytes(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/videos":
            self._send_json(200, self._list_videos())
            return

        # /api/clips/<video_id>/<clip_id>/waveform
        # /api/clips/<video_id>/<clip_id>/probe
        if path.startswith("/api/clips/") and (
            path.endswith("/waveform") or path.endswith("/probe")
        ):
            stripped = path[len("/api/clips/"):]
            base, action = stripped.rsplit("/", 1)
            if "/" in base:
                video_id, cid = base.split("/", 1)
                paths = self._paths_for(video_id)
                if paths is None:
                    self.send_error(404, "unknown video")
                    return
                if not _CLIP_ID_RE.match(cid):
                    self.send_error(400, "invalid clip id")
                    return
                mp4 = paths.root / "shorts" / f"{cid}.mp4"
                if not mp4.exists():
                    self.send_error(404, "clip mp4 missing")
                    return
                if action == "waveform":
                    cache = paths.root / "shorts" / f"{cid}.peaks.json"
                    peaks = wf.get_or_compute_peaks(mp4, cache)
                    self._send_json(200, {"peaks": peaks})
                    return
                else:  # probe
                    self._send_json(200, wf.probe_clip(mp4))
                    return

        # /api/clips/<video_id>
        if path.startswith("/api/clips/"):
            video_id = path[len("/api/clips/"):]
            paths = self._paths_for(video_id)
            if paths is None:
                self.send_error(404, "unknown video")
                return
            manifest = mf.load(paths)
            try:
                meta = json.loads(paths.metadata.read_text())
                manifest = dict(manifest)
                manifest["title"] = meta.get("title", "")
            except (OSError, json.JSONDecodeError):
                pass
            self._send_json(200, manifest)
            return

        # /video/<video_id>/<clip_id>.mp4 or .captioned.mp4 -- live in shorts/
        if path.startswith("/video/"):
            rest = path[len("/video/"):]
            if "/" in rest and rest.endswith(".mp4"):
                video_id, name = rest.split("/", 1)
                paths = self._paths_for(video_id)
                if paths is None:
                    self.send_error(404, "unknown video")
                    return
                # Try shorts/ then completed/
                mp4 = paths.root / "shorts" / name
                if not mp4.exists():
                    mp4 = paths.root / "completed" / name
                self._send_file(mp4)
                return

        # /completed/<filename> -- serve from the universal completed folder
        if path.startswith("/completed/"):
            name = path[len("/completed/"):]
            if "/" in name or ".." in name.split("/"):
                self.send_error(400, "invalid path")
                return
            self._send_file(self.output_root / "_completed" / name)
            return

        # /source/<video_id>.mp4 -- serve the original full video for trim preview
        if path.startswith("/source/") and path.endswith(".mp4"):
            video_id = path[len("/source/"):-len(".mp4")]
            paths = self._paths_for(video_id)
            if paths is None:
                self.send_error(404, "unknown video")
                return
            self._send_file(paths.root / "video.mp4")
            return

        # /file/<video_id>/<rel_path> -- generic, accepts shorts/* or completed/*
        if path.startswith("/file/"):
            rest = path[len("/file/"):]
            if "/" in rest:
                video_id, rel = rest.split("/", 1)
                paths = self._paths_for(video_id)
                if paths is None:
                    self.send_error(404, "unknown video")
                    return
                # Sandbox: only allow shorts/ or completed/ prefixes,
                # disallow .. components.
                rel = rel.lstrip("/")
                segs = rel.split("/")
                if ".." in segs or not segs[0] in ("shorts", "completed"):
                    self.send_error(400, "invalid path")
                    return
                self._send_file(paths.root / rel)
                return

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path

        # /api/clips/<video_id>/<clip_id>            -- save captions / re-burn
        # /api/clips/<video_id>/<clip_id>/archive    -- archive a clip
        # /api/clips/<video_id>/<clip_id>/restore    -- restore an archived clip
        if not path.startswith("/api/clips/"):
            self.send_error(404)
            return
        rest = path[len("/api/clips/"):]
        segs = rest.split("/")
        action = None
        if len(segs) == 3 and segs[2] in ("archive", "restore", "resync"):
            video_id, cid, action = segs
        elif len(segs) == 2:
            video_id, cid = segs
        else:
            self.send_error(400, "expected /api/clips/<video_id>/<clip_id>[/archive|/restore|/resync]")
            return
        paths = self._paths_for(video_id)
        if paths is None:
            self.send_error(404, "unknown video")
            return
        if not _CLIP_ID_RE.match(cid):
            self.send_error(400, "invalid clip id")
            return

        if action == "archive":
            manifest = mf.load(paths)
            entry = mf.archive_clip(paths, manifest, cid)
            if entry is None:
                self.send_error(404, "no active clip with that id")
                return
            mf.save(paths, manifest)
            self._send_json(200, entry)
            return

        if action == "restore":
            manifest = mf.load(paths)
            entry = mf.restore_clip(paths, manifest, cid)
            if entry is None:
                self.send_error(404, "no archived clip with that id")
                return
            mf.save(paths, manifest)
            self._send_json(200, entry)
            return

        if action == "resync":
            # Re-run faster-whisper on JUST the clip's audio segment
            # (rather than re-grouping the full-video transcript). This
            # is slower (~5-15s incl. model load on first run) but more
            # accurate for short clips. The captioned mp4 isn't
            # regenerated -- the user must click Save & render after.
            manifest = mf.load(paths)
            clip = mf.get_clip(manifest, cid)
            if clip is None:
                self.send_error(404, "no active clip with that id")
                return
            if not paths.audio.exists():
                self.send_error(409, "audio.wav missing -- run main pipeline first")
                return
            try:
                words = seg_tx.transcribe_segment(
                    paths.audio,
                    float(clip["start"]),
                    float(clip["end"]),
                )
            except Exception as e:  # pragma: no cover
                log.exception("segment transcribe failed")
                self.send_error(500, f"transcribe failed: {type(e).__name__}: {e}")
                return
            phrases = cap.group_phrases(words, clip_start=0.0, max_words=3)
            clip["captions"] = [p.to_dict() for p in phrases]
            if clip.get("render"):
                clip["render"]["captions_stale"] = True
            mf.save(paths, manifest)
            self._send_json(200, clip)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        manifest = mf.load(paths)
        clip = mf.get_clip(manifest, cid)
        if clip is None:
            self.send_error(404, f"unknown clip: {cid}")
            return

        # Did start/end change? If so we need to re-cut the raw mp4
        # from the source video for the new range, and regenerate
        # captions for the new range. We do that BEFORE applying any
        # caption payload, so caller-supplied caption edits are
        # discarded by an explicit range change (a range change is a
        # bigger operation that supersedes pending text edits).
        recut_error = None
        try:
            new_start = (float(payload["start"])
                         if "start" in payload else float(clip["start"]))
            new_end = (float(payload["end"])
                       if "end" in payload else float(clip["end"]))
        except (TypeError, ValueError):
            self.send_error(400, "start/end must be numbers")
            return
        if new_end <= new_start:
            self.send_error(400, "end must be greater than start")
            return
        range_changed = (
            abs(new_start - float(clip["start"])) > 0.005
            or abs(new_end - float(clip["end"])) > 0.005
        )
        if range_changed:
            try:
                src_video = video_download.fetch_video(paths)
                clip["start"] = new_start
                clip["end"] = new_end
                shorts = mf.shorts_dir(paths)
                out_raw = shorts / f"{cid}.mp4"
                # Re-detect face shots for the new range so each
                # camera angle gets its own stable crop.
                from . import detect as _det
                track = _det.detect_face_track(
                    src_video, new_start, new_end, sample_dt=1.0,
                )
                if track is not None and track["n_detected"] >= 2:
                    per_frame = _det.track_to_zoom_keyframes(track)
                    shots = _det.cluster_to_shots(per_frame)
                    if len(shots) > 1:
                        _det.refine_shot_boundaries(
                            src_video, new_start,
                            (track["src_w"], track["src_h"]),
                            shots,
                        )
                    clip["face_track"] = _det.shots_to_static_keyframes(shots)
                else:
                    clip["face_track"] = None
                spec = cut_mod.CutSpec(
                    src_video=src_video,
                    out_path=out_raw,
                    start_s=new_start,
                    end_s=new_end,
                    ass_path=None,  # raw preview only
                    style=clip.get("style", "blur-bg"),
                    zoom=float(clip.get("zoom", 1.0)),
                    focus_x=float(clip.get("focus_x", 0.5)),
                    focus_y=float(clip.get("focus_y", 0.5)),
                    face_track=clip.get("face_track"),
                )
                cut_mod.cut_clip(spec)
                # Regenerate captions from the transcript for the new
                # range. The user can resync (re-transcribe) or hand-
                # edit afterward.
                phrases = cap.phrases_for_clip(
                    paths.transcript_raw, new_start, new_end, max_words=3,
                )
                clip["captions"] = [p.to_dict() for p in phrases]
                clip["render"] = {
                    "path": str(out_raw.relative_to(paths.root)),
                    "rendered_at": mf.now_iso(),
                    "style": clip.get("style", "blur-bg"),
                    "captions_burned": False,
                    "captions_stale": True,
                }
            except Exception as e:  # pragma: no cover
                log.exception("range re-cut failed")
                recut_error = f"{type(e).__name__}: {e}"

        if not range_changed and "captions" in payload and isinstance(payload["captions"], list):
            cleaned = []
            for c in payload["captions"]:
                try:
                    cleaned.append({
                        "start": float(c["start"]),
                        "end": float(c["end"]),
                        "text": str(c.get("text", "")),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            cleaned.sort(key=lambda c: c["start"])
            clip["captions"] = cleaned

        if "title" in payload:
            clip["title"] = str(payload["title"])
        if "style" in payload and payload["style"] in ("blur-bg", "crop", "letterbox"):
            clip["style"] = payload["style"]

        # Re-render the captioned (.captioned.mp4) sibling so the user
        # immediately has an updated final output to download. The raw
        # <id>.mp4 stays untouched so the live-preview overlay keeps
        # working for further edits.
        burn_error = None
        try:
            shorts = mf.shorts_dir(paths)
            raw = shorts / f"{cid}.mp4"
            if not raw.exists():
                burn_error = "raw preview mp4 missing; re-cut first"
            elif not clip.get("captions"):
                # No captions to burn: clear any existing captioned file
                cap_path = shorts / f"{cid}.captioned.mp4"
                if cap_path.exists():
                    cap_path.unlink()
                if clip.get("render"):
                    clip["render"]["burned_path"] = None
                    clip["render"]["burned_at"] = None
            else:
                ass_path = shorts / f"{cid}.ass"
                phrases = [
                    cap.Phrase(p["start"], p["end"], p["text"])
                    for p in clip["captions"]
                ]
                cap.write_ass(phrases, ass_path)
                out = shorts / f"{cid}.captioned.mp4"
                cut_mod.burn_captions_onto(raw, ass_path, out)
                if clip.get("render") is None:
                    clip["render"] = {}
                clip["render"]["burned_path"] = str(out.relative_to(paths.root))
                clip["render"]["ass_path"] = str(ass_path.relative_to(paths.root))
                clip["render"]["burned_at"] = mf.now_iso()
                clip["render"].pop("captions_stale", None)
        except Exception as e:  # pragma: no cover - surface to UI
            log.exception("burn-on-save failed")
            burn_error = f"{type(e).__name__}: {e}"

        mf.save(paths, manifest)

        response = dict(clip)
        if burn_error:
            response["_burn_error"] = burn_error
        if recut_error:
            response["_recut_error"] = recut_error
        self._send_json(200, response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(output_root: Path = OUTPUT_ROOT, port: int = 7707) -> None:
    """Run the review server until Ctrl-C.

    Scans `output_root` on every request so newly-cut clips show up
    without restarting.
    """

    class Handler(_Handler):
        pass
    Handler.output_root = output_root

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log.info("serving %s at http://127.0.0.1:%d/", output_root, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()
