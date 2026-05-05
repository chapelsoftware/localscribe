"""CLI entry point: `localscribe-clips`.

Subcommands:
  fetch     download the source video (audio-only is what the main
            pipeline keeps; this pulls full video so we can cut from it)
  cut       define + render a single clip with auto-captions
  list      show the clips defined for a video
  render    re-render an existing clip (after editing captions, etc.)
  review    open a local web UI for caption editing
"""
from __future__ import annotations

import json
import logging
import re
import webbrowser
from pathlib import Path

import click

from ..config import Paths
from ..log import setup_logging
from ..stages import download as audio_download
from . import captions as cap
from . import cut as cut_mod
from . import download as video_download
from . import manifest as mf

log = logging.getLogger("clips.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$|^(\d+(?:\.\d+)?)$")


def _parse_focus(s: str) -> tuple[float, float]:
    """Parse '0.5,0.5' into (0.5, 0.5). Values clipped to [0..1]."""
    try:
        x, y = (float(p) for p in s.split(","))
    except (ValueError, TypeError):
        raise click.BadParameter(f"--focus: expected 'x,y' got {s!r}")
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def parse_time(s: str) -> float:
    """Accept seconds (`91.59`), `MM:SS.cc`, or `H:MM:SS.cc`."""
    s = str(s).strip()
    m = _TIME_RE.match(s)
    if not m:
        raise click.BadParameter(f"could not parse time: {s!r}")
    if m.group(4) is not None:
        return float(m.group(4))
    h = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = float(m.group(3))
    return h * 3600 + minutes * 60 + seconds


def resolve_paths(source: str) -> Paths:
    """Get a Paths object for an existing run. Source can be a URL, an
    11-char video id, or any input the main pipeline accepted."""
    video_id = audio_download.resolve_video_id(source)
    paths = Paths.for_video(video_id)
    if not paths.transcript_raw.exists():
        raise click.ClickException(
            f"No transcript for {video_id}. Run the main pipeline first:\n"
            f"  localscribe {source}"
        )
    return paths


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
def main(verbose: bool, quiet: bool) -> None:
    """Generate vertical short-form clips from a transcribed video."""
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")
    setup_logging(1 if verbose else (-1 if quiet else 0))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@main.command("fetch")
@click.argument("source")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def fetch_cmd(source: str, force: bool) -> None:
    """Download the original video stream for SOURCE.

    The main pipeline only saves audio. This pulls the full video so
    `cut` has something to trim from.
    """
    paths = resolve_paths(source)
    out = video_download.fetch_video(paths, force=force)
    click.echo(f"video: {out}")


# ---------------------------------------------------------------------------
# cut
# ---------------------------------------------------------------------------

@main.command("cut")
@click.argument("source")
@click.option("--start", default=None,
              help="Start time (s, MM:SS, or H:MM:SS). Mutually "
                   "exclusive with --start-phrase.")
@click.option("--end", default=None,
              help="End time. Mutually exclusive with --end-phrase.")
@click.option("--start-phrase", default=None,
              help="Define the start by what was said: a literal "
                   "phrase from the transcript. Resolved against the "
                   "word-level transcript. Punctuation and contractions "
                   "are normalized so 'we are' matches 'we're'.")
@click.option("--end-phrase", default=None,
              help="Define the end by what was said. Searched AFTER "
                   "the start match so a phrase that recurs in the "
                   "video doesn't snap backward.")
@click.option("--title", required=True, help="Human-readable title.")
@click.option("--id", "clip_id", default=None,
              help="Stable id (default: slug of title).")
@click.option("--style",
              default="blur-bg",
              type=click.Choice(["blur-bg", "crop", "letterbox"]),
              help="Reframe style for 9:16 conversion.")
@click.option("--zoom", default=1.0, type=float,
              help="Pre-9:16 zoom factor. 1.0 = no zoom. 1.4 crops in "
                   "to the inner ~70%% of the source so the speaker "
                   "fills more of the final frame. Values 1.0-2.0 "
                   "are typical. Ignored if --auto-zoom is on and "
                   "succeeds.")
@click.option("--focus", default="0.5,0.5",
              help="Focus point for zoom crop, as 'x,y' in [0..1] "
                   "normalized coords. Default '0.5,0.5' = centered. "
                   "Move toward the speaker if they're off-center. "
                   "Ignored if --auto-zoom is on and succeeds.")
@click.option("--auto-zoom", is_flag=True,
              help="Detect the speaker's face in sample frames and "
                   "auto-pick zoom + focus. Falls back to manual "
                   "--zoom/--focus if no face is found.")
@click.option("--font", default="DejaVu Sans",
              help="Caption font name (must be installed).")
@click.option("--font-size", default=78, type=int)
@click.option("--accent", default="#fbb53c",
              help="Hex color used for caption fill (currently sets "
                   "secondary color; future: word-highlight).")
@click.option("--width", default=1080, type=int)
@click.option("--height", default=1920, type=int)
@click.option("--burn/--no-burn", default=False,
              help="Burn captions into the rendered mp4. Default is "
                   "off so you can preview / edit captions in the "
                   "review UI before finalizing with `render`.")
@click.option("--no-captions", is_flag=True,
              help="Skip caption generation entirely.")
@click.option("--max-words", default=3, type=int,
              help="Max words per caption phrase.")
@click.option("--no-render", is_flag=True,
              help="Persist clip definition + captions but skip ffmpeg.")
@click.option("--head-pad", default=0.0, type=float,
              help="Extend clip start backward by N seconds (default 0). "
                   "Applied BEFORE silence snap.")
@click.option("--tail-pad", default=0.0, type=float,
              help="Extend clip end forward by N seconds (default 0). "
                   "Applied BEFORE silence snap.")
@click.option("--snap-silence/--no-snap-silence", default=True,
              help="Snap start/end to the actual speech onset / offset "
                   "by probing audio RMS. Leaves a tight ~60ms head and "
                   "~200ms tail of silence around the spoken content. "
                   "Default on; pass --no-snap-silence to use exactly "
                   "the boundaries you provided.")
def cut_cmd(
    source: str, start: str | None, end: str | None,
    start_phrase: str | None, end_phrase: str | None,
    title: str, clip_id: str | None,
    style: str, zoom: float, focus: str, auto_zoom: bool,
    font: str, font_size: int, accent: str,
    width: int, height: int, burn: bool, no_captions: bool,
    max_words: int, no_render: bool, head_pad: float, tail_pad: float,
    snap_silence: bool,
) -> None:
    """Define and render a single short clip.

    Boundaries can be given as timestamps (--start/--end) OR as
    transcript phrases (--start-phrase/--end-phrase). Phrase mode is
    handy when you've identified a clip by its content rather than
    by the clock.

    Auto-generates captions from the existing word-level transcript and
    saves them to clips.json. By default the rendered mp4 has NO
    captions burned in -- review and edit captions with
    `localscribe-clips review`, then finalize with
    `localscribe-clips render` (which burns them in). Pass `--burn` to
    skip that two-step flow and burn captions on first cut.
    """
    paths = resolve_paths(source)
    # Boundary resolution: timestamps OR phrases, exactly one of each.
    if (start is None) == (start_phrase is None):
        raise click.UsageError("supply exactly one of --start / --start-phrase")
    if (end is None) == (end_phrase is None):
        raise click.UsageError("supply exactly one of --end / --end-phrase")
    if start is not None:
        start_s = parse_time(start)
    if end is not None:
        end_s = parse_time(end)
    if start_phrase is not None or end_phrase is not None:
        from . import transcript as _tr
        # If only ONE side is a phrase, resolve just that side.
        if start_phrase is not None and end_phrase is not None:
            start_s, end_s = _tr.resolve_boundaries(
                paths.transcript_raw, start_phrase, end_phrase,
            )
        else:
            words = _tr._load_words(paths.transcript_raw)
            if start_phrase is not None:
                m = _tr.find_phrase(words, start_phrase)
                if m is None:
                    raise click.ClickException(
                        f"start phrase not found: {start_phrase!r}")
                start_s = max(0.0, m[0] - 0.1)
            if end_phrase is not None:
                m = _tr.find_phrase(words, end_phrase)
                if m is None:
                    raise click.ClickException(
                        f"end phrase not found: {end_phrase!r}")
                end_s = m[1] + 0.3
    if end_s <= start_s:
        raise click.BadParameter("end must be greater than start")

    # Apply head/tail padding first (rarely needed now that snap-silence
    # handles boundary slop, but kept for manual nudging).
    start_s = max(0.0, start_s - head_pad)
    end_s = end_s + tail_pad

    cid = clip_id or mf.slugify(title)

    # 1. auto-generate caption phrases from the word-level transcript
    # over the user's window. We do this BEFORE snap so caption
    # boundaries can drive the snap (the user's start/end often
    # include warm-up speech that bleeds outside the captioned
    # content).
    if no_captions:
        phrases = []
    else:
        phrases = cap.phrases_for_clip(
            paths.transcript_raw,
            start_s,
            end_s,
            max_words=max_words,
        )

    # Snap: tighten boundaries to the actual content. Uses caption
    # boundaries as the primary signal; falls back to audio RMS when
    # there are no captions.
    if snap_silence:
        from . import silence
        src_for_snap = video_download.fetch_video(paths)
        cap_dicts = [
            {"start": p.start, "end": p.end} for p in phrases
        ] if phrases else None
        snapped_s, snapped_e = silence.snap_to_content(
            src_for_snap, start_s, end_s, captions=cap_dicts,
        )
        if abs(snapped_s - start_s) > 0.005 or abs(snapped_e - end_s) > 0.005:
            click.echo(
                f"snap: {start_s:.2f}->{snapped_s:.2f} "
                f"({snapped_s - start_s:+.2f}s)  "
                f"{end_s:.2f}->{snapped_e:.2f} "
                f"({snapped_e - end_s:+.2f}s)"
            )
            # Re-derive captions for the snapped window so phrase
            # times line up with the new clip-zero.
            start_s, end_s = snapped_s, snapped_e
            if not no_captions:
                phrases = cap.phrases_for_clip(
                    paths.transcript_raw, start_s, end_s,
                    max_words=max_words,
                )
    caption_entries = [p.to_dict() for p in phrases]

    fx, fy = _parse_focus(focus)

    # Optional: auto-detect the speaker's face and pick zoom + focus.
    # First try a dense per-second face track so cuts that span camera
    # changes still keep the speaker centered. Fall back to a single
    # static crop if too few frames had a successful detection.
    face_track = None
    if auto_zoom:
        from . import detect
        src_video_for_detect = video_download.fetch_video(paths)
        track = detect.detect_face_track(
            src_video_for_detect, start_s, end_s, sample_dt=1.0,
        )
        if track is not None and track["n_detected"] >= 2:
            per_frame = detect.track_to_zoom_keyframes(track)
            shots = detect.cluster_to_shots(per_frame)
            # Refine each shot boundary to frame-level accuracy via
            # binary search -- otherwise transitions snap on the 1s
            # face-sample grid and lag the actual camera cut by up to
            # a second.
            if len(shots) > 1:
                detect.refine_shot_boundaries(
                    src_video_for_detect, start_s,
                    (track["src_w"], track["src_h"]),
                    shots,
                )
            face_track = detect.shots_to_static_keyframes(shots)
            shot_summary = ", ".join(
                f"@{k['t']:.1f}s zoom={k['zoom']:.2f} focus=({k['focus_x']:.2f},{k['focus_y']:.2f})"
                for k in face_track
            )
            click.echo(
                f"auto-zoom: {len(face_track)} shot{'s' if len(face_track) != 1 else ''} "
                f"detected from {track['n_detected']}/{track['n_sampled']} face "
                f"samples @ {track['src_w']}x{track['src_h']}"
            )
            if len(face_track) > 1:
                click.echo(f"  shots: {shot_summary}")
            # Seed static fields with the longest shot's values for
            # downstream `render` calls that don't replay the track.
            longest_shot = max(face_track, key=lambda k: k.get("n_frames", 1))
            zoom = longest_shot["zoom"]
            fx, fy = longest_shot["focus_x"], longest_shot["focus_y"]
        else:
            face = detect.detect_face_in_clip(
                src_video_for_detect, start_s, end_s, n_samples=6,
            )
            if face is not None:
                params = detect.auto_zoom_params(face)
                zoom = params["zoom"]
                fx, fy = params["focus_x"], params["focus_y"]
                click.echo(
                    f"auto-zoom (static): zoom={zoom:.2f} "
                    f"focus=({fx:.2f},{fy:.2f}) "
                    f"(face in {face.n_detected}/{face.n_samples} frames)"
                )
            else:
                click.echo("auto-zoom: no reliable face detection -- falling "
                           f"back to --zoom={zoom} --focus={fx},{fy}")

    # 2. persist to manifest
    manifest = mf.load(paths)
    new_entry = mf.make_clip_entry(
        clip_id=cid, title=title, start=start_s, end=end_s,
        style=style, zoom=zoom, focus_x=fx, focus_y=fy,
        face_track=face_track,
        captions=caption_entries,
    )
    # upsert_clip returns the canonical entry currently in the manifest
    # (an already-existing dict if cid was present, else the one we just
    # passed in). All later mutations must go through `entry` so they
    # stick after `mf.save`.
    entry = mf.upsert_clip(manifest, new_entry)

    if no_render:
        mf.save(paths, manifest)
        click.echo(f"saved (no render): {cid}  {start_s:.2f}-{end_s:.2f}  "
                   f"{len(caption_entries)} caption phrases")
        return

    # 3. ensure source video exists
    src_video = video_download.fetch_video(paths)

    # 4. write ASS file only if we're going to burn captions
    shorts = mf.shorts_dir(paths)
    ass_path = None
    will_burn = burn and phrases and not no_captions
    if will_burn:
        ass_path = shorts / f"{cid}.ass"
        cap.write_ass(
            phrases, ass_path,
            playres_x=width, playres_y=height,
            font=font, font_size=font_size,
            accent_color=accent,
        )

    # 5. ffmpeg
    out_path = shorts / f"{cid}.mp4"
    spec = cut_mod.CutSpec(
        src_video=src_video,
        out_path=out_path,
        start_s=start_s,
        end_s=end_s,
        ass_path=ass_path,  # None unless --burn
        width=width,
        height=height,
        style=style,
        zoom=zoom,
        focus_x=fx,
        focus_y=fy,
        face_track=face_track,
    )
    cut_mod.cut_clip(spec)

    # 6. record render in manifest + save
    entry["render"] = {
        "path": str(out_path.relative_to(paths.root)),
        "ass_path": str(ass_path.relative_to(paths.root)) if ass_path else None,
        "rendered_at": mf.now_iso(),
        "style": style,
        "captions_burned": will_burn,
    }
    mf.save(paths, manifest)

    click.echo(f"rendered: {out_path}")
    click.echo(f"  duration: {end_s - start_s:.2f}s, "
               f"{len(caption_entries)} caption phrases "
               f"({'burned in' if will_burn else 'preview only — run `render` to burn'})")


# ---------------------------------------------------------------------------
# cut-batch
# ---------------------------------------------------------------------------

@main.command("cut-batch")
@click.argument("source")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--replace", is_flag=True,
              help="Wipe every existing active clip (and its rendered "
                   "files) before defining the new ones. Manifest is "
                   "backed up first to clips.<utc>.bak.json next to it.")
@click.option("--auto-zoom/--no-auto-zoom", default=True,
              help="Default for clips that don't override it.")
@click.option("--style",
              default="blur-bg",
              type=click.Choice(["blur-bg", "crop", "letterbox"]),
              help="Default reframe style for clips that don't override it.")
def cut_batch_cmd(source: str, config: str, replace: bool,
                  auto_zoom: bool, style: str) -> None:
    """Cut many clips for one video from a JSON config.

    CONFIG is a JSON file shaped like:

      {
        "clips": [
          {
            "id": "growth-vs-health",
            "title": "Growth Is Not The Same As Health",
            "start_phrase": "But I think the pursuit of growth",
            "end_phrase": "eliminate health for a pursuit of growth"
          },
          {
            "id": "another",
            "title": "Another Clip",
            "start": 1448.0,
            "end": 1535.5,
            "style": "crop"
          }
        ]
      }

    Each clip can use either timestamps (`start`/`end` in seconds) or
    transcript phrases (`start_phrase`/`end_phrase`). Top-level
    `--auto-zoom` and `--style` defaults are inherited unless the clip
    overrides them.

    Pass `--replace` to wipe the existing manifest first (with backup).
    Without `--replace`, new clips are upserted alongside existing ones.
    """
    paths = resolve_paths(source)
    cfg = json.loads(Path(config).read_text())
    items = cfg.get("clips", [])
    if not items:
        raise click.ClickException("config has no `clips` list")

    if replace:
        bak = mf.backup(paths)
        manifest = mf.load(paths)
        wiped = mf.wipe_active(paths, manifest)
        mf.save(paths, manifest)
        click.echo(
            f"replaced: backed up to {bak.name if bak else '(none)'}, "
            f"wiped {len(wiped)} clip(s)"
        )

    # Resolve every boundary up front so a typo fails fast before we
    # spend ffmpeg time.
    plan = []
    from . import transcript as _tr
    words_cache = None
    for spec in items:
        cid = spec.get("id")
        title = spec.get("title")
        if not cid or not title:
            raise click.ClickException(f"clip missing id/title: {spec!r}")
        if "start" in spec and "start_phrase" in spec:
            raise click.ClickException(f"{cid}: only one of start/start_phrase")
        if "end" in spec and "end_phrase" in spec:
            raise click.ClickException(f"{cid}: only one of end/end_phrase")
        if "start_phrase" in spec or "end_phrase" in spec:
            if words_cache is None:
                words_cache = _tr._load_words(paths.transcript_raw)
        if "start" in spec:
            s = float(spec["start"])
        else:
            m = _tr.find_phrase(words_cache, spec["start_phrase"])
            if m is None:
                raise click.ClickException(
                    f"{cid}: start phrase not found: {spec['start_phrase']!r}")
            s = max(0.0, m[0] - 0.1)
        if "end" in spec:
            e = float(spec["end"])
        else:
            # Search end AFTER the start match if both phrases were given.
            if "start_phrase" in spec:
                start_m = _tr.find_phrase(words_cache, spec["start_phrase"])
                start_idx = start_m[2] - 1 if start_m else 0
            else:
                start_idx = 0
            m = _tr.find_phrase(words_cache, spec["end_phrase"], start_idx=start_idx)
            if m is None:
                raise click.ClickException(
                    f"{cid}: end phrase not found: {spec['end_phrase']!r}")
            e = m[1] + 0.3
        if e <= s:
            raise click.ClickException(f"{cid}: end <= start ({s:.2f}/{e:.2f})")
        plan.append((cid, title, s, e, spec))

    click.echo(f"cut-batch: {len(plan)} clip(s) to cut")
    ctx = click.get_current_context()
    failures: list[tuple[str, Exception]] = []
    for i, (cid, title, s, e, spec) in enumerate(plan, 1):
        click.echo(f"\n[{i}/{len(plan)}] {cid}  {s:.1f}-{e:.1f} ({e-s:.1f}s)")
        try:
            ctx.invoke(
                cut_cmd,
                source=source,
                start=f"{s:.3f}",
                end=f"{e:.3f}",
                start_phrase=None,
                end_phrase=None,
                title=title,
                clip_id=cid,
                style=spec.get("style", style),
                zoom=float(spec.get("zoom", 1.0)),
                focus=spec.get("focus", "0.5,0.5"),
                auto_zoom=bool(spec.get("auto_zoom", auto_zoom)),
                font=spec.get("font", "DejaVu Sans"),
                font_size=int(spec.get("font_size", 78)),
                accent=spec.get("accent", "#fbb53c"),
                width=int(spec.get("width", 1080)),
                height=int(spec.get("height", 1920)),
                burn=bool(spec.get("burn", False)),
                no_captions=bool(spec.get("no_captions", False)),
                max_words=int(spec.get("max_words", 3)),
                no_render=bool(spec.get("no_render", False)),
                head_pad=float(spec.get("head_pad", 0.0)),
                tail_pad=float(spec.get("tail_pad", 0.0)),
                snap_silence=bool(spec.get("snap_silence", True)),
            )
        except (click.ClickException, click.exceptions.Exit, RuntimeError) as ex:
            log.exception("cut-batch: %s failed", cid)
            failures.append((cid, ex))

    click.echo(f"\ncut-batch done: {len(plan) - len(failures)}/{len(plan)} ok")
    if failures:
        for cid, ex in failures:
            click.echo(f"  FAILED {cid}: {ex}", err=True)
        ctx.exit(1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@main.command("list")
@click.argument("source")
def list_cmd(source: str) -> None:
    """Show the clips defined for SOURCE."""
    paths = resolve_paths(source)
    manifest = mf.load(paths)
    if not manifest["clips"]:
        click.echo(f"(no clips for {paths.video_id})")
        return
    click.echo(f"{paths.video_id}: {len(manifest['clips'])} clip(s)")
    for c in manifest["clips"]:
        rendered = "✓" if c.get("render") else " "
        click.echo(
            f"  [{rendered}] {c['id']:36s}  "
            f"{c['start']:>7.2f}-{c['end']:<7.2f}  "
            f"({c['end'] - c['start']:>5.1f}s)  {c['title']}"
        )


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

@main.command("render")
@click.argument("source")
@click.option("--id", "clip_id", default=None,
              help="Render only this clip id (default: all).")
@click.option("--font", default="DejaVu Sans")
@click.option("--font-size", default=78, type=int)
@click.option("--accent", default="#fbb53c")
def render_cmd(source: str, clip_id: str | None,
               font: str, font_size: int, accent: str) -> None:
    """Re-render clips after editing captions in the manifest.

    Uses the captions stored on the clip entry rather than regenerating
    them from the word-level transcript.
    """
    paths = resolve_paths(source)
    manifest = mf.load(paths)
    src_video = video_download.fetch_video(paths)
    shorts = mf.shorts_dir(paths)

    targets = ([mf.get_clip(manifest, clip_id)] if clip_id
               else list(manifest["clips"]))
    if not targets or targets == [None]:
        raise click.ClickException("no clips to render")

    for entry in targets:
        if entry is None:
            continue
        cid = entry["id"]
        click.echo(f"-- {cid}")
        phrases = [
            cap.Phrase(start=c["start"], end=c["end"], text=c["text"])
            for c in entry.get("captions", [])
        ]
        ass_path = None
        if phrases:
            ass_path = shorts / f"{cid}.ass"
            cap.write_ass(phrases, ass_path,
                          playres_x=1080, playres_y=1920,
                          font=font, font_size=font_size,
                          accent_color=accent)
        spec = cut_mod.CutSpec(
            src_video=src_video,
            out_path=shorts / f"{cid}.mp4",
            start_s=float(entry["start"]),
            end_s=float(entry["end"]),
            ass_path=ass_path,
            style=entry.get("style", "blur-bg"),
            zoom=float(entry.get("zoom", 1.0)),
            focus_x=float(entry.get("focus_x", 0.5)),
            focus_y=float(entry.get("focus_y", 0.5)),
            face_track=entry.get("face_track"),
        )
        cut_mod.cut_clip(spec)
        entry["render"] = {
            "path": str(spec.out_path.relative_to(paths.root)),
            "ass_path": str(ass_path.relative_to(paths.root)) if ass_path else None,
            "rendered_at": mf.now_iso(),
            "style": entry.get("style", "blur-bg"),
            "captions_burned": ass_path is not None,
        }
        click.echo(f"   -> {spec.out_path}")

    mf.save(paths, manifest)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

@main.command("review")
@click.argument("source", required=False)
@click.option("--port", default=7707, type=int)
@click.option("--no-browser", is_flag=True,
              help="Don't open the browser automatically.")
def review_cmd(source: str | None, port: int, no_browser: bool) -> None:
    """Start a local web UI for editing captions on the defined clips.

    With no SOURCE the UI scans `output/` and exposes every video that
    has a `clips.json`, with a dropdown to switch between them. Pass an
    optional SOURCE to deep-link to one video on launch (the dropdown
    still lists everything).
    """
    from ..config import OUTPUT_ROOT
    from . import web

    initial_video_id = None
    if source:
        paths = resolve_paths(source)
        manifest = mf.load(paths)
        if not manifest["clips"]:
            raise click.ClickException(
                f"No clips defined for {paths.video_id}. "
                f"Run `localscribe-clips cut ...` first."
            )
        initial_video_id = paths.video_id

    n_videos = sum(1 for sub in OUTPUT_ROOT.iterdir()
                   if sub.is_dir() and (sub / "clips.json").exists())
    if n_videos == 0:
        raise click.ClickException(
            "No videos with a clips.json found. Run "
            "`localscribe-clips cut ...` first."
        )

    base = f"http://127.0.0.1:{port}/"
    url = f"{base}#v={initial_video_id}" if initial_video_id else base
    click.echo(f"Review UI: {url}  ({n_videos} video{'s' if n_videos != 1 else ''})")
    click.echo("Edit captions, click Save & render to produce a captioned mp4 alongside the raw preview.")
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    web.serve(OUTPUT_ROOT, port=port)


if __name__ == "__main__":
    main()
