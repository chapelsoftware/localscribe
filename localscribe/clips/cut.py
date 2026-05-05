"""ffmpeg pipeline that turns a source video + clip span + captions into
a vertical 9:16 short, suitable for YouTube Shorts / TikTok / Reels.

Three reframe styles are supported:
  - "blur-bg" (default): scale a copy of the source to fill the 9:16
    canvas, blur it heavily, and overlay the original (scaled to canvas
    width) on top. The Submagic / Opus look.
  - "crop": scale the source to fill the 9:16 height, then center-crop
    its width. Loses the sides; best when the speaker is centered.
  - "letterbox": scale the source to canvas width and pillar/letter-box
    on a solid color background. Cleanest, least flashy.

Captions are an ASS file produced by `captions.render_ass`. They are
burned into the final stream via libass (`subtitles=...`) so the output
plays without a sidecar file.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("clips.cut")


@dataclass
class CutSpec:
    src_video: Path
    out_path: Path
    start_s: float
    end_s: float
    ass_path: Path | None = None
    width: int = 1080
    height: int = 1920
    style: str = "blur-bg"  # blur-bg | crop | letterbox
    blur_strength: int = 20
    bg_color: str = "black"  # for letterbox
    # Pre-9:16 zoom: scale > 1.0 crops in on the center of the source
    # (cropping to iw/zoom × ih/zoom). Useful when the speaker fills
    # only a small portion of the original 16:9 frame -- a 1.4x zoom
    # makes them visibly larger in the final 9:16 composition.
    zoom: float = 1.0
    # Optional manual focus point in normalized [0..1] coords. Default
    # `(0.5, 0.5)` = center crop. Move toward the speaker's head if
    # they're off-center.
    focus_x: float = 0.5
    focus_y: float = 0.5
    # Optional time-varying face track. If given, overrides the static
    # zoom / focus_x / focus_y above. Each entry is
    # {"t": clip-relative seconds, "zoom": float,
    #  "focus_x": float in [0,1], "focus_y": float in [0,1]}.
    # Values are linearly interpolated between adjacent keyframes;
    # ffmpeg holds the first/last value beyond the boundary.
    face_track: list[dict] | None = None
    video_bitrate: str = "6M"
    audio_bitrate: str = "192k"
    crf: int = 20  # used if video_bitrate is None / unset
    preset: str = "medium"  # ffmpeg x264 preset

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _ass_for_filter(ass_path: Path) -> str:
    """Quote an ASS path for ffmpeg's `subtitles=...` filter.

    libass's filter parser is fussy: colons need escaping (Windows
    drive letters), backslashes too, and the whole value should be
    wrapped in single quotes inside the filtergraph.
    """
    s = str(ass_path.resolve())
    s = s.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return f"'{s}'"


def _piecewise_constant_expr(times: list[float], values: list[float]) -> str:
    """Build a STEP-FUNCTION ffmpeg expression: value `values[i]`
    holds from time `times[i]` until `times[i+1]` (exclusive). At
    `t < times[0]` the first value still applies, and at
    `t >= times[-1]` the last value applies.

    Each entry corresponds to one "shot" (a stable camera angle), so
    the crop stays still through that shot and SNAPS at the boundary
    when the source cuts to a new angle.

    Commas are pre-escaped (`\\,`) for nesting inside ffmpeg filter
    expressions.
    """
    assert len(times) == len(values) and len(times) > 0
    if len(times) == 1:
        return f"{values[0]:.6f}"
    expr = f"{values[-1]:.6f}"
    # Walk inside-out: at each successive boundary times[i], if t is
    # before it, use the previous segment's value, else fall through
    # to the inner expression.
    for i in range(len(times) - 1, 0, -1):
        expr = (f"if(lt(t\\,{times[i]:.6f})\\,"
                f"{values[i - 1]:.6f}\\,{expr})")
    return expr


def _zoom_inline(spec: CutSpec) -> str:
    """Return an inline `crop=...,` filter clip for pre-9:16 zoom.

    Empty string (no-op) when zoom == 1.0 and there's no face_track.
    When `face_track` is provided with multiple keyframes, the crop's
    width / height / x / y are time-varying piecewise-linear
    expressions in `t` -- ffmpeg's crop filter recomputes them each
    frame, so the speaker stays centered through camera changes.

    Width is determined by zoom (`iw / z`) -- this controls how tightly
    the speaker is framed horizontally. Height takes as much of the
    source as possible up to the canvas aspect ratio. At sufficient
    zoom the crop reshapes toward 9:16 so the foreground fills the
    canvas with no blur-bg margin.
    """
    canvas_h_over_w = spec.height / spec.width   # e.g. 1920/1080 = 1.7778

    track = spec.face_track or []
    if len(track) >= 2:
        # Step-function expressions (one keyframe per shot). Crop stays
        # still during a single camera angle and snaps at cuts.
        times = [float(k["t"]) for k in track]
        zooms = [max(1.0, float(k["zoom"])) for k in track]
        fxs = [max(0.0, min(1.0, float(k["focus_x"]))) for k in track]
        fys = [max(0.0, min(1.0, float(k["focus_y"]))) for k in track]
        z_expr = _piecewise_constant_expr(times, zooms)
        fx_expr = _piecewise_constant_expr(times, fxs)
        fy_expr = _piecewise_constant_expr(times, fys)
        cw = f"iw/({z_expr})"
        ch = f"min(ih\\,({cw})*{canvas_h_over_w:.4f})"
        cx = f"(iw-out_w)*({fx_expr})"
        cy = f"(ih-out_h)*({fy_expr})"
        return f"crop={cw}:{ch}:{cx}:{cy},"

    if spec.zoom <= 1.000001:
        return ""
    z = spec.zoom
    fx = max(0.0, min(1.0, spec.focus_x))
    fy = max(0.0, min(1.0, spec.focus_y))
    cw = f"iw/{z:.4f}"
    ch = f"min(ih\\,iw/{z:.4f}*{canvas_h_over_w:.4f})"
    cx = f"(iw-out_w)*{fx:.4f}"
    cy = f"(ih-out_h)*{fy:.4f}"
    return f"crop={cw}:{ch}:{cx}:{cy},"


def _build_filter(spec: CutSpec) -> str:
    """Construct the `-filter_complex` chain for the chosen style."""
    w, h = spec.width, spec.height
    style = spec.style
    zf = _zoom_inline(spec)

    if style == "blur-bg":
        # Background = ORIGINAL (un-cropped) source, stretched to fill
        # the canvas and blurred heavily. Stretching distorts the
        # aspect (16:9 squashed into 9:16 = ~0.56x horizontal,
        # ~1.78x vertical) but the heavy blur completely hides that --
        # what remains is a soft halo of the actual source colors and
        # rough composition, including content at the LEFT and RIGHT
        # of the original frame that the previous center-crop bg threw
        # away. Foreground = the cropped close-up scaled to canvas
        # width, overlaid centered on top.
        bg_blur = max(spec.blur_strength, 30)  # squashed bg needs more blur
        chain = (
            f"[0:v]split=2[orig][zsrc];"
            f"[zsrc]{zf}scale={w}:-2,setsar=1[fg];"
            f"[orig]scale={w}:{h},boxblur={bg_blur}:2,setsar=1[bg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]"
        )
    elif style == "crop":
        chain = f"[0:v]{zf}scale=-2:{h},crop={w}:{h},setsar=1[v0]"
    elif style == "letterbox":
        chain = (
            f"[0:v]{zf}scale={w}:-2,setsar=1[fg];"
            f"color=c={spec.bg_color}:s={w}x{h}:r=30:d={spec.duration:.3f},"
            f"format=yuv420p[bg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]"
        )
    else:
        raise ValueError(f"unknown style: {style!r}")

    # Tail: optionally burn subtitles
    if spec.ass_path is not None:
        chain += f";[v0]subtitles={_ass_for_filter(spec.ass_path)}[vout]"
    else:
        chain += ";[v0]copy[vout]"

    return chain


def burn_captions_onto(
    input_video: Path,
    ass_path: Path,
    output_path: Path,
    *,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Burn an ASS caption file onto an already-rendered clip.

    Cheaper than re-running the full cut: video gets re-encoded once
    through libass, audio is stream-copied. Used by the web UI's Save
    flow to produce a captioned version while leaving the raw preview
    mp4 untouched.
    """
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    if not ass_path.exists():
        raise FileNotFoundError(ass_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel", "error",
        "-stats",
        "-i", str(input_video),
        "-vf", f"subtitles={_ass_for_filter(ass_path)}",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info("burn: %s", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg finished but {output_path} is missing")
    return output_path


def cut_clip(spec: CutSpec) -> Path:
    """Run ffmpeg to produce one short clip. Returns the output path."""
    if not spec.src_video.exists():
        raise FileNotFoundError(spec.src_video)
    if spec.duration <= 0.05:
        raise ValueError(f"clip duration too short: {spec.duration:.3f}s")
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = _build_filter(spec)

    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output
        "-nostdin",
        "-loglevel", "error",
        "-stats",                   # progress on stderr
        "-ss", f"{spec.start_s:.3f}",       # accurate seek (after -i for accuracy)
        "-to", f"{spec.end_s:.3f}",
        "-i", str(spec.src_video),
        "-filter_complex", filter_chain,
        "-map", "[vout]",
        "-map", "0:a:0?",
        # Video encode
        "-c:v", "libx264",
        "-preset", spec.preset,
        "-pix_fmt", "yuv420p",
    ]
    if spec.video_bitrate:
        cmd += ["-b:v", spec.video_bitrate, "-maxrate", spec.video_bitrate,
                "-bufsize", "12M"]
    else:
        cmd += ["-crf", str(spec.crf)]

    cmd += [
        "-c:a", "aac",
        "-b:a", spec.audio_bitrate,
        "-movflags", "+faststart",
        str(spec.out_path),
    ]

    log.info("ffmpeg: %s", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (rc={e.returncode}); see log above") from e

    if not spec.out_path.exists():
        raise RuntimeError(f"ffmpeg finished but {spec.out_path} is missing")
    return spec.out_path
