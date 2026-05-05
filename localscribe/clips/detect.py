"""Optional speaker-face detection for auto-zoom.

Samples a handful of evenly-spaced frames across a clip's time range
(via ffmpeg) and runs OpenCV's bundled haar cascade to find the
largest face in each. The aggregated face box drives a sensible
`zoom` factor and `focus_x/y` for `cut.CutSpec` so the speaker fills
more of the final 9:16 frame.

Why haar cascades, not a DNN: they ship with `opencv-python-headless`
(no extra model download), and for a single, head-on speaker in
typical podcast lighting the false-negative rate is fine. If detection
fails we just fall back to zoom=1.0 instead of guessing.
"""
from __future__ import annotations

import logging
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("clips.detect")


@dataclass
class FaceBox:
    """Normalized face box (all values in [0..1]) plus metadata."""
    x_norm: float    # left edge as fraction of source width
    y_norm: float    # top edge as fraction of source height
    w_norm: float    # width as fraction of source width
    h_norm: float    # height as fraction of source height
    n_detected: int  # frames with a hit
    n_samples: int   # frames probed
    src_w: int       # source pixel width (for sanity / logging)
    src_h: int       # source pixel height

    @property
    def cx(self) -> float:
        return self.x_norm + self.w_norm / 2

    @property
    def cy(self) -> float:
        return self.y_norm + self.h_norm / 2


# ---------------------------------------------------------------------------
# Frame extraction + detection
# ---------------------------------------------------------------------------

def _grab_frame(video: Path, time_s: float, out: Path) -> bool:
    """Extract a single frame at `time_s` to `out`. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{time_s:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return False
    return out.exists() and out.stat().st_size > 0


def _detect_largest_face(frame_path: Path) -> tuple[float, float, float, float] | None:
    """Run haar cascade on one image. Returns (x, y, w, h) in pixels for
    the largest detected face, or None if nothing is found.
    """
    import cv2
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    h_img = img.shape[0]
    # minSize: don't trip on tiny noisy detections; ~6% of frame height
    min_side = max(40, int(h_img * 0.06))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    if len(faces) == 0:
        return None
    largest = max(faces, key=lambda f: f[2] * f[3])
    return tuple(float(v) for v in largest)


def detect_face_in_clip(
    video: Path,
    start_s: float,
    end_s: float,
    *,
    n_samples: int = 6,
) -> FaceBox | None:
    """Sample `n_samples` evenly-spaced frames from `[start_s, end_s]`,
    detect the largest face in each, and aggregate to a single
    normalized FaceBox.

    Returns None if too few frames had detections.
    """
    duration = max(0.0, end_s - start_s)
    if duration <= 0.05:
        return None

    boxes: list[tuple[float, float, float, float]] = []
    src_size: tuple[int, int] | None = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(n_samples):
            t = start_s + duration * (i + 0.5) / n_samples
            frame_path = tmp_dir / f"f{i}.png"
            if not _grab_frame(video, t, frame_path):
                continue

            # Read source dimensions once
            if src_size is None:
                import cv2
                img = cv2.imread(str(frame_path))
                if img is not None:
                    src_size = (img.shape[1], img.shape[0])

            box = _detect_largest_face(frame_path)
            if box is None or src_size is None:
                continue
            x, y, w, h = box
            boxes.append((x / src_size[0], y / src_size[1],
                          w / src_size[0], h / src_size[1]))

    if not boxes or src_size is None:
        log.info("no faces detected in %d sampled frames", n_samples)
        return None
    if len(boxes) < max(2, n_samples // 3):
        # Too unreliable -- fall back rather than zoom in on a blip
        log.info("only %d/%d frames had detections; skipping auto-zoom",
                 len(boxes), n_samples)
        return None

    return FaceBox(
        x_norm=statistics.median(b[0] for b in boxes),
        y_norm=statistics.median(b[1] for b in boxes),
        w_norm=statistics.median(b[2] for b in boxes),
        h_norm=statistics.median(b[3] for b in boxes),
        n_detected=len(boxes),
        n_samples=n_samples,
        src_w=src_size[0],
        src_h=src_size[1],
    )


# ---------------------------------------------------------------------------
# Translate a face box to (zoom, focus_x, focus_y)
# ---------------------------------------------------------------------------

def detect_face_track(
    video: Path,
    start_s: float,
    end_s: float,
    *,
    sample_dt: float = 1.0,
    smoothing: int = 3,
) -> dict | None:
    """Sample faces at a regular cadence across `[start_s, end_s]` so
    the caller can build a time-varying crop instead of a single static
    one.

    Returns `None` if too few frames had a detection. Otherwise returns:

        {
          "src_w": int, "src_h": int,
          "frames": [
            {"t": float (clip-relative seconds),
             "x_norm": float, "y_norm": float,
             "w_norm": float, "h_norm": float},
            ...
          ],
          "n_detected": int, "n_sampled": int
        }

    A median filter of width `smoothing` is applied to each component
    independently so a single jittery detection doesn't yank the crop.
    Frames where detection failed are silently skipped; the caller's
    interpolator will bridge the gap.
    """
    duration = max(0.0, end_s - start_s)
    if duration <= 0.05:
        return None

    n_samples = max(2, int(round(duration / sample_dt)) + 1)
    raw: list[tuple[float, float, float, float, float]] = []  # (t_rel, x, y, w, h) px
    src_size: tuple[int, int] | None = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(n_samples):
            t_abs = start_s + duration * i / max(1, n_samples - 1)
            t_rel = t_abs - start_s
            frame_path = tmp_dir / f"f{i}.png"
            if not _grab_frame(video, t_abs, frame_path):
                continue
            if src_size is None:
                import cv2
                img = cv2.imread(str(frame_path))
                if img is not None:
                    src_size = (img.shape[1], img.shape[0])
            box = _detect_largest_face(frame_path)
            if box is None or src_size is None:
                continue
            x, y, w, h = box
            raw.append((t_rel, x, y, w, h))

    if not src_size or len(raw) < 2:
        log.info("face track: only %d/%d frames had detections; "
                 "falling back to static auto-zoom", len(raw), n_samples)
        return None

    sw, sh = src_size
    frames: list[dict] = [
        {"t": t, "x_norm": x / sw, "y_norm": y / sh,
         "w_norm": w / sw, "h_norm": h / sh}
        for t, x, y, w, h in raw
    ]

    # Median-smooth each axis to dampen noisy single-frame detections.
    if smoothing >= 3 and len(frames) >= smoothing:
        half = smoothing // 2
        smoothed = []
        for i in range(len(frames)):
            lo = max(0, i - half)
            hi = min(len(frames), i + half + 1)
            win = frames[lo:hi]
            smoothed.append({
                "t": frames[i]["t"],
                "x_norm": statistics.median(f["x_norm"] for f in win),
                "y_norm": statistics.median(f["y_norm"] for f in win),
                "w_norm": statistics.median(f["w_norm"] for f in win),
                "h_norm": statistics.median(f["h_norm"] for f in win),
            })
        frames = smoothed

    log.info("face track: %d/%d frames detected, %.1fs span",
             len(frames), n_samples, duration)
    return {
        "src_w": sw,
        "src_h": sh,
        "frames": frames,
        "n_detected": len(frames),
        "n_sampled": n_samples,
    }


def track_to_zoom_keyframes(
    track: dict,
    *,
    target_face_w_frac: float = 0.45,
    max_zoom: float = 4.0,
    headroom_frac: float = 0.10,
) -> list[dict]:
    """Convert a face track into a list of crop keyframes.

    Each keyframe has clip-relative time `t` and normalized crop
    parameters (zoom, focus_x, focus_y). Caller may collapse these to
    one keyframe per shot (see `cluster_to_shots`) -- the framing is
    deliberately a touch loose (face = ~45% of the cropped frame) so
    subtle head movements don't appear to nudge the camera.
    """
    out = []
    for f in track["frames"]:
        fw = max(0.0001, f["w_norm"])
        zoom = max(1.0, min(max_zoom, target_face_w_frac / fw))
        cx = f["x_norm"] + f["w_norm"] / 2
        cy = f["y_norm"] + f["h_norm"] / 2 + headroom_frac / zoom
        out.append({
            "t": f["t"],
            "zoom": float(zoom),
            "focus_x": max(0.0, min(1.0, float(cx))),
            "focus_y": max(0.0, min(1.0, float(cy))),
        })
    return out


def cluster_to_shots(
    keyframes: list[dict],
    *,
    position_jump: float = 0.15,
    zoom_jump: float = 0.6,
) -> list[list[dict]]:
    """Group consecutive keyframes into shots.

    A new shot starts whenever the face's normalized position jumps by
    more than `position_jump` (in either x or y) from one sample to the
    next, OR when the implied zoom changes by more than `zoom_jump` --
    both are signs of a hard camera cut to a new angle.
    """
    if not keyframes:
        return []
    shots = [[keyframes[0]]]
    for prev, curr in zip(keyframes, keyframes[1:]):
        dx = abs(curr["focus_x"] - prev["focus_x"])
        dy = abs(curr["focus_y"] - prev["focus_y"])
        dz = abs(curr["zoom"] - prev["zoom"])
        if dx > position_jump or dy > position_jump or dz > zoom_jump:
            shots.append([curr])
        else:
            shots[-1].append(curr)
    return shots


def _binary_search_cut(
    video: Path,
    start_s: float,
    src_w: int,
    t_lo: float,
    t_hi: float,
    prev_focus_x: float,
    next_focus_x: float,
    *,
    max_iters: int = 6,
    min_gap: float = 0.04,
) -> float:
    """Binary search the clip-relative time interval [t_lo, t_hi] for
    the exact frame where the face position transitions from
    `prev_focus_x` to `next_focus_x`.

    Each iteration extracts one frame at the midpoint and runs the haar
    cascade. ~6 iterations narrows a 1s gap down to ~16ms (≈0.4 frames
    at 24fps). Returns the refined cut time in clip-relative seconds.
    """
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(max_iters):
            if t_hi - t_lo < min_gap:
                break
            t_mid = (t_lo + t_hi) / 2
            frame = tmp_dir / f"r{i}.png"
            t_abs = start_s + t_mid
            if not _grab_frame(video, t_abs, frame):
                return t_hi
            face = _detect_largest_face(frame)
            if face is None:
                # Detection failed at this frame -- often a transition
                # frame. Bias toward the next-shot side.
                t_hi = t_mid
                continue
            x_norm = (face[0] + face[2] / 2) / src_w
            d_prev = abs(x_norm - prev_focus_x)
            d_next = abs(x_norm - next_focus_x)
            if d_prev < d_next:
                # Still in previous shot
                t_lo = t_mid
            else:
                # Already on the next shot side
                t_hi = t_mid
    return t_hi


def refine_shot_boundaries(
    video: Path,
    start_s: float,
    src_size: tuple[int, int],
    shots: list[list[dict]],
    *,
    max_iters: int = 6,
) -> list[list[dict]]:
    """Pin each shot transition to the actual cut frame using binary
    search. Mutates `shots` in place: shifts `shots[i][0]['t']` (the
    start of each shot after the first) to the refined boundary.
    """
    sw, _sh = src_size
    for i in range(1, len(shots)):
        prev_shot = shots[i - 1]
        next_shot = shots[i]
        if not prev_shot or not next_shot:
            continue
        t_lo = prev_shot[-1]["t"]
        t_hi = next_shot[0]["t"]
        if t_hi - t_lo < 0.06:
            continue  # already snug
        prev_fx = statistics.median(f["focus_x"] for f in prev_shot)
        next_fx = statistics.median(f["focus_x"] for f in next_shot)
        refined = _binary_search_cut(
            video, start_s, sw, t_lo, t_hi, prev_fx, next_fx,
            max_iters=max_iters,
        )
        log.info("refined boundary: shot %d/%d  %.2fs (gap %.2fs) -> %.3fs",
                 i, len(shots) - 1, t_hi, t_hi - t_lo, refined)
        next_shot[0] = {**next_shot[0], "t": float(refined)}
    return shots


def shots_to_static_keyframes(shots: list[list[dict]]) -> list[dict]:
    """Collapse each shot into a single stable keyframe carrying that
    shot's median zoom and focus point.

    The returned list is suitable for a step-function ffmpeg
    expression: hold each value until the next shot's start time, then
    snap. No mid-shot interpolation -- so the crop stays still while
    the camera is on one angle, then jumps when the camera cuts.
    """
    out: list[dict] = []
    for shot in shots:
        if not shot:
            continue
        # Anchor each shot's keyframe at its first detected sample so
        # the step function flips to the new value at the cut, not at
        # the midpoint.
        out.append({
            "t": float(shot[0]["t"]),
            "zoom": float(statistics.median(f["zoom"] for f in shot)),
            "focus_x": float(statistics.median(f["focus_x"] for f in shot)),
            "focus_y": float(statistics.median(f["focus_y"] for f in shot)),
            "n_frames": len(shot),
        })
    return out


def auto_zoom_params(
    face: FaceBox,
    *,
    target_face_w_frac: float = 0.55,
    max_zoom: float = 4.0,
    headroom_frac: float = 0.10,
) -> dict:
    """Pick a zoom + focus point that frames the speaker comfortably.

    target_face_w_frac:
        Desired face width as a fraction of the *post-zoom* frame
        width (which equals the 9:16 fg width in blur-bg style).
        ~0.22 puts the face at a comfortable medium-shot size.
    max_zoom:
        Ceiling so we don't punch in so far the upscaling looks soft.
    headroom_frac:
        Shift the focus point downward from the face center by this
        fraction of the *cropped* frame height. This lifts the face
        slightly above the vertical midpoint of the output -- the
        classic "rule of thirds" headroom rather than a perfectly
        centered face.
    """
    fw = max(0.0001, face.w_norm)
    zoom = target_face_w_frac / fw
    zoom = max(1.0, min(max_zoom, zoom))

    fx = face.cx
    fy = face.cy + headroom_frac / zoom
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))

    return {"zoom": float(zoom), "focus_x": float(fx), "focus_y": float(fy)}
