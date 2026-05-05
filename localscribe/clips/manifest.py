"""Per-video clips manifest: `<output>/<video_id>/clips.json`.

A manifest tracks every clip the user has defined for a video. Caption
arrays are persisted (clip-relative times) so the web review UI can edit
them and the CLI can re-render with the edited captions.

Filesystem layout:

    output/
      _completed/                   UNIVERSAL: all archived clips'
                                    captioned mp4s, named
                                    "<video_id>__<clip_id>.captioned.mp4"
      <video_id>/
        shorts/                     active clips' raw + captioned mp4s + ASS
        _archive/<clip_id>/         archived clips' raw mp4 + ASS
        clips.json                  {clips: [...], archived: [...]}
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import OUTPUT_ROOT, Paths


def manifest_path(paths: Paths) -> Path:
    return paths.root / "clips.json"


def shorts_dir(paths: Paths) -> Path:
    d = paths.root / "shorts"
    d.mkdir(exist_ok=True)
    return d


def completed_root() -> Path:
    """The universal completed/ folder shared across all videos."""
    d = OUTPUT_ROOT / "_completed"
    d.mkdir(exist_ok=True)
    return d


def completed_filename(video_id: str, clip_id: str,
                       suffix: str = ".mp4") -> str:
    """Default filename for a completed clip in the universal folder.

    Returns `<clip_id><suffix>`. Files in `_completed/` are by
    definition captioned finals, so the `.captioned` infix that older
    archives carried is dropped.

    `video_id` is unused for the default name but kept in the signature
    because it's part of the disambiguated form (see
    `unique_completed_filename`).
    """
    return f"{clip_id}{suffix}"


def unique_completed_filename(video_id: str, clip_id: str,
                              manifest: dict,
                              suffix: str = ".mp4") -> str:
    """Pick a non-colliding filename in `_completed/`.

    Tries `<clip_id><suffix>` first. If that path already exists AND
    was placed by a DIFFERENT video (i.e. no entry in `manifest`'s
    own archived list claims it), falls back to
    `<clip_id>--<video_id><suffix>` so the existing file isn't
    clobbered.
    """
    desired = completed_filename(video_id, clip_id, suffix)
    desired_path = completed_root() / desired
    if not desired_path.exists():
        return desired
    # File is there. Did THIS video put it there? If we have an
    # archived entry pointing at it, it's ours -- safe to overwrite.
    expected = f"_completed/{desired}"
    for entry in manifest.get("archived", []) + manifest.get("clips", []):
        r = (entry or {}).get("render") or {}
        if r.get("burned_path") == expected:
            return desired
    # Collision with a clip from a different source video.
    return f"{clip_id}--{video_id}{suffix}"


def archive_root(paths: Paths) -> Path:
    return paths.root / "_archive"


def slugify(text: str) -> str:
    """Turn a human title into a stable filesystem-safe id."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "clip"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_manifest(video_id: str) -> dict:
    return {"video_id": video_id, "clips": [], "archived": []}


def load(paths: Paths) -> dict:
    """Load or create the manifest. Always returns a dict."""
    p = manifest_path(paths)
    if not p.exists():
        return empty_manifest(paths.video_id)
    data = json.loads(p.read_text())
    data.setdefault("video_id", paths.video_id)
    data.setdefault("clips", [])
    data.setdefault("archived", [])
    return data


def save(paths: Paths, manifest: dict) -> None:
    p = manifest_path(paths)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def backup(paths: Paths) -> Path | None:
    """Snapshot clips.json next to it as `clips.<utc_ts>.bak.json`.

    Returns the backup path, or None if there's no manifest yet.
    Cheap insurance before destructive operations like wipe-and-replace.
    """
    src = manifest_path(paths)
    if not src.exists():
        return None
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = src.with_name(f"clips.{ts}.bak.json")
    dst.write_bytes(src.read_bytes())
    return dst


def wipe_active(paths: Paths, manifest: dict) -> list[str]:
    """Remove every active clip entry and its rendered files.

    Used by cut-batch --replace before defining a fresh set of clips.
    Backup the manifest BEFORE calling this if you might want it back.
    Returns the list of clip ids that were wiped.
    """
    ids = [c["id"] for c in manifest.get("clips", [])]
    shorts = shorts_dir(paths)
    for cid in ids:
        for ext in ("mp4", "ass", "captioned.mp4", "peaks.json"):
            fp = shorts / f"{cid}.{ext}"
            if fp.exists():
                fp.unlink()
    manifest["clips"] = []
    return ids


def get_clip(manifest: dict, clip_id: str) -> dict | None:
    for c in manifest["clips"]:
        if c["id"] == clip_id:
            return c
    return None


def upsert_clip(manifest: dict, clip: dict) -> dict:
    """Insert or replace a clip by id; return the canonical entry."""
    existing = get_clip(manifest, clip["id"])
    if existing is None:
        manifest["clips"].append(clip)
        return clip
    existing.clear()
    existing.update(clip)
    return existing


def get_archived(manifest: dict, clip_id: str) -> dict | None:
    for c in manifest.get("archived", []):
        if c["id"] == clip_id:
            return c
    return None


def archive_clip(paths: Paths, manifest: dict, clip_id: str) -> dict | None:
    """Move a clip from `clips` to `archived`, relocating its files.

    File moves:
      - <video>/shorts/<id>.mp4           -> <video>/_archive/<id>/<id>.mp4
      - <video>/shorts/<id>.ass           -> <video>/_archive/<id>/<id>.ass
      - <video>/shorts/<id>.captioned.mp4 -> _completed/<id>.mp4
                                             (or <id>--<video>.mp4 on collision)

    The `_completed` folder is at OUTPUT_ROOT, shared across all
    videos. By default the captioned final lands at `_completed/<id>.mp4`
    so the file name matches the clip's slug. If a clip with that
    slug from a DIFFERENT source video is already there,
    `unique_completed_filename` falls back to a disambiguated name.
    """
    clip = get_clip(manifest, clip_id)
    if clip is None:
        return None

    shorts = shorts_dir(paths)
    completed = completed_root()
    archive_dir = archive_root(paths) / clip_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    raw_src = shorts / f"{clip_id}.mp4"
    if raw_src.exists():
        shutil.move(str(raw_src), str(archive_dir / raw_src.name))

    ass_src = shorts / f"{clip_id}.ass"
    if ass_src.exists():
        shutil.move(str(ass_src), str(archive_dir / ass_src.name))

    cap_src = shorts / f"{clip_id}.captioned.mp4"
    burned_filename = None
    if cap_src.exists():
        burned_filename = unique_completed_filename(
            paths.video_id, clip_id, manifest,
        )
        shutil.move(str(cap_src), str(completed / burned_filename))

    archived_entry = dict(clip)
    archived_entry["archived_at"] = now_iso()
    if burned_filename:
        if archived_entry.get("render"):
            archived_entry["render"] = dict(archived_entry["render"])
            # burned_path is the universal-completed filename. The UI
            # constructs URLs as `/completed/<filename>`.
            archived_entry["render"]["burned_path"] = (
                f"_completed/{burned_filename}"
            )

    manifest["clips"] = [c for c in manifest["clips"] if c["id"] != clip_id]
    manifest.setdefault("archived", []).append(archived_entry)
    return archived_entry


def restore_clip(paths: Paths, manifest: dict, clip_id: str) -> dict | None:
    """Reverse `archive_clip`: move files back to shorts/ and entry back
    to `clips`.
    """
    archived = get_archived(manifest, clip_id)
    if archived is None:
        return None

    shorts = shorts_dir(paths)
    completed = completed_root()
    archive_dir = archive_root(paths) / clip_id

    for ext in (".mp4", ".ass"):
        src = archive_dir / f"{clip_id}{ext}"
        if src.exists():
            shutil.move(str(src), str(shorts / src.name))

    # Pull the captioned mp4 back from wherever the manifest says it
    # lives. Fall back through historical naming conventions:
    #   - `_completed/<clip_id>.mp4`             (current default)
    #   - `_completed/<clip_id>--<video>.mp4`    (collision-disambiguated)
    #   - `_completed/<video>__<clip_id>.captioned.mp4` (pre-rename)
    #   - `<video>/completed/<clip_id>.captioned.mp4`   (per-video legacy)
    stored = (archived.get("render") or {}).get("burned_path") or ""
    candidates: list[Path] = []
    if stored.startswith("_completed/"):
        candidates.append(completed / stored[len("_completed/"):])
    candidates += [
        completed / f"{clip_id}.mp4",
        completed / f"{clip_id}--{paths.video_id}.mp4",
        completed / f"{paths.video_id}__{clip_id}.captioned.mp4",
        paths.root / "completed" / f"{clip_id}.captioned.mp4",
    ]
    for cap_src in candidates:
        if cap_src.exists():
            shutil.move(str(cap_src), str(shorts / f"{clip_id}.captioned.mp4"))
            break

    if archive_dir.exists():
        try:
            archive_dir.rmdir()
        except OSError:
            pass

    restored = {k: v for k, v in archived.items() if k != "archived_at"}
    if restored.get("render"):
        restored["render"] = dict(restored["render"])
        if (shorts / f"{clip_id}.captioned.mp4").exists():
            restored["render"]["burned_path"] = (
                f"shorts/{clip_id}.captioned.mp4"
            )

    manifest["archived"] = [c for c in manifest["archived"] if c["id"] != clip_id]
    manifest.setdefault("clips", []).append(restored)
    return restored


def make_clip_entry(
    *,
    clip_id: str,
    title: str,
    start: float,
    end: float,
    style: str = "blur-bg",
    zoom: float = 1.0,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    face_track: list[dict] | None = None,
    captions: list[dict] | None = None,
    notes: str = "",
) -> dict:
    return {
        "id": clip_id,
        "title": title,
        "start": float(start),
        "end": float(end),
        "style": style,
        "zoom": float(zoom),
        "focus_x": float(focus_x),
        "focus_y": float(focus_y),
        # Time-varying face track keyframes (clip-relative seconds).
        # Empty/None falls back to static zoom/focus_x/focus_y.
        "face_track": face_track or None,
        "captions": captions or [],
        "notes": notes,
        "render": None,  # set after a successful ffmpeg run
    }
