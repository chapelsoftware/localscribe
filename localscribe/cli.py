"""localscribe CLI."""
from __future__ import annotations

import shutil

import click

from . import __version__
from .config import OUTPUT_ROOT
from .log import setup_logging
from .stages import align, diarize, download, identify, report, summarize, transcribe

STAGE_NAMES = ["download", "transcribe", "diarize", "align",
               "identify", "summarize", "report"]

CONTENT_TYPE_CHOICES = ["auto", "debate", "interview", "panel",
                        "discussion", "monologue"]


@click.command()
@click.version_option(__version__, prog_name="localscribe")
@click.argument("source")
@click.option("--force", is_flag=True, help="Ignore caches and re-run every stage.")
@click.option("--force-stage", multiple=True,
              type=click.Choice(STAGE_NAMES),
              help="Force re-run of a specific stage (can be repeated).")
@click.option("--model", default="large-v3", help="Whisper model size.")
@click.option("--content-type", type=click.Choice(CONTENT_TYPE_CHOICES),
              default="auto",
              help="Force a content type (default: auto-detected by Claude).")
@click.option("--stop-after", type=click.Choice(STAGE_NAMES),
              help="Stop after the given stage.")
@click.option("--clean", is_flag=True,
              help="Delete the cached output dir for SOURCE and exit. Useful "
                   "for clearing a half-broken pipeline run.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose (DEBUG) logging.")
@click.option("-q", "--quiet", is_flag=True, help="Quiet (WARNING+) logging.")
def main(source: str, force: bool, force_stage: tuple[str, ...], model: str,
         content_type: str, stop_after: str | None, clean: bool,
         verbose: bool, quiet: bool) -> None:
    """Summarize a YouTube video or local audio/video file with speaker diarization.

    SOURCE is a YouTube URL, an 11-character YouTube video ID, or a path to
    a local audio/video file.
    """
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")
    setup_logging(1 if verbose else (-1 if quiet else 0))

    if clean:
        video_id = download.resolve_video_id(source)
        target = OUTPUT_ROOT / video_id
        if target.exists():
            shutil.rmtree(target)
            click.echo(f"Removed {target}")
        else:
            click.echo(f"Nothing to clean: {target} does not exist")
        return

    force_all = force
    forced = set(force_stage)

    def f(stage: str) -> bool:
        return force_all or stage in forced

    paths = download.run(source, force=f("download"))
    if stop_after == "download":
        return

    transcribe.run(paths, model_size=model, force=f("transcribe"))
    if stop_after == "transcribe":
        return

    diarize.run(paths, force=f("diarize"))
    if stop_after == "diarize":
        return

    align.run(paths, force=f("align"))
    if stop_after == "align":
        return

    ct_override = None if content_type == "auto" else content_type
    identify.run(paths, force=f("identify"),
                 content_type_override=ct_override)
    if stop_after == "identify":
        return

    summarize.run(paths, force=f("summarize"))
    if stop_after == "summarize":
        return

    report.run(paths, force=f("report"))

    click.echo("")
    click.echo(f"Done. Outputs in: {paths.root}")
    click.echo(f"  - {paths.report_html}  <-- open this one")
    # List whichever markdown outputs actually exist for this content type
    for md_file in sorted(paths.root.glob("*.md")):
        click.echo(f"  - {md_file}")
    # Note snapshots if any exist
    if paths.snapshots_dir.exists():
        snaps = sorted(p for p in paths.snapshots_dir.iterdir() if p.is_dir())
        if snaps:
            click.echo(f"  ({len(snaps)} snapshot{'s' if len(snaps) != 1 else ''} "
                       f"in {paths.snapshots_dir})")


if __name__ == "__main__":
    main()
