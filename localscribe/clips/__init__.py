"""Clip generation: turn transcribed videos into vertical shorts with captions.

A separate stage that depends on the main pipeline having already run so
that `transcript.raw.json` exists with word-level timestamps. The original
video stream is downloaded on-demand (the main pipeline only pulls audio).
"""
