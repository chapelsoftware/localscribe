"""Text utilities. Currently: ASCII sanitizer for output files."""
from __future__ import annotations

import unicodedata


# Targeted transliterations for characters Claude loves that have good
# multi-character ASCII equivalents. Applied before the NFKD fallback.
_TRANSLIT = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2212": "-",    # minus sign
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote / apostrophe
    "\u201a": "'",    # single low-9 quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u201e": '"',    # double low-9 quote
    "\u2026": "...",  # horizontal ellipsis
    "\u2022": "*",    # bullet
    "\u00b7": "*",    # middle dot
    "\u2192": "->",   # rightwards arrow
    "\u2190": "<-",   # leftwards arrow
    "\u21d2": "=>",   # rightwards double arrow
    "\u00b1": "+/-",  # plus-minus
    "\u2248": "~=",   # almost equal
    "\u00d7": "x",    # multiplication sign
    "\u00a0": " ",    # non-breaking space
    "\u200b": "",     # zero-width space
    "\ufeff": "",     # BOM / zero-width no-break space
    "\u00ae": "(R)",  # registered
    "\u2122": "(TM)", # trademark
    "\u00a9": "(C)",  # copyright
}


def to_ascii(text: str) -> str:
    """Transliterate text to plain ASCII.

    Steps:
      1. Explicit replacements for common punctuation.
      2. NFKD normalize and drop combining marks (strips accents:
         cafe' -> cafe, etc.).
      3. Encode to ASCII, silently dropping anything still non-ASCII.
    """
    if not text:
        return text
    # Step 1: explicit replacements
    for src, dst in _TRANSLIT.items():
        if src in text:
            text = text.replace(src, dst)
    # Step 2: decompose accented letters into base + combining mark
    text = unicodedata.normalize("NFKD", text)
    # Step 3: drop anything left that isn't plain ASCII
    return text.encode("ascii", "ignore").decode("ascii")


def write_ascii(path, text: str) -> None:
    """Write text to path as ASCII, UTF-8 encoded (ASCII is a UTF-8 subset)."""
    path.write_text(to_ascii(text), encoding="utf-8")
