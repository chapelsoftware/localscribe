# Contributing to localscribe

Thanks for the interest. localscribe is a small project, so the bar is
low: keep things simple, match the existing style, and explain in
your PR description.

## Dev setup

```bash
git clone <your-fork>
cd localscribe
./setup.sh                              # creates .venv, installs runtime deps
source .venv/bin/activate
pip install -e ".[dev]"                 # adds pytest
```

The project targets Python 3.10+ and runs end-to-end best on Linux with
an NVIDIA GPU, but pure-logic tests don't need any of that.

## Running tests

```bash
pytest                                  # all logic tests, no GPU/network needed
pytest tests/test_logic.py::test_to_ascii   # a single test
```

The tests cover the pure-Python pieces (URL/ID parsing, transcript
chunking, ASCII transliteration, etc.) — the stages that actually call
torch / yt-dlp / claude aren't unit-tested. If you add logic worth
testing, drop a function in `tests/test_logic.py`.

## Project layout

```
localscribe/
├── localscribe/
│   ├── cli.py            # click entry point + flag parsing
│   ├── config.py         # Paths dataclass, cache helpers, device pickers, env reader
│   ├── log.py            # logging setup
│   ├── text.py           # ASCII transliteration
│   ├── llm/              # LLM backend dispatch
│   │   ├── __init__.py   # ask() facade, picks backend by LOCALSCRIBE_LLM_BACKEND
│   │   ├── claude_cli.py # shells out to `claude -p`
│   │   ├── codex_cli.py  # shells out to `codex exec -`
│   │   ├── gemini_cli.py # shells out to `gemini` (stdin)
│   │   └── openai_api.py # OpenAI-compatible HTTP client (httpx)
│   └── stages/
│       ├── download.py   # 1. yt-dlp / local file -> 16kHz mono WAV
│       ├── transcribe.py # 2. faster-whisper or whisper.cpp -> word JSON
│       ├── diarize.py    # 3. pyannote.audio -> speaker turns
│       ├── align.py      # 4. merge words with speaker turns
│       ├── identify.py   # 5. claude: name speakers + classify content
│       ├── summarize.py  # 6. claude: chunk + final synthesis
│       └── report.py     # 7. render single-page HTML report
├── tests/
├── pyproject.toml
└── setup.sh
```

Each stage is independent: it reads cached artifacts written by earlier
stages, writes its own artifact to `output/<video-id>/`, and reports
status via its own logger (`logging.getLogger("download")`, etc.).

## Adding a new pipeline stage

If you need a new stage:

1. Add `localscribe/stages/yourstage.py` with a `run(paths, force=...)`
   that reads from earlier-stage paths, writes its artifact via
   `save_json` (or your own format), and is idempotent under `cached()`.
2. Wire it into `cli.py`'s `STAGE_NAMES` list and the `main()` dispatch.
3. If it produces a new on-disk artifact, add a `@property` for it in
   `config.py`'s `Paths` dataclass.
4. Use `log = logging.getLogger("yourstage")` for all status output —
   don't call `print()`.

## Adding a new LLM backend

The `localscribe/llm/` package dispatches to one of two backends today
(`claude_cli`, `openai_api`). To add a third (e.g. a provider-specific
SDK or a different protocol):

1. Add `localscribe/llm/yourbackend.py` exporting one function:
   `def ask(prompt: str, *, system: str | None, timeout: int) -> str`.
2. Wire it into the dispatch in `llm/__init__.py`.
3. Document its env vars in `.env.example` and the README's "LLM backend"
   section.
4. Update `setup.sh`'s LLM prompt to offer it, and `check_llm_backend`
   to validate any required prereqs (binaries, config).

If at all possible, prefer extending `openai_api` to a new endpoint over
adding a fourth backend — the OpenAI-compat schema covers most providers.

## Adding a new content type

The `summarize.py` `RECIPES` dict maps a content type to the list of
final outputs that get generated. To add a new type:

1. Extend `CONTENT_TYPES` and `RECIPES` in `summarize.py`.
2. Add the new type to `identify.py`'s prompt and to
   `CONTENT_TYPES` there.
3. Add any new output paths (`xxx_md`) to `Paths` in `config.py`.
4. Update the README's "What you get" section.

## Style

- Two-space-but-actually-four-space Python (PEP 8 with 100-char lines is
  fine). The existing code uses no formatter; matching is enough.
- Default to writing **no comments**. Only add one when why is
  non-obvious — a hidden constraint, a workaround, behaviour that would
  surprise a reader.
- Type hints on public functions; skip them for trivial private helpers
  if it makes the signature noisier.
- Replace `print()` with `log.info(...)` etc. — the CLI's
  `--verbose`/`--quiet` plumbing relies on logging.
- Don't add error handling, fallbacks, or validation for cases that
  can't actually happen. Trust framework guarantees and internal calls.

## Submitting a PR

1. Branch off `main`.
2. Run `pytest`. CI doesn't exist yet (see Roadmap), so this is on you.
3. If you exercised a stage end-to-end (e.g. you actually ran the
   pipeline on a real video with your change), say so in the PR
   description and include the model + backend you tested on.
4. Keep the PR focused. A bug fix shouldn't carry a refactor; a refactor
   shouldn't carry a feature.

## Hardware-specific changes

If you're touching the install / device-picker logic:

- The hardware-support table in the README is the source of truth for
  which configurations the code claims to support. If you add or
  validate a row, update the **Status** column in the same PR.
- `setup.sh` has `--check` for testing prereq detection without
  installing anything. Use it.

## Reporting bugs

There's no issue template yet. When filing an issue, please include:

- OS + distro + arch (`uname -a`, `/etc/os-release`)
- GPU + driver (`nvidia-smi` on Linux NVIDIA, otherwise just say what
  you have)
- Backend `setup.sh` selected (re-run `./setup.sh --check` to see)
- The full `localscribe ...` command and its output (include
  `--verbose` if the issue is mid-pipeline)
- Whether the failure reproduces after `localscribe --clean <id>`
