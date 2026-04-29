# localscribe

Local video summarizer with speaker diarization.

Runs transcription and diarization locally on your GPU. Only the resulting
text transcript is sent to Claude (via the `claude` CLI) for summarization —
video and audio never leave your machine.

## Pipeline

1. **Download** — `yt-dlp` pulls audio + metadata
2. **Transcribe** — `faster-whisper` (large-v3) or `whisper.cpp` (Metal/Vulkan), with word timestamps
3. **Diarize** — `pyannote.audio` 3.1, assigns speaker turns
4. **Align** — merge transcript words with speaker turns
5. **Identify speakers** — `claude -p` scans the transcript for self-intros
6. **Summarize** — chunked summarization via `claude -p`, then final synthesis
7. **Report** — render a single self-contained HTML page from the markdown outputs

## What you get

Under `output/<video-id>/`:

- `report.html` — single self-contained HTML report (open this one)
- `transcript.md` — full transcript with speaker labels + timestamps

The remaining outputs are chosen by content type, which Claude
auto-detects in the identify stage (override with `--content-type`):

| Content type | When it's picked | Outputs |
|---|---|---|
| **debate** | Formal debate with opposing positions, structured rounds, an explicit resolution being argued, usually with a moderator. | `tldr`, `per_speaker`, `chronological`, `debate_map` |
| **interview** | One interviewer asking questions of one or more guests; asymmetric Q&A. | `tldr`, `topics`, `per_speaker`, `chronological`, `highlights` |
| **panel** | Three or more participants on shared topics with a moderator; cooperative rather than adversarial. | `tldr`, `topics`, `per_speaker`, `chronological`, `highlights` |
| **discussion** | Two to a few speakers conversing without formal structure; symmetric turn-taking. | `tldr`, `topics`, `per_speaker`, `chronological`, `highlights` |
| **monologue** | A single speaker (lecture, talk, tutorial, solo podcast); may include brief Q&A. | `tldr`, `key_points`, `outline`, `quotes` |

What each output is:

- `tldr.md` — short TL;DR
- `per_speaker.md` — each speaker's positions, arguments, or contributions
- `chronological.md` — timeline summary
- `topics.md` — topic-by-topic breakdown with timestamps
- `highlights.md` — notable moments and quotes
- `debate_map.md` — structured claims / rebuttals / evidence
- `key_points.md` — main points from the talk
- `outline.md` — hierarchical outline of the content
- `quotes.md` — selected quotations

Cached intermediates (`audio.wav`, `transcript.raw.json`, `diarization.json`,
`transcript.aligned.json`, `speakers.json`, per-chunk summary notes) live
alongside, so re-running is cheap.

## Hardware support

`setup.sh` auto-detects the best transcription + diarization backend for
your hardware. Override with `--cpu`, `--cuda=cuXYZ`, `--mps`, or `--vulkan`.

| OS / hardware | Transcription | Diarization | Status |
|---|---|---|---|
| Linux + NVIDIA (driver ≥ 570) | faster-whisper (CUDA) | pyannote (CUDA) | **Tested** (cu128 path) |
| Linux + NVIDIA (driver 525–569) | faster-whisper (CUDA) | pyannote (CUDA) | Untested (cu121/cu124 wheel paths) |
| Linux + no GPU | faster-whisper (CPU) | pyannote (CPU) | Untested end-to-end |
| Linux + AMD/Intel + Vulkan | whisper.cpp (Vulkan) | pyannote (CPU) | **Untested** — pywhispercpp source build with `-DGGML_VULKAN=on` |
| macOS Apple Silicon | whisper.cpp (Metal) | pyannote (MPS) | **Untested** — pywhispercpp Metal wheel + pyannote MPS fallback |
| macOS Intel | faster-whisper (CPU) | pyannote (CPU) | Untested |
| Windows | — | — | Use WSL2 |

If you're on one of the untested rows and something breaks, please file
an issue with logs — the install + runtime paths are wired up but no one
has verified them on real hardware yet.

NVIDIA users always get `faster-whisper` for transcription because
ctranslate2 + CUDA outperforms whisper.cpp + CUDA. Everywhere else,
whisper.cpp is preferred when its build supports a GPU backend
(Metal on Apple Silicon, Vulkan on Linux), since whisper.cpp is the
only path to GPU-accelerated transcription off NVIDIA. If neither GPU
path is available, faster-whisper on CPU is still the fastest CPU
engine.

ROCm is not supported. AMD users on Linux should rely on Vulkan.

### Approximate runtime

Rough numbers for processing 1 hour of audio with `--model large-v3`.
These are estimates, not benchmarks — actual mileage varies a lot with
audio quality, number of speakers, and host load.

| Backend | Transcribe | Diarize | Total |
|---|---|---|---|
| RTX 3060 12 GB (faster-whisper, CUDA) | ~10–15 min | ~3–5 min | ~15–20 min |
| Modern x86 CPU, 16 cores (faster-whisper, int8) | ~30–60 min | ~10–20 min | ~45–75 min |
| Apple Silicon M2/M3 (whisper.cpp Metal + pyannote MPS) | ~15–25 min | ~5–10 min | ~25–35 min |
| AMD/Intel + Vulkan (whisper.cpp Vulkan + pyannote CPU) | ~15–30 min | ~10–20 min | ~30–50 min |

Drop to `--model medium` or `--model small` for faster iteration; quality
degrades but is often fine for clear-audio content.

## LLM backend

localscribe needs an LLM for the speaker-identify and summarize stages.
Four backends ship out of the box:

| Backend      | Best for                                  | Binary / auth | Status |
|--------------|-------------------------------------------|---------------|--------|
| `claude_cli` (default) | Claude Pro / Max subscription   | `claude` (Claude Code) | **Tested** |
| `codex_cli`  | ChatGPT Plus / Pro / Team subscription    | `codex` (OpenAI Codex CLI) → `codex login` | **Untested** — flags may drift |
| `gemini_cli` | Gemini Advanced / Google AI Pro subscription | `gemini` (Google Gemini CLI) → `gemini auth` | **Untested** — flags may drift |
| `openai_api` | Everyone else; any OpenAI-compatible HTTP API | API key (or none for local Ollama/LM Studio) | **Tested** dispatch, untested response handling on every provider |

The three `*_cli` backends use the corresponding subscription via that
vendor's official CLI — no API credits are consumed. The `openai_api`
backend speaks the OpenAI Chat Completions schema, which is the de
facto standard for LLM HTTP APIs. Confirmed-working endpoints include
OpenAI, the Anthropic API's OpenAI-compat shim,
[Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), vLLM,
[Groq](https://groq.com/), and [OpenRouter](https://openrouter.ai/).

**Note on system prompts**: `claude_cli` passes the system prompt via
`--append-system-prompt` and `openai_api` uses a `system` message.
`codex_cli` and `gemini_cli` have no equivalent flag, so localscribe
prepends the system instructions to the user prompt. This is slightly
less reliable for instruction adherence than a real system message —
mostly noticeable in the `identify` stage's JSON requirement. If JSON
parsing fails repeatedly with one of the CLI backends, falling back to
`claude_cli` or `openai_api` for that stage is the easy fix.

Set the backend via env vars (or `.env` — `setup.sh` writes both for you):

```bash
LOCALSCRIBE_LLM_BACKEND=openai_api
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # default
OPENAI_MODEL=gpt-4o                          # default
```

For local Ollama:

```bash
LOCALSCRIBE_LLM_BACKEND=openai_api
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:70b
# OPENAI_API_KEY can be blank for Ollama
```

Smaller local models (7B-class) often struggle to return clean JSON,
which the identify stage requires. If identify fails, try a larger model
(70B+) or fall back to `claude_cli` for that one stage by toggling the
env var.

## Setup

```bash
./setup.sh                    # auto-detects hardware, prompts for LLM backend
./setup.sh --check            # just verify prerequisites, don't install
./setup.sh --cpu              # force CPU-only install
./setup.sh --vulkan           # force Linux+Vulkan (whisper.cpp from source)
./setup.sh --llm=openai_api   # skip the LLM prompt; pick openai_api non-interactively
# You'll need a HuggingFace token with access to pyannote/speaker-diarization-3.1
# (accept the gated model terms at https://hf.co/pyannote/speaker-diarization-3.1)
export HF_TOKEN=hf_xxx
```

## Usage

```bash
source .venv/bin/activate

# Process a YouTube URL (default path)
localscribe https://www.youtube.com/watch?v=dQw4w9WgXcQ

# A bare 11-character video ID also works
localscribe dQw4w9WgXcQ

# Process a local audio or video file
localscribe ~/Downloads/talk.mp3

# Pick a smaller, faster model for testing
localscribe --model medium <url>

# Stop after a specific stage (handy for debugging mid-pipeline)
localscribe --stop-after diarize <url>

# Re-run a single stage (e.g. you tweaked the speaker-id prompt)
localscribe --force-stage identify --force-stage summarize <url>

# Override the auto-detected content type
localscribe --content-type interview <url>

# Quieter output (warnings + errors only) or verbose (DEBUG)
localscribe --quiet <url>
localscribe --verbose <url>

# Wipe the cached output dir for a video and exit
localscribe --clean <url-or-id>
```

Each stage caches its output, so re-running is cheap.

## Troubleshooting

**`No HuggingFace token found`** — pyannote's diarization model is
gated. Accept the terms at
<https://hf.co/pyannote/speaker-diarization-3.1> and
<https://hf.co/pyannote/segmentation-3.0>, then either run
`huggingface-cli login` or set `HF_TOKEN=hf_...` in `.env`.

**`claude CLI not found on PATH`** — only relevant when
`LOCALSCRIBE_LLM_BACKEND=claude_cli`. Either install Claude Code (`npm
install -g @anthropic-ai/claude-code`, see <https://claude.com/claude-code>)
or switch to one of the other backends (see "LLM backend" above).

**`codex CLI returned empty output`** — `codex 0.124.0` introduced a
regression where `codex exec` can silently crash when stdio is detached
from a TTY (which is exactly what `subprocess.run` does). See
<https://github.com/openai/codex/issues/19945>. Workarounds: downgrade
codex, or switch to `claude_cli` / `openai_api`.

**`openai_api 401`/`403`** — bad or missing `OPENAI_API_KEY`. For local
endpoints (Ollama, LM Studio) the key can be blank; for hosted services
it's required.

**`openai_api: unexpected response shape`** — the configured endpoint
returned something that isn't a Chat Completions response. Common cause:
`OPENAI_BASE_URL` is missing the `/v1` suffix, or points at a
non-OpenAI-compatible endpoint.

**CUDA out of memory** — the default `large-v3` whisper model needs
~10 GB of VRAM with batched inference. On an 8 GB card, drop to `--model
medium`. On 6 GB, `--model small`. Lowering `BATCH_SIZE` in
`localscribe/stages/transcribe.py` is another lever.

**`weights_only=False` warning during diarize** — expected. Pyannote's
checkpoints contain Python globals (`TorchVersion`, `Specifications`,
etc.) that torch 2.6+'s safe-unpickler rejects, so `diarize.py` forces
`weights_only=False` while loading. Pyannote checkpoints come from a
gated, authenticated HF repo, so trusting them is acceptable.

**yt-dlp blocks with "bot detection"** — export browser cookies to a
Netscape-format file (e.g. via the *Get cookies.txt* extension) and set
`YOUTUBE_COOKIES=/path/to/cookies.txt`. See `.env.example`.

**Half-broken pipeline state from an earlier crash** — `localscribe
--clean <url-or-id>` wipes the cached output dir for that video.

## Roadmap

Things that would be nice but aren't here yet:

- **Subtitle export (SRT/VTT)** with speaker prefixes — the aligned
  transcript already has everything needed; this is just a new
  `report.py` output.
- **Verified Apple Silicon and Vulkan support** — install + runtime
  paths are wired up but no one has run them on real hardware yet.
- **CPU-only end-to-end smoke test** — useful as a CI sanity check that
  doesn't require a GPU.

## Acknowledgments

localscribe stitches together a stack of upstream projects:

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — audio + metadata extraction
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fast
  Whisper inference on NVIDIA / CPU via
  [ctranslate2](https://github.com/OpenNMT/CTranslate2)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and
  [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp) — portable
  Whisper inference (Metal / Vulkan / CPU)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker
  diarization
- [Anthropic Claude](https://www.anthropic.com/claude) and
  [Claude Code](https://claude.com/claude-code) — speaker
  identification, content classification, and summarization

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
