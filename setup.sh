#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ---- Output helpers ----
err()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33mwarn:\033[0m  %s\n' "$*" >&2; }
info() { printf '\033[36m::\033[0m     %s\n' "$*"; }
ok()   { printf '\033[32mok:\033[0m    %s\n' "$*"; }

usage() {
    cat <<EOF
Usage: ./setup.sh [options]

Creates a Python venv and installs localscribe with a torch backend
appropriate for the host hardware, and configures an LLM backend for
speaker identification + summarization.

Hardware auto-detect: Linux+NVIDIA (CUDA), Linux+AMD/Intel (Vulkan via
whisper.cpp), Linux+CPU, macOS Apple Silicon (MPS + Metal whisper.cpp).
Native Windows is not supported (use WSL2).

Options:
  --cpu             Force CPU-only torch wheels.
  --cuda=VER        Force a CUDA wheel index. VER is one of: cu121, cu124, cu128.
  --mps             Force Apple Silicon install (stock PyPI torch + Metal whisper.cpp).
  --vulkan          Force Linux+Vulkan install (CPU torch + whisper.cpp built with Vulkan).
  --llm=BACKEND     LLM backend: claude_cli (default), codex_cli, gemini_cli,
                    or openai_api. Skips the interactive prompt.
  --check           Run prerequisite checks only; do not install anything.
  -h, --help        Show this help and exit.
EOF
}

# ---- Args ----
BACKEND_OVERRIDE=""
LLM_OVERRIDE=""
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --cpu)        BACKEND_OVERRIDE="cpu" ;;
        --mps)        BACKEND_OVERRIDE="mps" ;;
        --vulkan)     BACKEND_OVERRIDE="vulkan" ;;
        --cuda=cu121|--cuda=cu124|--cuda=cu128) BACKEND_OVERRIDE="${arg#--cuda=}" ;;
        --cuda=*)     err "invalid --cuda value (expected cu121, cu124, or cu128)"; exit 2 ;;
        --llm=claude_cli|--llm=codex_cli|--llm=gemini_cli|--llm=openai_api)
            LLM_OVERRIDE="${arg#--llm=}" ;;
        --llm=*)
            err "invalid --llm value (expected claude_cli, codex_cli, gemini_cli, or openai_api)"
            exit 2 ;;
        --check)      CHECK_ONLY=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)            err "unknown argument: $arg"; usage; exit 2 ;;
    esac
done

# ---- OS / distro detection + per-distro install commands ----
OS="$(uname -s)"
ARCH="$(uname -m)"
DISTRO=""
PKG_INSTALL=""
PKG_PYTHON=""
PKG_FFMPEG=""
PKG_VULKAN_BUILD=""    # vulkan loader+headers, glslc, cmake, c++ toolchain

case "$OS" in
    Linux)
        if [ -r /etc/os-release ]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            DISTRO="${ID:-linux}"
        fi
        case "$DISTRO" in
            ubuntu|debian|pop|linuxmint|elementary)
                PKG_INSTALL="sudo apt update && sudo apt install -y"
                PKG_PYTHON="python3 python3-venv"
                PKG_FFMPEG="ffmpeg"
                PKG_VULKAN_BUILD="libvulkan-dev glslang-tools cmake build-essential"
                ;;
            fedora|rhel|centos|rocky|almalinux)
                PKG_INSTALL="sudo dnf install -y"
                PKG_PYTHON="python3"
                PKG_FFMPEG="ffmpeg"
                PKG_VULKAN_BUILD="vulkan-loader-devel vulkan-headers glslang cmake gcc-c++"
                ;;
            arch|manjaro|endeavouros|cachyos)
                PKG_INSTALL="sudo pacman -S --needed"
                PKG_PYTHON="python"
                PKG_FFMPEG="ffmpeg"
                PKG_VULKAN_BUILD="vulkan-icd-loader vulkan-headers shaderc cmake base-devel"
                ;;
            opensuse*|sles)
                PKG_INSTALL="sudo zypper install -y"
                PKG_PYTHON="python3"
                PKG_FFMPEG="ffmpeg"
                PKG_VULKAN_BUILD="vulkan-devel glslang-devel cmake gcc-c++"
                ;;
            *)
                PKG_INSTALL="(install via your distro's package manager)"
                PKG_PYTHON="python3 (3.10+) and the venv module"
                PKG_FFMPEG="ffmpeg"
                PKG_VULKAN_BUILD="vulkan loader+headers, glslc shader compiler, cmake, C++ toolchain"
                ;;
        esac
        ;;
    Darwin)
        DISTRO="macos"
        PKG_INSTALL="brew install"
        PKG_PYTHON="python@3.12"
        PKG_FFMPEG="ffmpeg"
        ;;
    *)
        err "Unsupported OS: $OS. localscribe targets Linux and macOS."
        err "On Windows, please use WSL2."
        exit 1
        ;;
esac

# ---- Hardware backend detection ----
detect_backend() {
    if [ -n "$BACKEND_OVERRIDE" ]; then
        echo "$BACKEND_OVERRIDE"
        return
    fi
    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then echo "mps"; else echo "cpu"; fi
        return
    fi
    # Linux
    if command -v nvidia-smi >/dev/null 2>&1; then
        local drv major
        drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
               | head -n1 | tr -d ' ')" || drv=""
        major="${drv%%.*}"
        case "$major" in
            ''|*[!0-9]*) ;;
            *)
                if [ "$major" -ge 570 ]; then echo "cu128"; return; fi
                if [ "$major" -ge 550 ]; then echo "cu124"; return; fi
                if [ "$major" -ge 525 ]; then echo "cu121"; return; fi
                warn "NVIDIA driver $drv is older than 525.x; CUDA wheels need a newer driver. Falling back to CPU."
                ;;
        esac
    fi
    # No CUDA. Try Vulkan -- whisper.cpp can be built against it for AMD/Intel GPUs.
    if command -v vulkaninfo >/dev/null 2>&1; then
        if vulkaninfo --summary 2>/dev/null | grep -qi "devicename"; then
            echo "vulkan"
            return
        fi
    fi
    if command -v rocminfo >/dev/null 2>&1; then
        warn "AMD ROCm detected but no Vulkan and no CUDA. Using CPU."
        warn "Install Vulkan (mesa-vulkan-drivers + vulkan-tools) for GPU-accelerated whisper.cpp."
    fi
    echo "cpu"
}

BACKEND="$(detect_backend)"
info "OS: $OS ($DISTRO/$ARCH)    Hardware backend: $BACKEND"

# ---- LLM backend selection ----
# Holds OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY collected from the prompt
OPENAI_BASE_URL_VAL=""
OPENAI_MODEL_VAL=""
OPENAI_API_KEY_VAL=""

select_llm_backend() {
    if [ -n "$LLM_OVERRIDE" ]; then
        echo "$LLM_OVERRIDE"
        return
    fi
    # Non-interactive (no tty, e.g. CI): default silently
    if [ ! -t 0 ]; then
        echo "claude_cli"
        return
    fi
    # In --check mode we don't want to prompt either; just default.
    if [ "$CHECK_ONLY" -eq 1 ]; then
        echo "claude_cli"
        return
    fi
    # Interactive prompt -- write to stderr so stdout stays the chosen value.
    {
        echo
        echo "Which LLM backend will you use for speaker ID + summarization?"
        echo "  1) Claude CLI    -- uses your Claude Pro/Max subscription (the \`claude\` binary)"
        echo "  2) Codex CLI     -- uses your ChatGPT Plus/Pro/Team subscription (the \`codex\` binary)"
        echo "  3) Gemini CLI    -- uses your Google Gemini Advanced subscription (the \`gemini\` binary)"
        echo "  4) OpenAI API    -- any OpenAI-compatible HTTP endpoint (OpenAI, Anthropic API,"
        echo "                     Ollama, vLLM, LM Studio, Groq, OpenRouter, ...)"
    } >&2
    local choice
    while true; do
        read -r -p "Choice [1]: " choice >&2 || choice=""
        choice="${choice:-1}"
        case "$choice" in
            1) echo "claude_cli"; return ;;
            2) echo "codex_cli";  return ;;
            3) echo "gemini_cli"; return ;;
            4) echo "openai_api"; return ;;
            *) printf 'Please enter 1-4.\n' >&2 ;;
        esac
    done
}

prompt_openai_config() {
    # Only interactive; --llm=openai_api on its own keeps env defaults.
    if [ -n "$LLM_OVERRIDE" ] || [ ! -t 0 ] || [ "$CHECK_ONLY" -eq 1 ]; then
        return
    fi
    local default_url="https://api.openai.com/v1"
    local default_model="gpt-4o"
    {
        echo
        echo "Configure the OpenAI-compatible endpoint."
        echo "Defaults work for OpenAI; override for Ollama/LM Studio/Anthropic API/etc."
    } >&2
    read -r -p "Base URL [$default_url]: " OPENAI_BASE_URL_VAL >&2 || OPENAI_BASE_URL_VAL=""
    OPENAI_BASE_URL_VAL="${OPENAI_BASE_URL_VAL:-$default_url}"
    read -r -p "Model name [$default_model]: " OPENAI_MODEL_VAL >&2 || OPENAI_MODEL_VAL=""
    OPENAI_MODEL_VAL="${OPENAI_MODEL_VAL:-$default_model}"
    read -r -p "API key (leave blank to set in .env later): " OPENAI_API_KEY_VAL >&2 || OPENAI_API_KEY_VAL=""
}

LLM_BACKEND="$(select_llm_backend)"
info "LLM backend: $LLM_BACKEND"
if [ "$LLM_BACKEND" = "openai_api" ]; then
    prompt_openai_config
fi

# ---- Prerequisite checks ----
MISSING=()

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        MISSING+=("python3 (3.10+)|$PKG_INSTALL $PKG_PYTHON")
        return
    fi
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        local v
        v="$(python3 --version 2>&1 || true)"
        MISSING+=("python3 >= 3.10 (have: $v)|$PKG_INSTALL $PKG_PYTHON")
        return
    fi
    if ! python3 -c 'import venv' 2>/dev/null; then
        MISSING+=("python3 venv module|$PKG_INSTALL $PKG_PYTHON")
    fi
}

check_ffmpeg() {
    if ! command -v ffmpeg >/dev/null 2>&1; then
        MISSING+=("ffmpeg|$PKG_INSTALL $PKG_FFMPEG")
    fi
}

# Only require the matching CLI binary for the LLM backend the user picked.
# openai_api needs no system binary -- httpx ships via pip.
check_llm_backend() {
    case "$LLM_BACKEND" in
        claude_cli)
            command -v claude >/dev/null 2>&1 || MISSING+=(
                "claude CLI|npm install -g @anthropic-ai/claude-code   (https://claude.com/claude-code)"
            )
            ;;
        codex_cli)
            command -v codex >/dev/null 2>&1 || MISSING+=(
                "codex CLI|npm install -g @openai/codex   (then run: codex login)"
            )
            ;;
        gemini_cli)
            command -v gemini >/dev/null 2>&1 || MISSING+=(
                "gemini CLI|npm install -g @google/gemini-cli   (then run: gemini auth)"
            )
            ;;
        openai_api)
            : ;;
    esac
}

# Only relevant when BACKEND=vulkan: pywhispercpp has to be built from source
# with -DGGML_VULKAN=on, which needs cmake, a C++ compiler, the Vulkan loader+
# headers, and a GLSL shader compiler.
check_vulkan_build_deps() {
    [ "$BACKEND" = "vulkan" ] || return 0
    if ! command -v cmake >/dev/null 2>&1; then
        MISSING+=("cmake (Vulkan build)|$PKG_INSTALL $PKG_VULKAN_BUILD")
    fi
    if ! command -v glslc >/dev/null 2>&1 && ! command -v glslangValidator >/dev/null 2>&1; then
        MISSING+=("glslc shader compiler (Vulkan build)|$PKG_INSTALL $PKG_VULKAN_BUILD")
    fi
}

check_python
check_ffmpeg
check_llm_backend
check_vulkan_build_deps

if [ ${#MISSING[@]} -gt 0 ]; then
    err "Missing prerequisites:"
    for entry in "${MISSING[@]}"; do
        printf '  - %s\n      %s\n' "${entry%%|*}" "${entry#*|}"
    done
    exit 1
fi
ok "All prerequisites present."

if [ "$CHECK_ONLY" -eq 1 ]; then
    info "--check passed; skipping install."
    exit 0
fi

# ---- Venv ----
if [ ! -d .venv ]; then
    info "Creating .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# ---- Install torch with the selected backend ----
case "$BACKEND" in
    cu121|cu124|cu128)
        info "Installing torch from PyTorch's $BACKEND wheel index"
        pip install --index-url "https://download.pytorch.org/whl/$BACKEND" torch torchaudio
        ;;
    cpu|vulkan)
        # Vulkan accelerates whisper.cpp, not pyannote -- pyannote stays on CPU
        # torch since there's no ROCm path wired up.
        info "Installing CPU-only torch"
        pip install --index-url "https://download.pytorch.org/whl/cpu" torch torchaudio
        ;;
    mps)
        info "Installing torch from PyPI (Apple Silicon — MPS support is in stock wheels)"
        pip install torch torchaudio
        ;;
    *)
        err "Unknown backend: $BACKEND"
        exit 1
        ;;
esac

# ---- Install localscribe + the right whisper engine ----
# Apple Silicon: pywhispercpp wheel ships with Metal support out of the box.
# Vulkan: pywhispercpp has to be source-built with CMAKE_ARGS=-DGGML_VULKAN=on
# (and the wheel cache disabled) to actually use the GPU.
case "$BACKEND" in
    mps)
        pip install -e ".[whisper-cpp]"
        ;;
    vulkan)
        pip install -e .
        info "Building pywhispercpp from source with Vulkan support (this can take a few minutes)..."
        CMAKE_ARGS="-DGGML_VULKAN=on" \
            pip install --no-binary pywhispercpp --no-cache-dir "pywhispercpp>=1.2"
        ;;
    *)
        pip install -e .
        ;;
esac

# ---- Bootstrap .env and write LLM config ----
ENV_CREATED=0
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    ENV_CREATED=1
fi

# Set or update KEY=VAL in a file. Updates the line in place if KEY is
# already there (commented-out or live); appends otherwise. Uses awk for
# portability across BSD (macOS) and GNU sed.
upsert_env() {
    local file="$1" key="$2" value="$3"
    if [ ! -f "$file" ]; then
        printf '%s=%s\n' "$key" "$value" > "$file"
        return
    fi
    if grep -q -E "^#? *${key}=" "$file"; then
        awk -v key="$key" -v val="$value" '
            BEGIN { re = "^#? *" key "=" }
            $0 ~ re { print key "=" val; next }
            { print }
        ' "$file" > "$file.new" && mv "$file.new" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
}

if [ -f .env ]; then
    upsert_env .env LOCALSCRIBE_LLM_BACKEND "$LLM_BACKEND"
    if [ "$LLM_BACKEND" = "openai_api" ]; then
        [ -n "$OPENAI_BASE_URL_VAL" ] && upsert_env .env OPENAI_BASE_URL "$OPENAI_BASE_URL_VAL"
        [ -n "$OPENAI_MODEL_VAL" ]    && upsert_env .env OPENAI_MODEL    "$OPENAI_MODEL_VAL"
        [ -n "$OPENAI_API_KEY_VAL" ]  && upsert_env .env OPENAI_API_KEY  "$OPENAI_API_KEY_VAL"
    fi
fi

# ---- Summary ----
echo
ok "Setup complete."
echo
echo "Next steps:"
echo "  1. Accept terms at https://hf.co/pyannote/speaker-diarization-3.1"
echo "  2. Accept terms at https://hf.co/pyannote/segmentation-3.0"
if [ "$ENV_CREATED" -eq 1 ]; then
    echo "  3. Edit .env (just created) and set HF_TOKEN=hf_..."
else
    echo "  3. Set HF_TOKEN in .env (or export HF_TOKEN=hf_...)"
fi
if [ "$LLM_BACKEND" = "openai_api" ] && [ -z "$OPENAI_API_KEY_VAL" ]; then
    echo "  4. Set OPENAI_API_KEY in .env (or leave blank for local Ollama / LM Studio)"
    echo "  5. source .venv/bin/activate"
    echo "  6. localscribe <youtube-url>"
else
    echo "  4. source .venv/bin/activate"
    echo "  5. localscribe <youtube-url>"
fi

case "$BACKEND" in
    cpu)
        echo
        warn "Running on CPU (no GPU detected). The default large-v3 model will be slow."
        warn "Consider --model small or --model medium for testing."
        ;;
    mps)
        echo
        info "Apple Silicon: transcription will use whisper.cpp + Metal (via pywhispercpp)."
        info "Diarization will use pyannote on MPS. Both will be GPU-accelerated."
        ;;
    vulkan)
        echo
        info "Linux+Vulkan: transcription will use whisper.cpp + Vulkan (via pywhispercpp)."
        info "Diarization will run on CPU (no ROCm/Vulkan path for pyannote)."
        ;;
esac
