#!/bin/bash
# One-time setup of the Strata model for LocalAI (run as root on the LocalAI LXC).
#
# Downloads Strata, runs its own setup for Qwen3.8-Flash-Next IQ2_XS (text only,
# 64K context, CUDA), and writes the LocalAI model config strata.yaml. Safe to run
# again: Strata's setup resumes an interrupted download and keeps what is done.
#
# Settings can be changed through the environment, e.g.:
#   STRATA_MODEL=IQ3_S STRATA_CONTEXT=32768 bash strata-setup.sh
set -euo pipefail

MODELS_DIR="${LOCALAI_MODELS_PATH:-/opt/localai_data/models}"
DEST="${STRATA_DEST:-$MODELS_DIR/strata}"
FAMILY="${STRATA_FAMILY:-qwen}"
MODEL="${STRATA_MODEL:-IQ2_XS}"
CONTEXT="${STRATA_CONTEXT:-65536}"
RESERVE_MIB="${STRATA_VRAM_RESERVE_MIB:-1500}"
NEED_GB="${STRATA_NEED_GB:-85}"
REPO="${STRATA_REPO:-https://github.com/Niko1221/Strata.git}"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run this as root (it installs git/python3-venv if they are missing)"
[ -d "$MODELS_DIR" ] || die "$MODELS_DIR does not exist; set LOCALAI_MODELS_PATH"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: the NVIDIA driver is not visible in this container"

say "GPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

say "Checking tools"
missing=()
command -v git >/dev/null || missing+=(git)
python3 -c 'import venv, ensurepip' 2>/dev/null || missing+=(python3-venv)
if [ ${#missing[@]} -gt 0 ]; then
    command -v apt-get >/dev/null || die "please install: ${missing[*]}"
    apt-get update
    apt-get install -y "${missing[@]}"
fi
echo "git and python3-venv OK"

mkdir -p "$DEST"
avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
say "Disk: ${avail_gb} GB free in $DEST"
if [ -d "$DEST/Strata-data" ]; then
    echo "an earlier download exists; setup continues it"
elif [ "$avail_gb" -lt "$NEED_GB" ]; then
    die "Strata needs about ${NEED_GB} GB free (the download is ~70 GB); only ${avail_gb} GB free"
fi

say "Getting Strata"
if [ -d "$DEST/Strata/.git" ]; then
    git -C "$DEST/Strata" pull --ff-only
else
    git clone --depth 1 "$REPO" "$DEST/Strata"
fi
echo "Strata at commit $(git -C "$DEST/Strata" rev-parse --short HEAD)"

say "Running Strata's setup ($FAMILY $MODEL, context $CONTEXT, VRAM reserve $RESERVE_MIB MiB)"
echo "This downloads ~70 GB and can take an hour or more. If it stops, run this script again."
cd "$DEST/Strata"
./setup.sh --yes --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" \
    --vision no --backend cuda --vram-reserve-mib "$RESERVE_MIB" \
    --no-start --no-browser

say "Finding the run config"
configs=()
while IFS= read -r f; do configs+=("$f"); done < <(
    find "$DEST/Strata" -maxdepth 1 -name 'strata-*.json' \
        ! -name '*.shared-settings.json' | sort)
[ ${#configs[@]} -gt 0 ] || die "setup finished but wrote no strata-*.json in $DEST/Strata"
tag=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
cfg=""
for f in "${configs[@]}"; do
    case "$(basename "$f")" in *"$tag"*) cfg="$f" ;; esac
done
[ -n "$cfg" ] || cfg="${configs[0]}"
rel="${cfg#"$MODELS_DIR"/}"
echo "run config: $cfg"

say "Writing the LocalAI model config"
yaml="$MODELS_DIR/strata.yaml"
if [ -e "$yaml" ]; then
    yaml="$MODELS_DIR/strata.yaml.new"
    echo "strata.yaml already exists: writing $yaml to compare instead"
fi
cat > "$yaml" <<EOF
name: strata
backend: strata
parameters:
  model: $rel
context_size: $CONTEXT
# Only one of Ridge and Strata fits on the 3090: loading one evicts the other.
concurrency_groups:
  - main-llm
known_usecases:
  - chat
# Strata applies its own chat template and parses tool calls and reasoning.
template:
  use_tokenizer_template: true
options:
  # VRAM left free beside Strata's expert cache. The cache is sized from the VRAM
  # that is free when Strata starts, so Jarvis (pinned, already loaded) is not
  # counted here: this is only headroom for Jarvis's spikes while it speaks.
  - vram_reserve_mib:$RESERVE_MIB
  # The first start reads 35-55 GB into RAM; allow it time.
  - load_timeout:1800
EOF
echo "wrote $yaml"

say "Done"
cat <<EOF
Next:
  1. Install the strata backend in LocalAI if you have not yet
     (Backends -> From URL: ghcr.io/vyssecodes/localai-strata-backend:latest, name strata).
  2. Restart LocalAI so it reads strata.yaml.
  3. Try it: pin your failover chain to strata, or call the model "strata" directly.
EOF
