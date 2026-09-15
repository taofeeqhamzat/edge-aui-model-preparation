#!/bin/bash
# upload_to_hf.sh
# Script to push local datasets to the Hugging Face Hub using the `hf` CLI.
# Supports custom commit messages and descriptions via flags, positional arguments, or environment variables.

set -e

# Default settings
DEFAULT_REPO="T40/edge-aui-framework-data"
DEFAULT_DATA_DIR=".data/"

HF_REPO=""
DATA_DIR=""
COMMIT_MSG="${COMMIT_MESSAGE:-}"
COMMIT_DESC="${COMMIT_DESCRIPTION:-}"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [HF_REPO] [DATA_DIR] [COMMIT_MESSAGE] [COMMIT_DESCRIPTION]

Upload local dataset directory to Hugging Face Hub using the 'hf' CLI.

Options:
  -m, --message, --commit-message TEXT       Commit summary / message
  -d, --description, --commit-description TEXT Commit description
  -r, --repo REPO_ID                         Hugging Face repository ID (default: ${DEFAULT_REPO})
  -p, --dir, --data-dir PATH                 Local directory path (default: ${DEFAULT_DATA_DIR})
  -h, --help                                 Display this help message and exit

Positional Arguments (alternative to options):
  1. HF_REPO           Repository ID (or single commit message if no slash '/' in argument)
  2. DATA_DIR          Path to dataset directory (default: ${DEFAULT_DATA_DIR})
  3. COMMIT_MESSAGE    Commit message
  4. COMMIT_DESCRIPTION Commit description

Environment Variables:
  COMMIT_MESSAGE       Default commit message if not specified on CLI
  COMMIT_DESCRIPTION   Default commit description if not specified on CLI

Examples:
  $(basename "$0") -m "feat(data): update AdSERP microtensors with 7-class taxonomy"
  $(basename "$0") "T40/edge-aui-framework-data" ".data/" "Update dataset"
  $(basename "$0") "feat(data): quick update"
EOF
}

# Parse options and positional arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message|--commit-message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        -d|--description|--commit-description)
            COMMIT_DESC="$2"
            shift 2
            ;;
        -r|--repo)
            HF_REPO="$2"
            shift 2
            ;;
        -p|--dir|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -eq 1 ]]; then
    if [[ "${POSITIONAL_ARGS[0]}" == *"/"* ]]; then
        HF_REPO="${POSITIONAL_ARGS[0]}"
    else
        # Single non-repo argument treated as commit message
        if [[ -z "$COMMIT_MSG" ]]; then
            COMMIT_MSG="${POSITIONAL_ARGS[0]}"
        fi
    fi
elif [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    HF_REPO="${POSITIONAL_ARGS[0]}"
    DATA_DIR="${POSITIONAL_ARGS[1]}"
    if [[ ${#POSITIONAL_ARGS[@]} -ge 3 && -z "$COMMIT_MSG" ]]; then
        COMMIT_MSG="${POSITIONAL_ARGS[2]}"
    fi
    if [[ ${#POSITIONAL_ARGS[@]} -ge 4 && -z "$COMMIT_DESC" ]]; then
        COMMIT_DESC="${POSITIONAL_ARGS[3]}"
    fi
fi

# Apply defaults
HF_REPO="${HF_REPO:-$DEFAULT_REPO}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"

# Validate prerequisites
if ! command -v hf &> /dev/null; then
    echo "Error: Hugging Face CLI ('hf') is not installed."
    echo "Install it via: brew install hf"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# Authentication check
if ! hf auth whoami &> /dev/null; then
    echo "Authenticating Hugging Face CLI..."
    hf auth login
else
    echo "Hugging Face CLI authenticated as: $(hf auth whoami)"
fi

echo "Uploading dataset from '${DATA_DIR}' to Hugging Face Hub repository: '${HF_REPO}'..."
if [ -n "$COMMIT_MSG" ]; then
    echo "Commit Message: $COMMIT_MSG"
fi
if [ -n "$COMMIT_DESC" ]; then
    echo "Commit Description: $COMMIT_DESC"
fi

UPLOAD_ARGS=(
    "${HF_REPO}"
    "${DATA_DIR}"
    --repo-type=dataset
    --exclude "**/.git/**"
    --exclude "**/.DS_Store"
)

if [ -n "$COMMIT_MSG" ]; then
    UPLOAD_ARGS+=(--commit-message "${COMMIT_MSG}")
fi

if [ -n "$COMMIT_DESC" ]; then
    UPLOAD_ARGS+=(--commit-description "${COMMIT_DESC}")
fi

hf upload "${UPLOAD_ARGS[@]}"

echo ""
echo "Upload complete! The dataset can now be streamed programmatically in your Python pipeline using:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('${HF_REPO}')"
