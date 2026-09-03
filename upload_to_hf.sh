#!/bin/bash
# upload_to_hf.sh
# Script to push local datasets to the Hugging Face Hub using the `hf` CLI.

HF_REPO=${1:-"T40/edge-aui-framework-data"}
DATA_DIR=${2:-".data/"}

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "Hugging Face CLI ('hf') is not installed."
    echo "Install it via: brew install hf"
    exit 1
fi

echo "Authenticating Hugging Face CLI..."
hf auth login

echo "Uploading dataset from ${DATA_DIR} to Hugging Face Hub repository: ${HF_REPO}..."
# Push the dataset files to the Hugging Face repository
hf upload ${HF_REPO} ${DATA_DIR} --repo-type=dataset

echo ""
echo "Upload complete! The dataset can now be streamed programmatically in your Python pipeline using:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('${HF_REPO}')"
