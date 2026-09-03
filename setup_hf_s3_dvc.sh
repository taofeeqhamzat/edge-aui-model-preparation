#!/bin/bash
# setup_hf_s3_dvc.sh
# Configures Hugging Face S3-compatible gateway as the DVC remote backend for automated MLOps.

set -e

if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
fi

HF_ORG=${1:-"T40"}
HF_DATASET=${2:-"edge-aui-framework-data"}
DVC_STORE_PATH=${3:-"dvc-store"}

echo "Configuring DVC remote 'hfremote' via Hugging Face S3 Gateway..."
dvc remote add -d -f hfremote s3://${HF_DATASET}/${DVC_STORE_PATH}
dvc remote modify hfremote endpointurl https://s3.hf.co/${HF_ORG}
dvc remote modify hfremote region us-east-1
dvc remote modify hfremote jobs 2


echo ""
echo "=== DVC Remote Successfully Configured ==="
echo "Endpoint:    https://s3.hf.co/${HF_ORG}"
echo "Bucket Path: s3://${HF_DATASET}/${DVC_STORE_PATH}"
echo ""
echo "To authenticate for push/pull:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Click on your token -> 'Generate S3 credentials'"
echo "3. Run locally:"
echo "   dvc remote modify --local hfremote access_key_id '<YOUR_ACCESS_KEY>'"
echo "   dvc remote modify --local hfremote secret_access_key '<YOUR_SECRET_KEY>'"
echo "4. Push your tracked data:"
echo "   dvc push"
