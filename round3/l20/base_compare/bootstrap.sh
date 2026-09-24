#!/usr/bin/env bash
set -euo pipefail
apt-get update > /work/output/compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/compiler_install.log 2>&1
python -m venv --system-site-packages /work/modern
/work/modern/bin/python -m pip install --no-cache-dir 'transformers==5.17.0' 'flash-linear-attention==0.5.2' 'safetensors==0.8.0' 'accelerate==1.13.0' 'scikit-learn==1.7.2' > /work/output/install_modern.log 2>&1
python -m venv --system-site-packages /work/legacy
if test -f /work/wheels/SHA256SUMS; then
  (cd /work/wheels && sha256sum -c SHA256SUMS)
  /work/legacy/bin/python -m pip install --no-index --find-links /work/wheels 'transformers==4.55.0' 'tokenizers==0.21.4' 'safetensors==0.8.0' 'scikit-learn==1.7.2' > /work/output/install_legacy_offline.log 2>&1
else
  /work/legacy/bin/python -m pip install --no-cache-dir 'transformers==4.55.0' 'tokenizers==0.21.4' 'safetensors==0.8.0' 'scikit-learn==1.7.2' > /work/output/install_legacy.log 2>&1
fi
/work/modern/bin/python /work/input/fetch_models.py > /work/output/download_models.log 2>&1
/work/modern/bin/python -c 'import torch, transformers, fla; from transformers import Qwen3_5ForConditionalGeneration; assert "L20" in torch.cuda.get_device_name(0)'
touch /work/output/environment_ready
