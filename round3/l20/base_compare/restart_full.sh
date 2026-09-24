#!/usr/bin/env bash
set -euo pipefail
apt-get update > /work/output/compiler_install_r2.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/compiler_install_r2.log 2>&1
/work/modern/bin/python -c 'import torch, transformers, fla; from transformers import Qwen3_5ForConditionalGeneration; print(dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.cuda.get_device_name(0)))'
exec bash /work/input/start.sh
