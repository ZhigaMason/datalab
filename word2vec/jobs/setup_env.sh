#!/bin/bash
# One-time setup on a MetaCentrum frontend: installs uv and the locked environment on storage.
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
mkdir -p "$W2V_STORAGE/.local/bin" "$W2V_DATA" "$W2V_RUNS"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W2V_STORAGE/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv sync --frozen --extra prep
w2v_py -c "import torch, onnxruntime, ufal.udpipe; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
