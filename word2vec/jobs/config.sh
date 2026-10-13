# Sourced by every job script. Edit W2V_STORAGE for your account.
export W2V_STORAGE="${W2V_STORAGE:-/storage/brno12-cerit/home/$USER/w2v}"
export W2V_DATA="$W2V_STORAGE/data"
export W2V_RUNS="$W2V_STORAGE/runs"
export UV_CACHE_DIR="$W2V_STORAGE/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$W2V_STORAGE/.uv-python"
export UV_PROJECT_ENVIRONMENT="$W2V_STORAGE/.venv"
export UV_LINK_MODE=copy
export HF_HOME="$W2V_STORAGE/.hf"
export PATH="$W2V_STORAGE/.local/bin:$PATH"

# Preprocessing array size per language (one PBS subjob per shard).
declare -A W2V_SHARDS=([en]=64 [cs]=16 [uk]=32)

# Run project Python without re-syncing the shared environment.
w2v_py() { uv run --frozen --no-sync python "$@"; }
