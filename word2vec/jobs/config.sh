# Sourced by every job script. Edit W2V_STORAGE for your account.
export W2V_STORAGE="${W2V_STORAGE:-/storage/brno12-cerit/home/$USER/w2v}"
export W2V_DATA="$W2V_STORAGE/data"
export W2V_RUNS="${W2V_RUNS:-$W2V_STORAGE/runs}"  # override for a separate grid, e.g. longer training
export UV_CACHE_DIR="$W2V_STORAGE/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$W2V_STORAGE/.uv-python"
export UV_PROJECT_ENVIRONMENT="$W2V_STORAGE/.venv"
export UV_LINK_MODE=copy
export HF_HOME="$W2V_STORAGE/.hf"
export PATH="$W2V_STORAGE/.local/bin:$PATH"

# Preprocessing array size per language (one PBS subjob per shard).
declare -A W2V_SHARDS=([en]=16 [cs]=16 [uk]=32)
# Parquet files downloaded per language (evenly spaced; 0 = all). English is capped to max(cs, uk) tokens
# (~340M), which 8 of its 41 files (~73M tokens each) cover with margin.
declare -A W2V_MAX_FILES=([en]=8 [cs]=0 [uk]=0)
# <pad> tokens between sentences in tokens.bin; must be >= the largest training window.
export W2V_PAD=20

# Run the project Python from the synced environment (jobs need no uv on PATH).
w2v_py() { "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"; }
