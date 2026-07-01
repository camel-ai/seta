SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

source /data/qijia/conda/envs/terminal_agent/bin/activate /data/qijia/conda/envs/terminal_agent

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# bash /data1/qijia/terminal_agent/seta_env/runtimes/slot_pool_service/start.sh --daemon --dataset seta-env-v2

curl -s -X POST http://localhost:8000/cleanup

# bash "$REPO_ROOT/seta_env/runtimes/slot_pool_service/setup_dataset.sh" seta-env-v2

# exit 0
python -m areal.launcher.local rl_train.py \
        --config configs/config_train_remote_seta_v2.yaml \
        &> rl_train.log


