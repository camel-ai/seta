# Slot Pool Service

Distributes Docker environments across remote nodes. Config lives in `seta_env/runtimes/slot_pool_service/`.

### 1. Edit `seta_env/runtimes/slot_pool_service/nodes.yaml`

```yaml
nodes:
  - url: "http://<NODE_IP>:8001"
    slots: 16
    deploy:
      ssh_key: /path/to/ssh/key
      ssh_user: root
      api_key: your-api-key
```

### 2. Deploy + start + activate dataset

```bash
bash seta_env/runtimes/slot_pool_service/start.sh --dataset seta-env-v2
```

### 3. Run eval with remote_docker

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    runtime.env_type=remote_docker \
    runtime.scheduler_url=http://127.0.0.1:8000
```
