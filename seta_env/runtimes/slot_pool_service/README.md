# Slot Pool Service

Two components:
- **Scheduler** — runs on the training machine; tracks slots across all nodes and allocates groups for GRPO rollouts.
- **Node manager** — runs on each remote node; wraps Docker Compose operations over HTTP.

## Cleanup (reset all nodes between eval runs)

With the scheduler running, one command stops and removes all containers across every node:

```bash
curl -s -X POST http://127.0.0.1:8000/cleanup | python3 -m json.tool
```

Output:
```json
{
  "results": [
    {"node": "http://135.181.63.176:8001", "status": 200, "body": {"stopped_containers": 4, "errors": []}},
    {"node": "http://31.22.104.161:8001",  "status": 200, "body": {"stopped_containers": 2, "errors": []}}
  ],
  "success": true,
  "failed_nodes": []
}
```

This fans out `POST /cleanup` to all nodes in parallel — stops all containers, prunes Docker networks, and clears session state on each node.

---

## Quick Start (Example: 2 nodes, 16 slots each)

### Step 1 — Edit `nodes.yaml`

Open `nodes.yaml` and set your two nodes. Replace the example IPs and SSH key path with yours:

```yaml
nodes:
  - url: "http://135.181.63.176:8001"
    slots: 16
    deploy:
      ssh_key: ~/.ssh/id_ed25519
      ssh_user: root
      api_key: harbor-node-dev-key

  - url: "http://31.22.104.161:8001"
    slots: 16
    deploy:
      ssh_key: ~/.ssh/id_ed25519
      ssh_user: root
      api_key: harbor-node-dev-key
```

- `url` — the node's public IP, port `8001` is the node manager's HTTP port
- `slots` — max concurrent Docker containers on that node (set to 16)
- `ssh_key` — your local SSH private key that can access both nodes
- `api_key` — a shared secret used to authenticate requests to node managers

### Step 2 — Run start.sh

From the repo root (not from inside the `slot_pool_service` folder):

```bash
cd <REPO_ROOT>

# Deploy nodes + start scheduler + activate dataset in one command
bash seta_env/runtimes/slot_pool_service/start.sh --dataset seta-env-harbor
```

Or without a dataset (you can activate one later with `setup_dataset.sh`):

```bash
bash seta_env/runtimes/slot_pool_service/start.sh
```

This will, for each node in `nodes.yaml`:
1. Install system packages (`python3`, `git`, `curl`)
2. Configure the Docker daemon address pool (`10.200.0.0/16 /28`) and **restart Docker** — prevents "address pool exhausted" errors when many containers run in parallel
3. Create a Python virtualenv and install the node manager dependencies
4. Copy `node_manager.py`, `datasets.yaml` (from `seta_env/dataset/`), and compose files to the node
5. Install the node manager as a systemd service on port `8001`
6. **Open port `8001` in the firewall** (`ufw` / `firewalld` / `iptables`, whichever is present)
7. Start the node manager service

Then starts the scheduler locally on `http://127.0.0.1:8000`.

The terminal will block — the scheduler runs until you press `Ctrl-C`.

> **Note:** Docker is restarted on each node during deploy (step 2 above). Any containers already running on those nodes will be stopped.

### Step 3 — Verify everything is running

In a second terminal, check the scheduler:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

Check each node manager is alive:

```bash
curl http://135.181.63.176:8001/health
curl http://31.22.104.161:8001/health
```

You should see `"status":"ok"` in each response.

---

### Step 4 (optional) — Switch dataset while running

If you didn't pass `--dataset` at startup, or want to switch to a different dataset after the scheduler is already running, use `setup_dataset.sh` in a second terminal:

```bash
# Activate a dataset across all nodes (scheduler must be running)
bash seta_env/runtimes/slot_pool_service/setup_dataset.sh seta-env-harbor

# Switch to a different dataset
bash seta_env/runtimes/slot_pool_service/setup_dataset.sh terminal-bench-2.0
```

Output:
```
=== Setting up dataset: seta-env-harbor ===
    Scheduler: http://127.0.0.1:8000

  [OK] http://135.181.63.176:8001 (downloaded)
  [OK] http://31.22.104.161:8001 (downloaded)

All nodes ready.
```

Each node git-clones the dataset on first use and caches it — switching back to an already-downloaded dataset is instant.

**Available datasets** are defined in `seta_env/dataset/datasets.yaml` (copied to every node during deploy):

```yaml
datasets:
  seta-env-harbor:
    repo: "https://github.com/Michaelsqj/tbench_data_converted.git"
    subfolder: "seta-env-harbor"
  terminal-bench-2.0:
    repo: "https://github.com/Michaelsqj/tbench_data_converted.git"
    subfolder: "terminal-bench-2.0"
  terminal-bench-core_migrated:
    repo: "https://github.com/Michaelsqj/tbench_data_converted.git"
    subfolder: "terminal-bench-core_migrated"
```

To add a new dataset, append an entry then re-run `start.sh` to push the updated file to the nodes:

```yaml
  my-dataset:
    repo: "https://github.com/yourorg/your-data-repo.git"
    subfolder: "the-subfolder-to-use"   # omit if you want the whole repo
```

---

### Re-deploying vs restarting

| Situation | Command |
|---|---|
| First time, or nodes need reinstall | `bash start.sh` |
| First time + activate dataset immediately | `bash start.sh --dataset seta-env-harbor` |
| Nodes already running, just restart scheduler | `bash start.sh --skip-deploy` |
| Nodes already running + activate dataset | `bash start.sh --skip-deploy --dataset seta-env-harbor` |

---

## One-click startup

```bash
cd seta_env/runtimes/slot_pool_service

# Deploy node managers + start scheduler
bash start.sh

# Skip deploy (node managers already running)
bash start.sh --skip-deploy

# Custom scheduler port or bind address
bash start.sh --skip-deploy --port 9000 --host 0.0.0.0
```

The scheduler starts on `http://127.0.0.1:8000` by default and blocks until Ctrl-C.

## nodes.yaml

Defines the remote nodes. Add a `deploy:` block to any node to have `start.sh` auto-deploy its node manager via SSH. Omit it to skip deployment for that node.

```yaml
nodes:
  - url: "http://<host>:<port>"
    slots: 64                         # max concurrent Docker containers
    deploy:                           # remove this block to skip auto-deploy
      ssh_key: /path/to/id_ed25519
      ssh_user: root
      api_key: harbor-node-dev-key        # default; override with NODE_MANAGER_API_KEY
```

## Manual node deploy

To deploy a single node without `start.sh`:

```bash
bash deploy_node.sh <host> <ssh_key> [--user root] [--port 8001] [--api-key KEY]
```

## Scheduler endpoints

| Method | Path              | Description                                          |
|--------|-------------------|------------------------------------------------------|
| GET    | `/health`         | Liveness check                                       |
| GET    | `/status`         | Slot counts and active groups                        |
| POST   | `/allocate_group` | `{"task_id": "...", "n_slots": N}`                   |
| POST   | `/release_group`  | `{"task_id": "..."}`                                 |
| POST   | `/setup_dataset`  | `{"dataset_name": "..."}` — fan out to all nodes     |
| POST   | `/cleanup`        | Stop all containers on every node (hard reset)       |

## Node manager endpoints

| Method | Path        | Description                                              |
|--------|-------------|----------------------------------------------------------|
| GET    | `/health`   | Node status and active dataset                           |
| POST   | `/setup`    | `{"dataset_name": "..."}` — activate dataset on node    |
| POST   | `/build`    | `{"task_name": "..."}` — build Docker image              |
| POST   | `/cleanup`  | Stop + remove ALL containers, prune networks (hard reset)|
| POST   | `/gc`       | Kill orphaned/expired containers only                    |

To reset a node between eval runs:

```bash
curl -s -X POST http://<node>:8001/cleanup -H "X-API-Key: harbor-node-dev-key"
```

## GRPORollout config

Point `GRPORollout` at the scheduler via `env_config`:

```python
env_config = {
    "environment_type": "remote_docker",
    "scheduler_url": "http://127.0.0.1:8000",
    "node_api_key": "harbor-node-dev-key",  # must match what was deployed
    ...
}
```
