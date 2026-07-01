# FRP Tunnel Usage

## Setup (one-time)

```bash
# Deploy relay to CPU Server 1
python manage_tunnel.py deploy-relay
```

## Per training run

```bash
# Start tunnel (ranks discovered from sglang at runtime)
python manage_tunnel.py start gpu-a --ranks "172.18.0.2:31051,172.18.0.2:31052"

# Check status
python manage_tunnel.py status

# Stop
python manage_tunnel.py stop gpu-a
```

## Agents connect via relay

```python
base_url = f"http://65.108.32.175:{39001 + rank}/v1/{SESSION_ID}"
```

## Adding a new GPU machine

Edit `tunnel_config.yaml`, pick a non-overlapping `base_remote_port`:

```yaml
gpu_machines:
  - name: gpu-a
    base_remote_port: 39001
    num_ranks: 4
  - name: gpu-b              # new
    ssh_host: "10.0.2.50"
    base_remote_port: 39101   # no overlap
    num_ranks: 8
```

```bash
python manage_tunnel.py validate
python manage_tunnel.py start gpu-b --ranks "..."
```

## Testing

```bash
# Mock test (no sglang needed)
PYTHON=/path/to/python bash deploy_and_test.sh --num-mock-ranks 2 \
  --test-from "135.181.71.8" --cleanup

# Sglang test
python test_tunnel.py --base-url http://65.108.32.175:39001/v1 --concurrency 64
```
