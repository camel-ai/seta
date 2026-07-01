# Remote Execution Environment Usage

This guide explains how to set up and use the remote execution environment for the Terminal Agent. This allows running tasks in sandboxed Docker containers on a fleet of remote EC2 instances.

## Prerequisites

1.  **AWS Credentials**: Ensure you have AWS credentials configured (e.g., `~/.aws/credentials`) or exported as environment variables.
2.  **SSH Key Pair**: You need an SSH key pair (e.g., `~/.ssh/id_rsa`) to access EC2 instances.
3.  **Python Dependencies**: Install local dependencies:
    ```bash
    pip install boto3 paramiko requests
    ```

## 1. Setting Up the Remote Infrastructure

You can launch the EC2 fleet using the provided variable `launch_instances.py` script (if available) or manually via AWS Console using the User Data script `src/remote_env/setup_remote.sh`.

### Automatic Launch
If you have a launch script:
```bash
python scripts/launch_instances.py --count 1
```

### Manual Launch
1.  Launch an Ubuntu 22.04 instance.
2.  In "Advanced Details" -> "User Data", paste the content of `src/remote_env/setup_remote.sh`.
3.  Wait for initialization (installing Docker, Python, etc.).

## 2. Deploying Agent Code

Since the agent logic runs partially on the remote server (the `sidecar.py` and `provisioner.py`), you need to copy the necessary files to the remote instance(s).

Assume your remote instance IP is `1.2.3.4` and user is `ubuntu`.

1.  **Create Code Directory**
    ```bash
    ssh ubuntu@1.2.3.4 "mkdir -p /home/ubuntu/agent_code/remote"
    ```

2.  **Copy Files**
    You need to copy `terminal_toolkit.py` and the `remote/` directory.
    ```bash
    # From project root
    scp src/remote_env/terminal_toolkit.py ubuntu@1.2.3.4:/home/ubuntu/agent_code/
    scp src/remote_env/remote/*.py ubuntu@1.2.3.4:/home/ubuntu/agent_code/remote/
    ```

3.  **Start the Provisioner**
    The provisioner manages the lifecycle of task containers.
    ```bash
    ssh ubuntu@1.2.3.4 "nohup python3 /home/ubuntu/agent_code/remote/provisioner.py > /home/ubuntu/provisioner.log 2>&1 &"
    ```
    *Note: The provisioner listens on port 30000.*

## 3. Preparing the Dataset

The remote provisioner expects tasks to be located in a dataset root directory, defined by the `DATASET_ROOT` environment variable (default: `/home/ubuntu/dataset`).

1.  **Clone/Download Dataset**
    ssh into the remote server and prepare the tasks.
    ```bash
    ssh ubuntu@1.2.3.4
    mkdir -p /home/ubuntu/dataset
    cd /home/ubuntu/dataset
    git clone https://github.com/your-org/tbench-tasks.git .
    ```
    Ensure your task folders (e.g., `3d-model-format-legacy`) are inside `/home/ubuntu/dataset`.

## 4. Running Tasks Remotely

Use the `RemoteManager` in your local Python code to execute tasks.

```python
from src.remote_env.remote_manager import RemoteManager

# 1. Initialize Manager (loads server list from registry or you can manually configure)
manager = RemoteManager() 
# Ensure server_registry.json is populated or manager serves known IPs.

# 2. Get a Toolkit for a specific task
# This tells the remote provisioner to look for '3d-model-format-legacy' in DATASET_ROOT
# and launch it using docker-compose or Dockerfile found there.
toolkit = manager.get_remote_toolkit(task_id="3d-model-format-legacy")

# 3. Use the Toolkit as if it were local
# All commands run inside the remote task container.
output = toolkit.shell_exec("ls -la")
print(output)

# 4. Clean up
manager.stop_toolkit(toolkit)
```

## 5. Troubleshooting

*   **Logs**: Check `/home/ubuntu/provisioner.log` on the remote server for startup errors.
*   **Safety**: The default security group should allow traffic on ports 30000-32000 from your IP.
*   **Docker**: Ensure the `ubuntu` user has docker permissions (handled by `setup_remote.sh`).
