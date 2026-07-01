# 1. install miniforge
SCRIPT_PATH=$(realpath "$0")
PROJECT_DIR=$(dirname "$SCRIPT_PATH")
#!/bin/bash
cd $PROJECT_DIR/../
apt-get update && apt-get install -y libnuma1
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# run the installer without interactive mode, all yes to prompts
bash Miniforge3-$(uname)-$(uname -m).sh -b
# initialize conda
source ~/miniforge3/bin/activate
conda init
conda create -n terminal_agent python=3.12 -y
conda activate terminal_agent
pip install uv
# 2. install dependencies
# if nvcc is not found, install cuda toolkit
if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found, installing CUDA toolkit..."
    mamba install nvidia::cuda-toolkit -y
else
    echo "nvcc found, skipping CUDA toolkit installation."
fi
conda install -c conda-forge git-lfs -y
cd ${PROJECT_DIR}/external/camel && uv pip install -e .
cd ${PROJECT_DIR}/external/harbor && uv pip install -e .
cd ${PROJECT_DIR}/external/areal && uv pip install -e .[all]

pip install --no-cache --no-build-isolation flash-attn==2.8.3
pip install --no-cache --no-build-isolation transformers==4.57.1
pip install --no-cache --no-build-isolation datasets==4.5.0
pip install --no-cache --no-build-isolation "numpy<2.3,>=2.0"

cd ${PROJECT_DIR} && uv pip install -e .

# 3. install docker if not found
if ! command -v docker &> /dev/null
then
    echo "Docker not found, installing..."
    cd ${PROJECT_DIR}/../
    # install docker using the convenience script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh ./get-docker.sh
    # add current user to docker group
    sudo usermod -aG docker $USER
    newgrp docker
else
    echo "Docker found, skipping installation."
fi


# 4. modify docker to increase network address pool

DOCKER_DAEMON_CONFIG='/etc/docker/daemon.json'

# Backup existing daemon.json if it exists
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    echo "Backing up existing Docker daemon configuration..."
    sudo cp "$DOCKER_DAEMON_CONFIG" "${DOCKER_DAEMON_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Create or update daemon.json with network pool settings
echo "Configuring Docker daemon..."
sudo tee "$DOCKER_DAEMON_CONFIG" > /dev/null <<EOF
{
  "default-address-pools": [
    {
      "base": "10.200.0.0/16",
      "size": 28
    }
  ]
}
EOF

# Restart Docker to apply changes
echo "Restarting Docker daemon..."
sudo systemctl restart docker

echo "Docker configuration complete!"
