# The Harbor compatible runtime

- Daytona
- Docker
- Docker remote
- Modal
- K8S
- slot_pool

## Configuration

### Daytona

    - DAYTONA_API_KEY
    - DAYTONA_API_URL

### Docker remote

    - DOCKER_HOST
        Add DOCKER_HOST to DockerEnvironmentEnvVars explicitly


@terminal_agent/seta_env/runtimes/docker_harbor_runtime.py 
@terminal_agent/external/harbor/src/harbor/environments/docker/docker.py 

I want to host the docker container on remote servers. 

1. write a scheduler service script, which runs on one node use fast api service. it will be started with knowledge of a config file, with list of node url and spots. it always schedule spots in group base for grpo convenience in a balanced mode, the scheduling is running locked to prevent competing condition. and a node manager to start on each server which 

2. a runtime harbor runtime  


help me modify the current runtime and harbor docker environment, which can accept remote url, and add a runtime build function can run separately. 

so that i can async run build when a task comes, to build the docker image on remote, then run the containers concurrently on remote servers 

experiment with 95.133.253.67

add a orchestrator script under <REPO_ROOT>/seta_env/environments, which can 