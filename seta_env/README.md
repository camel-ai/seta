# Terminal Env



## Test dataset

    - dataset/seta-env-harbor
    - dataset/terminal-bench-core_migrated
    - dataset/terminal-bench-core-0.1.1_migrated
    - dataset/terminal-bench-2.0


## Development plans
prepare a dataset, seta-env!

1. Runtimes part. 

    1.1 adapt to harbor format runtimes, with daytona, modal, docker tested

    1.2 implement EKS k8s and slotpool for cheaper alternative, harbor fork

    1.3 fault tolerant, spot instance recovery on k8s

2. Terminal toolkit adaptation to runtimes, tested, daytona, modal, docker first

3. Agent config following `camel/camel/services/agent_openapi_server.py`, integrating 

    runtimes, tool_names, prompt, model_backend (session affinity and export)

4. Harbor evaluation integration

4. model_backend for areal + miles

5. TerminalEnv, integrateing env reset, runtime reset, agent step, evaluation, reward calculation

6. task manager 

    6.1 local load tasks, record results, redis cloud service + local storage.

    6.2 redis + mongodb load tasks and record results

7. trajectory filter converter from record results -> sft and mid-training.

7. env service: follow client `camel/camel/services/agent_openapi_server.py`


## Experiments can be done with different phase, decides implementaiton priority

0. migrate terminal-bench 1.0 full and core (the official tb1.0 test set)
```bash
harbor tasks migrate -i terminal-bench-core==0.1.1 -o terminal-bench-core==0.1.1_migrated
```

1. with environment + modal backend + harbor daytona runtimes implemented, we could run async miles rollout already.

2. with local task manager implemented, we could run curriculum learning experiment already

3. with k8s and fault tolerant implemented we could run longer training.

4. with redis + mongodb 