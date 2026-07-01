## 1. mongoDB

DB stores hierarchical structure
```bash
tasks
|__ task_id # (task_folder) # independent document, 
    |__ environments/
    |__ solutions/
    |__ verifiers/
    |__ task.toml
    |__ instruction.md
|__ task_info
    |__ rollout_group_idx_list
    |__ rollout_reward_list # None for rollout exit with error
    |__ rollout_failure_list    # 1 for rollout exit with error

rollouts    # collection
|__ rollouts_idx    # independent document, 16MB size limit per document
    |__ task_id
    |__ uid
    |__ reward
    |__ trajectories/   # store whole conversation same as CAMEL_LOG_DIR
    |__ evaluation/
        |__ test_results # dict of unit test results, unit test : 0/1, empty if exit with error
    |__ error_info/
    |__ task_results/
        |__ timings
        |__ agent_summary/
```

## 2. Redis

redis is started with basic parameter `n_traj`
`queue_size`

it will then use sampler based on task_weights from MongoDB, and queue n_tasks equal to `queue_size` and `n_traj` per task push to the queue. and it will check and continously to maintain the queue in the `queue_size`.

## 3. pipeline

1. in miles `generate` function or in areal's `arun_episode` funtion, it will have a input `data`, which now doesn't contain any task info, but just a marker for the trainer that which returned trajectories belong to the same group.
(may need a helper for redis groupset, group query. or traj level query and set)

2. the generate function will then query the redis, either in a group manner, or in traj manner, depending on which framework we use. after that, mongodb was queried with the task_id, and returned with full task dictionary, which will then be sent to `TerminalEnv` instance. and await the `TerminalEnv` to finish and return. the task and traj on redis status is changed from `queue` to `running`

3. Inside `TerminalEnv`, it will start remote runtime, depends on preconfigured runtime backend, like `k8s`, `daytona`, `modal`, `docker` etc., and run the agent, upload verifier and run verification. after it finishes with error or success, the env will return the results with reward, trajectory other related info.

4. The generate or arun_episode function will then marked the task and trajs as finished. 

5. upload to the mongodb with the results.

6. filter and return trainer side function

we want to abstract the load/update mongodb part as a abstract task_manager class, expose unified function to load/record locally or to mongodb