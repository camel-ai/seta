1. follow similar design as terminal_agent/seta_env/runtimes/slot_pool_service

   given nodes.yaml and remote servers and slots, each slot is equipped with a node manager, which handles 

   terminal_env's life cycle. and we have a local scheduler which schedule which server node to send to. now node manager are called env services. env services will be started with a fixed configuration, and can be easily reconfigured from local start services script without redployment by fanning out to different services on different nodes. also the dataset deployment follow slot pool services.

2. when proxy session is set in terminal_agent/external/areal/examples/experimental/proxy/gsm8k_grpo_proxy.py,

    a slot will be requested from a local scheduler, actually since grpo, the n_traj number of slots will be requested, find a way to send the terminal env request to the same node when possible. likely from scheduler perspective, in a fixed window time like 2min it will send the traj slot with same task_id to the same node, but also balance the nodes by relative total slot availability (/data/qijia/terminal_agent/seta_env/orchestrators/grpo_rollout.py)

3. remote env services receives the task payload, same as the terminal_env step method input arguments. maintain a built image registry with keys equal to task_id. and decides whether need to build before env start, following grpo_rollout.py. now because the env services request are sent individually not group-wise, it needs to consider recieving requests and check build sequentially, but running terminal env async. i.e., one env services check if the incoming task_id has image built, if no, the first such request will build the image and others will be waiting, if built fail then all return with None or built failure message. but this build gate is async and parallel for different task_id, if different task_id terminal env request is sent, the image check and build can be processed in paralle.

4. after the env services finishes, it returns whatever the terminal_env.step returns to local env services request. and clean up and destroy the terminal env instances.

consider the full loop, local + remote setup, running inside terminal_agent/external/areal/examples/experimental/proxy/gsm8k_grpo_proxy.py areal proxy example and terminal env local example /data/qijia/terminal_agent/scripts/areal/rl_train.py 


FYI, may need to consider frp tunnel in some limited network case, check terminal_agent/seta_env/services/frp_tunnel