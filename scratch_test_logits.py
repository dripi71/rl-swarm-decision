import yaml
import os
from experiments.experiment import Experiment
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import torch
import numpy as np
from ray.rllib import SampleBatch

exp = Experiment()
exp.config["experiment"]["test_run_name"] = "run_Easy_sampling_reward_checkpoint-600"
register_env("swarm_decision_v1", exp.create_env)

config = (
    PPOConfig()
    .environment("swarm_decision_v1")
    .framework("torch")
    .env_runners(num_env_runners=1)
    .multi_agent(
        policies={"shared_policy": (None, None, None, {})},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
)
algo = config.build_algo()
abs_path = os.path.abspath("policies/run_Easy_sampling_reward_checkpoint-600")
algo.restore(abs_path)
module = algo.get_module("shared_policy")

env = exp.create_env({})
obs, info = env.reset()

agent_ids = list(obs.keys())
obs_tensor = torch.from_numpy(np.array(list(obs.values()), dtype=np.float32))

with torch.no_grad():
    output = module.forward_inference({SampleBatch.OBS: obs_tensor})

logits = output["action_dist_inputs"]
num_loc_actions = exp.config["experiment"]["num_locations"] + 1

loc_logits  = logits[:, :num_loc_actions]
print("Initial Loc Logits (Loc 0, Loc 1, Nest):")
print(loc_logits)

algo.stop()
