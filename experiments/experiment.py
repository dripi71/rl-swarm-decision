from ray.rllib import SampleBatch
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import os
import numpy as np
import torch
import logging

class Experiment:
    def __init__(self):
        with open("config/configuration.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        logging.basicConfig(filename="logs/last_run.txt", filemode="w", level=logging.INFO, format='%(message)s')

    def create_env(self, env_config):
        with open("config/configuration.yaml", "r") as f:
            config = yaml.safe_load(f)
        lambdas = [0.1, 0.8]
        env = SwarmDecisionEnvironment(config, lambdas)
        return PettingZooEnv(env)

    def run(self):
        training = self.config["experiment"]["training"]

        if training:
            self.train()
        else:
            self.test()

    def train(self):
        print("Starting training with MAPPO (PPO + shared policy)")

        register_env("swarm_decision_v1", self.create_env)

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

        if self.config["experiment"]["load_checkpoint"]:
            checkpoint_path = os.path.abspath(self.config["experiment"]["test_run_name"])
            algo.restore(checkpoint_path)
            print(f"Checkpoint loaded from: {checkpoint_path}")


        training_iterations = 20

        for i in range(training_iterations):
            result = algo.train()
            reward = (result.get("env_runners", {}).get("episode_return_mean")
                        or result.get("episode_reward_mean")
                        or result.get("sampler_results", {}).get("episode_reward_mean")
                        or "N/A")
            print(f"Iteration {i+1}/{training_iterations} | Mean Episode Reward: {reward}")

        algo.stop()
        print("Training complete!")
        checkpoint_path = os.path.abspath(self.config["experiment"]["test_run_name"])
        algo.save(checkpoint_path)

    def test(self):
        print("Started policy test")
        register_env("swarm_decision_v1", self.create_env)

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
        checkpoint_path = os.path.abspath(self.config["experiment"]["test_run_name"])
        algo.restore(checkpoint_path)
        print(f"Checkpoint geladen von: {checkpoint_path}")
        env = self.create_env({}) # Eine Instanz der Env für den Test
        obs, info = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        total_reward = 0
        module = algo.get_module("shared_policy")

        print("Starte Test-Lauf...")

        # Actions have to be retrieved unneccessarily complicated over logits, because of 
        # https://github.com/ray-project/ray/issues/40312
        while not terminateds["__all__"] and not truncateds["__all__"]:
            
            agent_ids = list(obs.keys())
            obs_tensor = torch.from_numpy(np.array(list(obs.values()), dtype=np.float32))
            output = module.forward_inference({SampleBatch.OBS: obs_tensor})
            logits = output["action_dist_inputs"]

            num_loc_actions = self.config["experiment"]["num_locations"] + 1
            num_dur_actions = self.config["experiment"]["max_wait"] + 1
    
            loc_logits = logits[:, :num_loc_actions]
            dur_logits = logits[:, num_loc_actions:num_loc_actions + num_dur_actions]
    
            loc_dist = torch.distributions.Categorical(logits=loc_logits)
            dur_dist = torch.distributions.Categorical(logits=dur_logits)
    
            loc_actions = loc_dist.sample().cpu().numpy()
            dur_actions = dur_dist.sample().cpu().numpy()
    
            actions = {
                agent_id: np.array([loc, dur], dtype=np.int64)
                for agent_id, loc, dur in zip(agent_ids, loc_actions, dur_actions)
            }
    
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            logging.info(f"Actions: {actions}")
            total_reward += sum(rewards.values())      
            
            # --- Debug Ausgaben ---
            base_env = env.env.unwrapped
            logging.info(f"\n--- Step: {base_env.current_step} ---")
            agents_in_nest = len(base_env.nesting_agents)
            votes = [0 for _ in range(self.config["experiment"]["num_locations"])]
            sampling_agents = [len(base_env.sampling_agents[l]) for l in range(self.config["experiment"]["num_locations"])]


            correct_voting_agents = 0
            for agent in base_env.agent_objects:
                if agent.current_vote is not None:
                    votes[agent.current_vote] += 1
                    if agent.current_vote == np.argmin(base_env.lambdas):
                        correct_voting_agents += 1

            logging.info(f"Agents in Nest: {agents_in_nest}")
            logging.info(f"Agents in Sampling Locs: {sampling_agents}")
            logging.info(f"Votes: {votes}")
            logging.info(f"Lambdas: {base_env.lambdas}")
            logging.info(f"Correct Voting Agents: {correct_voting_agents}")
            logging.info(f"Consensus: {correct_voting_agents / base_env.config["experiment"]["num_agents"]} [Needs: {base_env.config["experiment"]["quorum_threshold"]}]")   
            
        print(f"\nTest abgeschlossen! Gesamt-Reward: {total_reward}")
        algo.stop()
        
