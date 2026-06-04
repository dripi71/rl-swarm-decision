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
import random

class Experiment:
    def __init__(self):
        with open("config/configuration.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        logging.basicConfig(filename="logs/last_run.txt", filemode="w", level=logging.INFO, format='%(message)s')
        test_run_name = self.config["experiment"]["test_run_name"]
        self.action_logger = self.setup_logger(name="action_logger", log_file=f"logs/actions/{test_run_name}.csv", mode="w")
        self.metric_logger = self.setup_logger(name="metric_logger", log_file=f"logs/metrics/{test_run_name}.csv", mode="w")

    def create_env(self, env_config):
        with open("config/configuration.yaml", "r") as f:
            config = yaml.safe_load(f)
        env = SwarmDecisionEnvironment(config)
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

        policy_file_name = os.path.abspath(self.config["experiment"]["test_run_name"])
        initial_train_checkpoint = int(policy_file_name.split("checkpoint-")[-1]) if "checkpoint-" in policy_file_name else 0
        last_saved_train_checkpoint = initial_train_checkpoint

        train_seed = self.config["experiment"]["base_seed"] + initial_train_checkpoint

        config = (
            PPOConfig()
            .debugging(seed=train_seed)
            .environment("swarm_decision_v1")
            .framework("torch")
            .training(train_batch_size=16000)
            .env_runners(num_env_runners=16)
            .multi_agent(
                policies={"shared_policy": (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
        )
        algo = config.build_algo()

        if self.config["experiment"]["load_checkpoint"]:
            algo.restore(policy_file_name)
            print(f"Checkpoint loaded from: {policy_file_name}")


        training_iterations = self.config["experiment"]["training_iterations"]

        for i in range(training_iterations):
            result = algo.train()
            reward = (result.get("env_runners", {}).get("episode_return_mean")
                        or result.get("episode_reward_mean")
                        or result.get("sampler_results", {}).get("episode_reward_mean")
                        or "episode still running...")
            print(f"Iteration {i+1}/{training_iterations} | Mean Episode Reward: {reward}")
            if (i + 1) % 50 == 0:
                checkpoint_train_iteration = initial_train_checkpoint + i + 1
                filename = policy_file_name.split("checkpoint-")[0] + "checkpoint-" + str(checkpoint_train_iteration)
                algo.save(filename)
                print(f"Latest save: training iteration {checkpoint_train_iteration}")

        algo.stop()
        print("Training complete!")
        checkpoint_iteration = initial_train_checkpoint + training_iterations
        policy_file_name = policy_file_name.split("checkpoint-")[0] + "checkpoint-" + str(checkpoint_iteration)
        algo.save(policy_file_name)

    def test(self):
        print("Started policy test")
        register_env("swarm_decision_v1", self.create_env)
        base_seed = self.config["experiment"]["base_seed"]

        config = (
            PPOConfig()
            .debugging(seed=base_seed)
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

        eval_iterations = self.config["experiment"]["eval_iterations"]
        

        for i in range(eval_iterations):
            episode_seed = base_seed + i
            self.set_global_seeds(episode_seed)

            env = self.create_env({})
            obs, info = env.reset(seed=episode_seed)
            terminateds = {"__all__": False}
            truncateds = {"__all__": False}
            total_reward = 0
            module = algo.get_module("shared_policy")

            self.action_logger.info(f"+ Starting new run")

            # Actions have to be retrieved unneccessarily complicated over logits, because of 
            # https://github.com/ray-project/ray/issues/40312
            while not terminateds["__all__"] and not truncateds["__all__"]:
                
                agent_ids = list(obs.keys())
                print("AGENT IDS: ", agent_ids)
                obs_tensor = torch.from_numpy(np.array(list(obs.values()), dtype=np.float32))
                output = module.forward_inference({SampleBatch.OBS: obs_tensor})

                # action_dist_inputs layout for Dict(location: Discrete(N+1), duration_params: Box(2,), vote: Discrete(N+1)):
                #   [:N+1]         → location logits (Categorical)
                #   [N+1:N+3]      → duration Gaussian means (mu_x1, mu_x2)
                #   [N+3:N+5]      → duration Gaussian log std devs
                #   [N+5:N+5+N+1]  → vote logits (Categorical)
                logits = output["action_dist_inputs"]
                num_loc_actions = self.config["experiment"]["num_locations"] + 1

                loc_logits  = logits[:, :num_loc_actions]
                dur_means   = logits[:, num_loc_actions : num_loc_actions + 2]  # (batch, 2)
                vote_logits = logits[:, num_loc_actions + 4 : num_loc_actions + 4 + num_loc_actions]

                # Deterministic location: argmax over Categorical logits
                loc_actions = torch.argmax(loc_logits, dim=1).cpu().numpy()
                # Deterministic duration: pass the raw means; environment applies softplus + Gamma
                dur_params  = dur_means.cpu().numpy()  # shape (batch, 2)
                # Deterministic vote: argmax over Categorical logits
                vote_actions = torch.argmax(vote_logits, dim=1).cpu().numpy()

                actions = {
                    agent_id: {
                        "location":        np.int64(loc),
                        "duration_params":  np.array([dp[0], dp[1]], dtype=np.float32),
                        "vote":            np.int64(vote),
                    }
                    for agent_id, loc, dp, vote in zip(agent_ids, loc_actions, dur_params, vote_actions)
                }
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                total_reward += sum(rewards.values())      
                
                # --- Debug Ausgaben ---
                base_env = env.env.unwrapped
                logging.info(f"\n--- Step: {base_env.current_step} ---")
                agents_in_nest = len(base_env.nesting_agents)
                votes = [0 for _ in range(self.config["experiment"]["num_locations"])]
                sampling_agents = [len(base_env.sampling_agents[l]) for l in range(self.config["experiment"]["num_locations"])]

                waiting_time = base_env._sample_gamma_duration(dur_params[0])
                self.action_logger.info(f"{agent_ids[0]},{loc_actions[0]},{dur_params[0][0]:.3f},{dur_params[0][1]:.3f},{waiting_time}")
        


                correct_voting_agents = 0
                for agent in base_env.agent_objects:
                    if agent.current_vote is not None:
                        votes[agent.current_vote] += 1
                        if agent.current_vote == base_env.experiment_best_location:
                            correct_voting_agents += 1

                logging.info(f"Agents in Nest: {agents_in_nest}")
                logging.info(f"Agents in Sampling Locs: {sampling_agents}")
                logging.info(f"Votes: {votes}")
                logging.info(f"Lambdas: {base_env.lambdas}")
                logging.info(f"Correct Voting Agents: {correct_voting_agents}")
                logging.info(f"Consensus: {correct_voting_agents / base_env.config["experiment"]["num_agents"]} [Needs: {base_env.config["experiment"]["quorum_threshold"]}]")   
                
            print(f"\nTest abgeschlossen! Gesamt-Reward: {total_reward}")
            
            steps_to_decision = base_env.current_step
            correct_decision = base_env.swarm_decision == base_env.experiment_best_location
            total_events_until_decision = sum([sum(agent.events_at_location) for agent in base_env.agent_objects])
            events_experienced_per_agent = total_events_until_decision / self.config["experiment"]["num_agents"]        
            lambda_difficulty = np.round((np.max(base_env.lambdas) - np.min(base_env.lambdas)) / np.max(base_env.lambdas), 3)
            truncated = truncateds["__all__"]
            lambdas = base_env.lambdas

            self.metric_logger.info(f"{episode_seed},{steps_to_decision},{correct_decision},{total_events_until_decision},{events_experienced_per_agent},{lambda_difficulty},{truncated},{lambdas}")
            algo.stop()
        
    def setup_logger(self, name, log_file, mode, level=logging.INFO, fmt='%(message)s'):
         handler = logging.FileHandler(log_file, mode=mode)
         handler.setFormatter(logging.Formatter(fmt))
         logger = logging.getLogger(name)
         logger.setLevel(level)
         logger.addHandler(handler)
         logger.propagate = False
         return logger
    
    def set_global_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
