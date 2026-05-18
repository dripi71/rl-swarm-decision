import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import os

class Experiment:
    def __init__(self):
        with open("config/configuration.yaml", "r") as f:
            self.config = yaml.safe_load(f)

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
        for i in range(60):
            result = algo.train()
            reward = (result.get("env_runners", {}).get("episode_return_mean")
                        or result.get("episode_reward_mean")
                        or result.get("sampler_results", {}).get("episode_reward_mean")
                        or "N/A")
            print(f"Iteration {i+1}/50 | Mean Episode Reward: {reward}")

        algo.stop()
        print("Training complete!")
        checkpoint_path = os.path.abspath("First_Test_run_001")
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
        checkpoint_path = os.path.abspath("First_Test_run_001")
        algo.restore(checkpoint_path)
        print(f"Checkpoint geladen von: {checkpoint_path}")
        env = self.create_env({}) # Eine Instanz der Env für den Test
        obs, info = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        total_reward = 0

        print("Starte Test-Lauf...")
        while not terminateds["__all__"] and not truncateds["__all__"]:
            
            agent_ids = list(obs.keys())
            obs_batch = list(obs.values())

            module = algo.get_module("shared_policy")
            output = module.forward_inference({"obs": obs_batch})

            actions = {
                agent_id: action for agent_id, action in zip(agent_ids, output["actions"])
            }
            
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            print(f"Prio Q:")
            print(env.prio_Q.print())


            total_reward += sum(rewards.values())
            
        print(f"Test abgeschlossen! Gesamt-Reward: {total_reward}")
        algo.stop()
        
