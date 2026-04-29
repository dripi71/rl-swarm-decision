import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment


class Experiment:
    def __init__(self):
        with open("config/configuration.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def create_env(self, env_config):
        with open("config/configuration.yaml", "r") as f:
            config = yaml.safe_load(f)
        # Zum Testen einfach Dummy Lambdas mitgeben
        lambdas = [0.1, 0.8]
        env = SwarmDecisionEnvironment(config, lambdas)
        return PettingZooEnv(env)

    def run(self):
        training = self.config["experiment"]["training"]
        #currentModel = self.config["experiment"]["currentModel"]

        if training:
            self.train()
        else:
            self.evaluate()

    def train(self):
        print("Starting training with MAPPO")

        register_env("swarm_decision_v1", self.create_env)

        config = (
            PPOConfig()
            .environment("swarm_decision_v1")
            .framework("torch")
            .rollouts(num_rollout_workers=1)
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
            )
        )
        algo = config.build()
        for i in range(100):
            result = algo.train()
            print(f"Iteration {i}: Reward = {result['episode_reward_mean']}")

    def evaluate(self):
        pass