# tune_parameters.py
#
# ARCHITECTURE NOTE:
#   num_env_runners=0 is mandatory in this Ray version when running inside Ray Tune.
#   Any num_env_runners > 0 causes a placement group bundle conflict (Ray bug).
#   To compensate for single-process rollouts being too slow to complete full
#   episodes, we reduce max_steps in the environment for tuning only (500k vs 8M).
#   This keeps episodes ~16x shorter so episode_reward_mean is actually reported.

import math
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import yaml
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment

# Ray Tune trial workers run in a different CWD (Ray's scratch dir on the node).
# Resolve all paths relative to this script file, not the working directory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "configuration.yaml")


def create_env(env_config):
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["experiment"].update(env_config.get("experiment_overrides", {}))
    config["rewards"].update(env_config.get("reward_overrides", {}))
    return PettingZooEnv(SwarmDecisionEnvironment(config))


def train_ppo(config):
    register_env("swarm_tune_v1", create_env)

    env_cfg = {
        # Reduce max_steps for tuning so episodes finish in single-process mode.
        # Real training uses 8e6; here we use 5e5 (~16x faster episode completion).
        "experiment_overrides": {"max_steps": 500_000},
        "reward_overrides": {
            "r_uncert_amp": config["r_uncert_amp"],
            "r_explore_time_amp": config["r_explore_time_amp"],
            "r_vote_amp": config["r_vote_amp"],
        },
    }

    ppo_cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .debugging(seed=462)
        .environment("swarm_tune_v1", env_config=env_cfg)
        .framework("torch")
        .training(
            # Smaller batch than real training (30k) to keep iterations fast
            # in single-process mode.
            train_batch_size=3000,
            entropy_coeff=config["entropy_coeff"],
            clip_param=config["clip_param"],
            vf_clip_param=10.0,
        )
        # num_env_runners=0: all rollouts in the local process.
        # This is the only value that does NOT trigger the placement group
        # bundle-index error when running inside Ray Tune on Ray 2.55.
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
    )
    algo = ppo_cfg.build_algo()

    for i in range(config["training_iterations"]):
        result = algo.train()
        reward = (
            result.get("episode_reward_mean")
            or result.get("sampler_results", {}).get("episode_reward_mean")
        )
        # Only report if a real episode completed (not NaN)
        if reward is not None and not math.isnan(float(reward)):
            tune.report({"episode_reward_mean": float(reward), "training_iteration": i + 1})

    algo.stop()


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# num_env_runners=0 → each trial only needs 1-2 CPUs (main process).
# With 64 CPUs we can run ~30 trials in parallel.
CPUS_PER_TRIAL = 2

search_space = {
    "entropy_coeff": tune.loguniform(0.005, 0.05),
    "clip_param": tune.choice([0.1, 0.2, 0.3]),
    "r_uncert_amp": tune.uniform(5.0, 40.0),
    "r_explore_time_amp": tune.uniform(0.1, 2.0),
    "r_vote_amp": tune.uniform(2.0, 20.0),
    "training_iterations": 200,
}

if __name__ == "__main__":
    ray.init(num_cpus=64)

    scheduler = ASHAScheduler(
        metric="episode_reward_mean",
        mode="max",
        max_t=200,
        grace_period=50,
        reduction_factor=2,
    )

    searcher = OptunaSearch(
        metric="episode_reward_mean",
        mode="max",
    )

    tuner = tune.Tuner(
        tune.with_resources(train_ppo, resources={"cpu": CPUS_PER_TRIAL}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=32,
        ),
    )
    results = tuner.fit()

    try:
        best = results.get_best_result("episode_reward_mean", "max", filter_nan_and_inf=True)
        print("Best config:", best.config)
        print("Best reward:", best.metrics["episode_reward_mean"])
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("All trials reported NaN — max_steps may still be too large.")
        df = results.get_dataframe()
        print(df[["trial_id", "episode_reward_mean", "training_iteration"]].to_string())
