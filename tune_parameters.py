import math
import os

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import yaml
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
from ray.rllib.algorithms.callbacks import DefaultCallbacks

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "configuration.yaml")


class SwarmMetricsCallback(DefaultCallbacks):
    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        pz_env = base_env.get_sub_environments()[env_index]
        swarm_env = pz_env.env

        correct_decision = (
            swarm_env.swarm_decision == swarm_env.experiment_best_location
            if swarm_env.swarm_decision is not None
            else False
        )

        total_events = sum(
            [sum(agent.events_at_location) for agent in swarm_env.agent_objects]
        )
        num_agents = len(swarm_env.agent_objects)
        events_per_agent = total_events / num_agents if num_agents > 0 else 0.0

        episode.custom_metrics["correct_decision"] = float(correct_decision)
        episode.custom_metrics["events_per_agent"] = float(events_per_agent)


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
        .callbacks(SwarmMetricsCallback)
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

        # Retrieve custom metrics logged by the callback
        custom_metrics = result.get("env_runners", {}).get("custom_metrics", {})
        success_rate = custom_metrics.get("correct_decision_mean")
        events_per_agent = custom_metrics.get("events_per_agent_mean")

        # Calculate a combined score to optimize: we want high success rate and few events
        # Weight success rate highly (500) and penalize events/agent (1.0)
        if success_rate is not None and events_per_agent is not None:
            tuning_score = float(success_rate) * 500.0 - float(events_per_agent)
        else:
            tuning_score = None

        # Only report if we have valid metrics
        if tuning_score is not None and not math.isnan(tuning_score):
            tune.report(
                {
                    "tuning_score": tuning_score,
                    "success_rate": float(success_rate),
                    "events_per_agent": float(events_per_agent),
                    "training_iteration": i + 1,
                }
            )

    algo.stop()


CPUS_PER_TRIAL = 1

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
        metric="tuning_score",
        mode="max",
        max_t=200,
        grace_period=50,
        reduction_factor=2,
    )

    searcher = OptunaSearch(
        metric="tuning_score",
        mode="max",
    )

    tuner = tune.Tuner(
        tune.with_resources(train_ppo, resources={"cpu": CPUS_PER_TRIAL}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=512,
        ),
    )
    results = tuner.fit()

    try:
        best = results.get_best_result("tuning_score", "max", filter_nan_and_inf=True)
        print("Best config:", best.config)
        print("Best tuning score:", best.metrics["tuning_score"])
        print("Best success rate:", best.metrics.get("success_rate"))
        print("Best events/agent:", best.metrics.get("events_per_agent"))
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("All trials reported NaN — max_steps may still be too large.")
        df = results.get_dataframe()
        print(
            df[
                [
                    "trial_id",
                    "tuning_score",
                    "success_rate",
                    "events_per_agent",
                    "training_iteration",
                ]
            ].to_string()
        )
