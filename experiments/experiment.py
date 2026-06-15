from ray.rllib import SampleBatch
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
from swarm_environment.env.baseline_environment import BaselineSimulation
import os
import numpy as np
import torch
import logging
import math
import random


class Experiment:
    def __init__(self):
        with open("config/configuration.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        logging.basicConfig(
            filename="logs/last_run.txt",
            filemode="w",
            level=logging.INFO,
            format="%(message)s",
        )
        test_run_name = self.config["experiment"]["test_run_name"]
        self.action_logger = self.setup_logger(
            name="action_logger", log_file=f"logs/actions/{test_run_name}.csv", mode="w"
        )
        self.metric_logger = self.setup_logger(
            name="metric_logger", log_file=f"logs/metrics/{test_run_name}.csv", mode="w"
        )

    def create_env(self, env_config):
        with open("config/configuration.yaml", "r") as f:
            config = yaml.safe_load(f)
        env = SwarmDecisionEnvironment(config)
        return PettingZooEnv(env)

    def run(self):

        mode = self.config["experiment"]["mode"]

        if mode == "test":
            self.test_policy_against_baseline()
        elif mode == "compare":
            self.compare_policies()
        elif mode == "baseline":
            self.run_bayes_baseline_test()
        elif mode == "train":
            self.train()
        else:
            print(
                "No valid mode selecte, choose between 'test', 'compare', 'baseline' or 'train'"
            )

    def test_policy_against_baseline(self):
        print("=" * 60)
        print("Starting RL vs DMMD Baseline Comparison Suite")
        print(f"Policy: {self.config['experiment']['test_run_name']}")
        print(f"Episodes: {self.config['experiment']['eval_iterations']}")
        print("=" * 60)

        # 1. Determine run directory: eval/runxx
        eval_dir = "eval"
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir, exist_ok=True)

        existing_runs = []
        for d in os.listdir(eval_dir):
            if d.startswith("run") and d[3:].isdigit():
                existing_runs.append(int(d[3:]))
        next_run_idx = max(existing_runs) + 1 if existing_runs else 1
        run_dir = os.path.join(eval_dir, f"run{next_run_idx:02d}")
        os.makedirs(run_dir, exist_ok=True)

        config_out_path = os.path.join(run_dir, "configuration_stamp.yaml")
        with open(config_out_path, "w") as f_cfg:
            yaml.safe_dump(self.config, f_cfg, default_flow_style=False)

        print(f"Saving comparison results and config timestamp to directory: {run_dir}")

        rl_csv_path = os.path.join(run_dir, "rl-algo.csv")
        dmmd_csv_path = os.path.join(run_dir, "dmmd-algo.csv")

        # Register environment if not already registered
        register_env("swarm_decision_v1", self.create_env)
        base_seed = self.config["experiment"]["base_seed"]
        eval_iterations = self.config["experiment"]["eval_iterations"]

        # --- Phase 1: Evaluate RL Policy ---
        print("\nEvaluating RL Policy...")
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
        checkpoint_path = os.path.abspath(
            f"policies/{self.config['experiment']['test_run_name']}"
        )
        algo.restore(checkpoint_path)
        print(f"Checkpoint loaded from: {checkpoint_path}")

        module = algo.get_module("shared_policy")

        rl_results = []
        with open(rl_csv_path, "w") as f:
            f.write(
                "seed,correct,false,truncated,steps_until_decision,events_per_agent,events_total,lambdas,votes\n"
            )
            for i in range(eval_iterations):
                episode_seed = base_seed + i
                self.set_global_seeds(episode_seed)

                env = self.create_env({})
                obs, info = env.reset(seed=episode_seed)
                terminateds = {"__all__": False}
                truncateds = {"__all__": False}
                total_reward = 0

                while not terminateds["__all__"] and not truncateds["__all__"]:
                    agent_ids = list(obs.keys())
                    obs_tensor = torch.from_numpy(
                        np.array(list(obs.values()), dtype=np.float32)
                    )
                    with torch.no_grad():
                        output = module.forward_inference({SampleBatch.OBS: obs_tensor})
                    logits = output["action_dist_inputs"]
                    num_loc_actions = self.config["experiment"]["num_locations"] + 1

                    loc_logits = logits[:, :num_loc_actions]
                    dur_means = logits[:, num_loc_actions : num_loc_actions + 2]
                    vote_logits = logits[
                        :, num_loc_actions + 4 : num_loc_actions + 4 + num_loc_actions
                    ]

                    loc_actions = torch.argmax(loc_logits, dim=1).cpu().numpy()
                    dur_params = dur_means.cpu().numpy()
                    vote_actions = torch.argmax(vote_logits, dim=1).cpu().numpy()

                    actions = {
                        agent_id: {
                            "location": np.int64(loc),
                            "duration_params": np.array(
                                [dp[0], dp[1]], dtype=np.float32
                            ),
                            "vote": np.int64(vote),
                        }
                        for agent_id, loc, dp, vote in zip(
                            agent_ids, loc_actions, dur_params, vote_actions
                        )
                    }
                    obs, rewards, terminateds, truncateds, infos = env.step(actions)
                    total_reward += sum(rewards.values())

                base_env = env.env.unwrapped
                steps_to_decision = base_env.current_step
                correct_decision = (
                    base_env.swarm_decision == base_env.experiment_best_location
                    if base_env.swarm_decision is not None
                    else False
                )
                truncated = truncateds["__all__"]
                false_decision = not correct_decision and not truncated
                total_events_until_decision = sum(
                    [sum(agent.events_at_location) for agent in base_env.agent_objects]
                )
                events_experienced_per_agent = (
                    total_events_until_decision
                    / self.config["experiment"]["num_agents"]
                )
                lambdas = [float(x) for x in base_env.lambdas]

                votes = [0] * self.config["experiment"]["num_locations"]
                for agent in base_env.agent_objects:
                    if agent.current_vote is not None:
                        if agent.current_vote < len(votes):
                            votes[agent.current_vote] += 1
                votes = [int(v) for v in votes]

                f.write(
                    f'{episode_seed},{correct_decision},{false_decision},{truncated},{steps_to_decision},{events_experienced_per_agent:.3f},{total_events_until_decision},"{lambdas}","{votes}"\n'
                )

                rl_results.append(
                    {
                        "correct": correct_decision,
                        "truncated": truncated,
                        "steps": steps_to_decision,
                        "total_events": total_events_until_decision,
                    }
                )

                if (i + 1) % 10 == 0 or (i + 1) == eval_iterations:
                    print(f"RL Episode {i + 1}/{eval_iterations} complete.")

        algo.stop()
        print("RL Policy Evaluation Complete.")

        # --- Phase 2: Evaluate DMMD Baseline ---
        print("\nEvaluating DMMD Baseline...")
        sim = BaselineSimulation(self.config)

        dmmd_results = []
        with open(dmmd_csv_path, "w") as f:
            f.write(
                "seed,correct,false,truncated,steps_until_decision,events_per_agent,events_total,lambdas,votes\n"
            )
            for i in range(eval_iterations):
                episode_seed = base_seed + i
                self.set_global_seeds(episode_seed)
                sim.reset(seed=episode_seed)

                metrics = sim.run_episode()

                steps = metrics["steps"]
                correct = metrics["correct"]
                truncated = metrics["truncated"]
                false_dec = not correct and not truncated
                total_events = metrics["total_events"]
                events_per_agent = metrics["events_per_agent"]
                lambdas = [float(x) for x in metrics["lambdas"]]
                votes = [
                    int(metrics["final_votes"].get(l, 0))
                    for l in range(self.config["experiment"]["num_locations"])
                ]

                f.write(
                    f'{episode_seed},{correct},{false_dec},{truncated},{steps},{events_per_agent:.3f},{total_events},"{lambdas}","{votes}"\n'
                )

                dmmd_results.append(
                    {
                        "correct": correct,
                        "truncated": truncated,
                        "steps": steps,
                        "total_events": total_events,
                    }
                )

                if (i + 1) % 10 == 0 or (i + 1) == eval_iterations:
                    print(f"DMMD Episode {i + 1}/{eval_iterations} complete.")

        print("DMMD Baseline Evaluation Complete.")

        # --- Print side-by-side summary table ---
        print("\n" + "=" * 60)
        print("Evaluation Comparison Summary")
        print("=" * 60)
        print(
            f"| Metric | RL ({self.config['experiment']['test_run_name']}) | DMMD Baseline |"
        )
        print("| --- | --- | --- |")

        rl_success = np.mean([r["correct"] for r in rl_results]) * 100
        dmmd_success = np.mean([r["correct"] for r in dmmd_results]) * 100
        print(f"| Success Rate | {rl_success:.1f}% | {dmmd_success:.1f}% |")

        rl_trunc = np.mean([r["truncated"] for r in rl_results]) * 100
        dmmd_trunc = np.mean([r["truncated"] for r in dmmd_results]) * 100
        print(f"| Truncation Rate | {rl_trunc:.1f}% | {dmmd_trunc:.1f}% |")

        rl_non_trunc = [r["steps"] for r in rl_results if not r["truncated"]]
        dmmd_non_trunc = [r["steps"] for r in dmmd_results if not r["truncated"]]
        rl_avg_steps = np.mean(rl_non_trunc) if rl_non_trunc else float("nan")
        dmmd_avg_steps = np.mean(dmmd_non_trunc) if dmmd_non_trunc else float("nan")
        print(f"| Avg Steps (decided) | {rl_avg_steps:.1f} | {dmmd_avg_steps:.1f} |")

        rl_avg_events = (
            np.mean([r["total_events"] for r in rl_results])
            / self.config["experiment"]["num_agents"]
        )
        dmmd_avg_events = (
            np.mean([r["total_events"] for r in dmmd_results])
            / self.config["experiment"]["num_agents"]
        )
        print(f"| Avg Events/Agent | {rl_avg_events:.1f} | {dmmd_avg_events:.1f} |")
        print("=" * 60)

        print(f"Comparison complete! CSV files written to:")
        print(f"  - RL policy: {rl_csv_path}")
        print(f"  - DMMD baseline: {dmmd_csv_path}")
        print("=" * 60)

    def train(self):
        print("Starting training with MAPPO (PPO + shared policy)")
        print("Run name: ", self.config["experiment"]["test_run_name"])
        print("=" * 60)
        print(self.config["experiment"])
        print("=")
        print(self.config["rewards"])
        print("=" * 60)

        register_env("swarm_decision_v1", self.create_env)

        policy_file_name = os.path.abspath(
            f"policies/{self.config['experiment']['test_run_name']}"
        )
        initial_train_checkpoint = (
            int(policy_file_name.split("checkpoint-")[-1])
            if "checkpoint-" in policy_file_name
            else 0
        )
        last_saved_train_checkpoint = initial_train_checkpoint

        train_seed = self.config["experiment"]["base_seed"] + initial_train_checkpoint

        steps_per_iter = 30000
        training_iterations = self.config["experiment"]["training_iterations"]
        entropy_coeff = self.config["experiment"]["entropy_coeff"]

        entropy_coeff_schedule = [
            [0, entropy_coeff],
            [int(training_iterations * 0.5 * steps_per_iter), 0.5 * entropy_coeff],
            [int(training_iterations * 0.8 * steps_per_iter), 0.1 * entropy_coeff],
        ]

        config = (
            PPOConfig()
            .debugging(seed=train_seed)
            .environment("swarm_decision_v1")
            .framework("torch")
            .training(
                train_batch_size=steps_per_iter,
                entropy_coeff=entropy_coeff_schedule,
                clip_param=self.config["experiment"]["clip_param"],
                vf_clip_param=self.config["experiment"]["vf_clip_param"],
            )
            .env_runners(num_env_runners=60)
            .multi_agent(
                policies={"shared_policy": (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
        )
        algo = config.build_algo()

        if self.config["experiment"]["load_checkpoint"]:
            algo.restore(policy_file_name)
            print(f"Checkpoint loaded from: {policy_file_name}")

        for i in range(training_iterations):
            result = algo.train()
            raw = result.get("env_runners", {}).get("episode_return_mean")
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                raw = result.get("episode_reward_mean")
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                raw = result.get("sampler_results", {}).get("episode_reward_mean")
            reward = raw if raw is not None else "episode still running..."

            # Extract learner stats (entropy & entropy_coeff)
            # New API: result["learners"]["shared_policy"]["entropy"]
            # Old API: result["info"]["learner"]["shared_policy"]["learner_stats"]["entropy"]
            entropy = "N/A"
            curr_entropy_coeff = "N/A"

            # Try new API path first (Ray 2.x with build_algo)
            learners = result.get("learners", {})
            if "shared_policy" in learners:
                entropy = learners["shared_policy"].get("entropy", "N/A")
                curr_entropy_coeff = learners["shared_policy"].get(
                    "curr_entropy_coeff", "N/A"
                )

            # Fallback: old API path
            if entropy == "N/A":
                info = result.get("info", {})
                learner_info = info.get("learner", {}).get(
                    "shared_policy", info.get("learner", {})
                )
                learner_stats = learner_info.get("learner_stats", learner_info)
                entropy = learner_stats.get("entropy", "N/A")
                curr_entropy_coeff = learner_stats.get(
                    "entropy_coeff", curr_entropy_coeff
                )

            # Debug: print result keys on first iteration to find correct path
            if i == 0:
                print(f"[DEBUG] Top-level result keys: {list(result.keys())}")
                if "learners" in result:
                    print(
                        f"[DEBUG] result['learners'] keys: {list(result['learners'].keys())}"
                    )
                    for k, v in result["learners"].items():
                        if isinstance(v, dict):
                            print(
                                f"[DEBUG] result['learners']['{k}'] keys: {list(v.keys())}"
                            )

            # Format to 4 decimal places if float
            entropy_str = (
                f"{entropy:.4f}"
                if isinstance(entropy, (float, int)) and entropy != "N/A"
                else str(entropy)
            )
            coeff_str = (
                f"{curr_entropy_coeff:.4f}"
                if isinstance(curr_entropy_coeff, (float, int))
                and curr_entropy_coeff != "N/A"
                else str(curr_entropy_coeff)
            )

            print(
                f"Iteration {i + 1}/{training_iterations} | Mean Episode Reward: {reward} | Entropy: {entropy_str} | Entropy Coeff: {coeff_str}"
            )

            if (i + 1) % 100 == 0:
                checkpoint_train_iteration = initial_train_checkpoint + i + 1
                filename = (
                    policy_file_name.split("checkpoint-")[0]
                    + "checkpoint-"
                    + str(checkpoint_train_iteration)
                )
                algo.save(filename)
                print(f"Latest save: training iteration {checkpoint_train_iteration}")

        algo.stop()
        print("Training complete!")
        checkpoint_iteration = initial_train_checkpoint + training_iterations
        policy_file_name = (
            policy_file_name.split("checkpoint-")[0]
            + "checkpoint-"
            + str(checkpoint_iteration)
        )
        algo.save(policy_file_name)

    def compare_policies(self):
        print("=" * 60)
        print("Starting Policy Comparison Suite")
        print("=" * 60)

        register_env("swarm_decision_v1", self.create_env)
        base_seed = self.config["experiment"]["base_seed"]
        eval_iterations = self.config["experiment"]["eval_iterations"]
        mode = self.config.get("compare_policies", {}).get("mode", "both")

        path_a = os.path.abspath(
            f"policies/{self.config['compare_policies']['policy_a_path']}"
        )
        path_b = os.path.abspath(
            f"policies/{self.config['compare_policies']['policy_b_path']}"
        )

        print(f"Loading Policy A from: {path_a}")
        config_a = (
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
        algo_a = config_a.build_algo()
        algo_a.restore(path_a)
        module_a = algo_a.get_module("shared_policy")

        print(f"Loading Policy B from: {path_b}")
        config_b = (
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
        algo_b = config_b.build_algo()
        algo_b.restore(path_b)
        module_b = algo_b.get_module("shared_policy")

        def get_actions(module, obs_dict, ids):
            if not ids:
                return {}
            obs_tensor = torch.from_numpy(
                np.array([obs_dict[aid] for aid in ids], dtype=np.float32)
            )
            with torch.no_grad():
                output = module.forward_inference({SampleBatch.OBS: obs_tensor})
            logits = output["action_dist_inputs"]
            num_loc_actions = self.config["experiment"]["num_locations"] + 1

            loc_logits = logits[:, :num_loc_actions]
            dur_means = logits[:, num_loc_actions : num_loc_actions + 2]
            vote_logits = logits[
                :, num_loc_actions + 4 : num_loc_actions + 4 + num_loc_actions
            ]

            loc_actions = torch.argmax(loc_logits, dim=1).cpu().numpy()
            dur_params = dur_means.cpu().numpy()
            vote_actions = torch.argmax(vote_logits, dim=1).cpu().numpy()

            return {
                aid: {
                    "location": np.int64(loc),
                    "duration_params": np.array([dp[0], dp[1]], dtype=np.float32),
                    "vote": np.int64(vote),
                }
                for aid, loc, dp, vote in zip(
                    ids, loc_actions, dur_params, vote_actions
                )
            }

        # -------------------------------------------------------------
        # Independent Evaluation Mode
        # -------------------------------------------------------------
        print("\n" + "-" * 60)
        print("Running Independent Evaluation (Side-by-Side)")
        print("-" * 60)

        results = {"Policy A": [], "Policy B": []}
        for label, module in [("Policy A", module_a), ("Policy B", module_b)]:
            print(f"Evaluating {label}...")
            for i in range(eval_iterations):
                episode_seed = base_seed + i
                self.set_global_seeds(episode_seed)

                env = self.create_env({})
                obs, info = env.reset(seed=episode_seed)
                terminateds = {"__all__": False}
                truncateds = {"__all__": False}
                total_reward = 0

                while not terminateds["__all__"] and not truncateds["__all__"]:
                    agent_ids = list(obs.keys())
                    actions = get_actions(module, obs, agent_ids)
                    obs, rewards, terminateds, truncateds, infos = env.step(actions)
                    total_reward += sum(rewards.values())

                base_env = env.env.unwrapped
                steps = base_env.current_step
                correct = base_env.swarm_decision == base_env.experiment_best_location
                truncated = truncateds["__all__"]
                total_events = sum(
                    [sum(agent.events_at_location) for agent in base_env.agent_objects]
                )

                results[label].append(
                    {
                        "steps": steps,
                        "correct": correct,
                        "truncated": truncated,
                        "total_events": total_events,
                        "reward": total_reward,
                    }
                )

            # Print Independent Results Table
            print("\n### Independent Comparison Summary ###")
            print("| Metric | Policy A | Policy B |")
            print("| --- | --- | --- |")
            for metric in [
                "Success Rate",
                "Avg Steps to Decision",
                "Truncation Rate",
                "Avg Events/Agent",
                "Avg Total Reward",
            ]:
                if metric == "Success Rate":
                    val_a = np.mean([r["correct"] for r in results["Policy A"]]) * 100
                    val_b = np.mean([r["correct"] for r in results["Policy B"]]) * 100
                    print(f"| {metric} | {val_a:.1f}% | {val_b:.1f}% |")
                elif metric == "Avg Steps to Decision":
                    # average steps of non-truncated runs
                    non_trunc_a = [
                        r["steps"] for r in results["Policy A"] if not r["truncated"]
                    ]
                    non_trunc_b = [
                        r["steps"] for r in results["Policy B"] if not r["truncated"]
                    ]
                    val_a = np.mean(non_trunc_a) if non_trunc_a else float("nan")
                    val_b = np.mean(non_trunc_b) if non_trunc_b else float("nan")
                    print(f"| {metric} | {val_a:.1f} | {val_b:.1f} |")
                elif metric == "Truncation Rate":
                    val_a = np.mean([r["truncated"] for r in results["Policy A"]]) * 100
                    val_b = np.mean([r["truncated"] for r in results["Policy B"]]) * 100
                    print(f"| {metric} | {val_a:.1f}% | {val_b:.1f}% |")
                elif metric == "Avg Events/Agent":
                    val_a = (
                        np.mean([r["total_events"] for r in results["Policy A"]])
                        / self.config["experiment"]["num_agents"]
                    )
                    val_b = (
                        np.mean([r["total_events"] for r in results["Policy B"]])
                        / self.config["experiment"]["num_agents"]
                    )
                    print(f"| {metric} | {val_a:.1f} | {val_b:.1f} |")
                elif metric == "Avg Total Reward":
                    val_a = np.mean([r["reward"] for r in results["Policy A"]])
                    val_b = np.mean([r["reward"] for r in results["Policy B"]])
                    print(f"| {metric} | {val_a:.2f} | {val_b:.2f} |")

        # Cleanup algo resources
        algo_a.stop()
        algo_b.stop()

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
        checkpoint_path = os.path.abspath(
            f"policies/{self.config['experiment']['test_run_name']}"
        )
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
                obs_tensor = torch.from_numpy(
                    np.array(list(obs.values()), dtype=np.float32)
                )
                output = module.forward_inference({SampleBatch.OBS: obs_tensor})

                # action_dist_inputs layout for Dict(location: Discrete(N+1), duration_params: Box(2,), vote: Discrete(N+1)):
                #   [:N+1]         → location logits (Categorical)
                #   [N+1:N+3]      → duration Gaussian means (mu_x1, mu_x2)
                #   [N+3:N+5]      → duration Gaussian log std devs
                #   [N+5:N+5+N+1]  → vote logits (Categorical)
                logits = output["action_dist_inputs"]
                num_loc_actions = self.config["experiment"]["num_locations"] + 1

                loc_logits = logits[:, :num_loc_actions]
                dur_means = logits[
                    :, num_loc_actions : num_loc_actions + 2
                ]  # (batch, 2)
                vote_logits = logits[
                    :, num_loc_actions + 4 : num_loc_actions + 4 + num_loc_actions
                ]

                # Deterministic location: argmax over Categorical logits
                loc_actions = torch.argmax(loc_logits, dim=1).cpu().numpy()
                # Deterministic duration: pass the raw means; environment applies softplus + Gamma
                dur_params = dur_means.cpu().numpy()  # shape (batch, 2)
                # Deterministic vote: argmax over Categorical logits
                vote_actions = torch.argmax(vote_logits, dim=1).cpu().numpy()

                actions = {
                    agent_id: {
                        "location": np.int64(loc),
                        "duration_params": np.array([dp[0], dp[1]], dtype=np.float32),
                        "vote": np.int64(vote),
                    }
                    for agent_id, loc, dp, vote in zip(
                        agent_ids, loc_actions, dur_params, vote_actions
                    )
                }
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                total_reward += sum(rewards.values())

                # --- Debug Ausgaben ---
                base_env = env.env.unwrapped
                logging.info(f"\n--- Step: {base_env.current_step} ---")
                agents_in_nest = len(base_env.nesting_agents)
                votes = [0 for _ in range(self.config["experiment"]["num_locations"])]
                sampling_agents = [
                    len(base_env.sampling_agents[l])
                    for l in range(self.config["experiment"]["num_locations"])
                ]

                waiting_time = base_env._sample_gamma_duration(dur_params[0])
                self.action_logger.info(
                    f"+ agentids, loc_actions, dur_params[1], dur_params[2], waiting_time"
                )
                self.action_logger.info(
                    f"{agent_ids[0]},{loc_actions[0]},{dur_params[0][0]:.3f},{dur_params[0][1]:.3f},{waiting_time}"
                )

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
                logging.info(
                    f"Consensus: {correct_voting_agents / base_env.config['experiment']['num_agents']} [Needs: {base_env.config['experiment']['quorum_threshold']}]"
                )

            print(f"\nTest abgeschlossen! Gesamt-Reward: {total_reward}")

            steps_to_decision = base_env.current_step
            correct_decision = (
                base_env.swarm_decision == base_env.experiment_best_location
            )
            total_events_until_decision = sum(
                [sum(agent.events_at_location) for agent in base_env.agent_objects]
            )
            events_experienced_per_agent = (
                total_events_until_decision / self.config["experiment"]["num_agents"]
            )
            lambda_difficulty = np.round(
                (np.max(base_env.lambdas) - np.min(base_env.lambdas))
                / np.max(base_env.lambdas),
                3,
            )
            truncated = truncateds["__all__"]
            lambdas = base_env.lambdas

            self.metric_logger.info(
                f"{episode_seed},{steps_to_decision},{correct_decision},{total_events_until_decision},{events_experienced_per_agent},{lambda_difficulty},{truncated},{lambdas}"
            )
            algo.stop()

    def setup_logger(self, name, log_file, mode, level=logging.INFO, fmt="%(message)s"):
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

    def run_bayes_baseline_test(self):
        """
        Runs the paper's DMMD baseline algorithm (Bayesian belief model +
        nest belief-sharing)
        """
        print("=" * 60)
        print("Starting DMMD Baseline Test (no neural network)")
        print(f"Run name : {self.config['experiment']['test_run_name']}")
        print(f"Episodes : {self.config['experiment']['eval_iterations']}")
        print("=" * 60)

        base_seed = self.config["experiment"]["base_seed"]
        eval_iterations = self.config["experiment"]["eval_iterations"]
        num_agents = self.config["experiment"]["num_agents"]

        sim = BaselineSimulation(self.config)

        results = []
        for i in range(eval_iterations):
            episode_seed = base_seed + i
            self.set_global_seeds(episode_seed)
            sim.reset(seed=episode_seed)

            metrics = sim.run_episode()

            steps = metrics["steps"]
            correct = metrics["correct"]
            truncated = metrics["truncated"]
            total_events = metrics["total_events"]
            events_per_agent = metrics["events_per_agent"]
            lambdas = metrics["lambdas"]
            lambda_difficulty = np.round(
                (max(lambdas) - min(lambdas)) / max(lambdas), 3
            )

            results.append(metrics)

            self.metric_logger.info(
                f"{episode_seed},{steps},{correct},{total_events},"
                f"{events_per_agent:.3f},{lambda_difficulty},{truncated},{lambdas}"
            )

            outcome_str = metrics["outcome"].upper()
            print(
                f"Episode {i + 1:4d}/{eval_iterations} | {outcome_str:10s} | "
                f"Steps: {int(steps):>8,} | Events/Agent: {events_per_agent:6.1f} | "
                f"Lambdas: {[f'{l:.5f}' for l in lambdas]}"
            )

        # --- Summary ---
        success_rate = np.mean([r["correct"] for r in results]) * 100
        trunc_rate = np.mean([r["truncated"] for r in results]) * 100
        decided = [r for r in results if not r["truncated"]]
        avg_steps = np.mean([r["steps"] for r in decided]) if decided else float("nan")
        avg_events = np.mean([r["events_per_agent"] for r in results])

        print("\n" + "=" * 60)
        print("DMMD Baseline Summary")
        print(f"  Success Rate      : {success_rate:.1f}%")
        print(f"  Truncation Rate   : {trunc_rate:.1f}%")
        print(f"  Avg Steps (decided): {avg_steps:,.0f}")
        print(f"  Avg Events/Agent  : {avg_events:.1f}")
        print("=" * 60)
